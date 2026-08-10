#!/usr/bin/env python3
"""Phase-2 proxy: waxal-300m oracle vs local mms-lug-ft-v2 on lug val proxy (40).

Compares architectures honestly. CTC conf multi-hyp only reported as diagnostic;
if conf scales differ across families, do NOT recommend conf-mix for upload.

Outputs:
  outputs/phase2_proxy_ft_lug.json
  outputs/phase2_proxy_ft_lug.md
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, DATA_DIR, HF_DATASET, OUTPUT_DIR, TARGET_SR
from src.dataset import _decode_audio_item
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase2_proxy_ft_lug")

WAXAL_LUG = "waxal-benchmarking/mms-300m-waxal-lug"
FT_LUG = CHECKPOINT_DIR / "mms-lug-ft-v2"
PROXY_CSV = DATA_DIR / "proxy_val_index.csv"


def free_mem(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_proxy_lug_rows() -> pd.DataFrame:
    df = pd.read_csv(PROXY_CSV)
    sub = df[df["language"] == "lug"].copy()
    if len(sub) == 0:
        raise RuntimeError("No lug rows in proxy_val_index.csv")
    return sub.reset_index(drop=True)


def _resolve_val_parquets(lang: str) -> list[str]:
    """Prefer cached HF hub snapshot parquet shards for lang-validation."""
    import os

    hub_ds = HF_DATASET.replace("/", "--")
    cache_root = Path(
        os.environ.get("HF_HUB_CACHE") or (Path.home() / ".cache/huggingface/hub")
    )
    snap_root = cache_root / f"datasets--{hub_ds}" / "snapshots"
    needle = f"{lang}-validation-"
    if not snap_root.is_dir():
        return []
    for snap in sorted(snap_root.iterdir(), reverse=True):
        asr_dir = snap / "data" / "ASR" / lang
        if not asr_dir.is_dir():
            continue
        found = sorted(
            p for p in asr_dir.glob("*.parquet") if needle in p.name and p.resolve().is_file()
        )
        if found:
            return [str(p.resolve()) for p in found]
    return []


def load_lug_val_by_ids(ids: set[str]) -> dict[str, dict]:
    """Load validation audio for given ids via parquet + soundfile (no torchcodec)."""
    from datasets import Audio, load_dataset

    logger.info("Loading lug validation audio for %d proxy ids (soundfile path)", len(ids))
    files = _resolve_val_parquets("lug")
    if files:
        logger.info("Using %d cached parquet file(s)", len(files))
        ds = load_dataset("parquet", data_files={"validation": files}, split="validation")
    else:
        from src.dataset import load_hf_asr_split

        logger.info("Falling back to load_hf_asr_split")
        ds = load_hf_asr_split("lug", "validation", max_samples=None)

    all_ids = [str(x) for x in ds["id"]]
    positions = [i for i, uid in enumerate(all_ids) if uid in ids]
    if not positions:
        raise RuntimeError("No matching proxy ids in lug validation")
    sub = ds.select(positions)
    try:
        sub = sub.cast_column("audio", Audio(decode=False))
    except Exception:
        pass

    out: dict[str, dict] = {}
    for i in tqdm(range(len(sub)), desc="decode-lug-val"):
        ex = sub[i]
        eid = str(ex.get("id") or ex.get("ID"))
        arr_info = _decode_audio_item(ex["audio"], TARGET_SR)
        ref = normalize_text(ex.get("transcription") or ex.get("text") or "")
        out[eid] = {
            "array": np.asarray(arr_info["array"], dtype=np.float32),
            "sr": int(arr_info["sampling_rate"]),
            "ref": ref,
        }
    missing = ids - set(out)
    if missing:
        logger.warning("Missing %d proxy ids in HF val (sample): %s", len(missing), list(missing)[:5])
    return out


def model_fingerprint(model, processor, name: str, path: str) -> dict:
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    vocab = getattr(processor.tokenizer, "vocab_size", None)
    if vocab is None:
        try:
            vocab = len(processor.tokenizer)
        except Exception:
            vocab = None
    return {
        "name": name,
        "path": path,
        "architectures": list(getattr(cfg, "architectures", None) or []),
        "hidden_size": int(getattr(cfg, "hidden_size", -1)),
        "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", -1)),
        "vocab_size": int(getattr(cfg, "vocab_size", -1)),
        "tokenizer_vocab_size": vocab,
        "num_params": int(n_params),
        "family": (
            "waxal-300m"
            if getattr(cfg, "hidden_size", 0) <= 1024
            else "mms-1b-ft"
        ),
    }


def conf_stats(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    if len(a) == 0:
        return {}
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
    }


@torch.inference_mode()
def decode_rows(
    model,
    processor,
    device: torch.device,
    rows: list[dict],
    desc: str,
) -> list[dict]:
    out = []
    for r in tqdm(rows, desc=desc):
        text, conf = transcribe_waveform(
            model,
            processor,
            r["array"],
            r["sr"],
            device=device,
            return_confidence=True,
        )
        out.append(
            {
                "id": r["id"],
                "ref": r["ref"],
                "hyp": text,
                "conf": float(conf),
            }
        )
    return out


def metrics_from_rows(decoded: list[dict]) -> dict:
    refs = [d["ref"] for d in decoded]
    hyps = [d["hyp"] for d in decoded]
    sc = score_pairs(refs, hyps)
    z = 1.0 - sc["score"]
    return {
        "n": int(sc["n"]),
        "wer": float(sc["wer"]),
        "cer": float(sc["cer"]),
        "error": float(sc["score"]),
        "zindi_est": float(z),
        "S": float(z),
    }


def per_clip_oracle(wax: list[dict], ft: list[dict]) -> dict:
    """Upper bound: per-clip pick better hyp by min(0.5*wer+0.5*cer) if both exist."""
    by_w = {d["id"]: d for d in wax}
    by_f = {d["id"]: d for d in ft}
    ids = sorted(set(by_w) & set(by_f))
    hyps = []
    refs = []
    picks = {"waxal": 0, "ft": 0, "tie_waxal": 0}
    for i in ids:
        ref = by_w[i]["ref"]
        hw, hf = by_w[i]["hyp"], by_f[i]["hyp"]
        sw = score_pairs([ref], [hw])["score"]
        sf = score_pairs([ref], [hf])["score"]
        if sf < sw - 1e-12:
            hyps.append(hf)
            picks["ft"] += 1
        elif sw < sf - 1e-12:
            hyps.append(hw)
            picks["waxal"] += 1
        else:
            hyps.append(hw)
            picks["tie_waxal"] += 1
        refs.append(ref)
    sc = score_pairs(refs, hyps)
    return {
        "n": int(sc["n"]),
        "wer": float(sc["wer"]),
        "cer": float(sc["cer"]),
        "zindi_est": float(1.0 - sc["score"]),
        "picks": picks,
        "note": "oracle pick by true error — not deployable",
    }


def multi_hyp_conf(wax: list[dict], ft: list[dict]) -> dict:
    """Pick higher CTC mean-argmax-logprob across families (diagnostic only)."""
    by_w = {d["id"]: d for d in wax}
    by_f = {d["id"]: d for d in ft}
    ids = sorted(set(by_w) & set(by_f))
    hyps, refs = [], []
    picks = {"waxal": 0, "ft": 0}
    for i in ids:
        w, f = by_w[i], by_f[i]
        if f["conf"] > w["conf"]:
            hyps.append(f["hyp"])
            picks["ft"] += 1
        else:
            hyps.append(w["hyp"])
            picks["waxal"] += 1
        refs.append(w["ref"])
    sc = score_pairs(refs, hyps)
    return {
        "n": int(sc["n"]),
        "wer": float(sc["wer"]),
        "cer": float(sc["cer"]),
        "zindi_est": float(1.0 - sc["score"]),
        "picks": picks,
        "recommended_for_upload": False,
        "note": "cross-family conf pick; scales often incomparable",
    }


def conf_comparable(fp_w: dict, fp_f: dict, stats_w: dict, stats_f: dict) -> dict:
    same_family = fp_w.get("family") == fp_f.get("family")
    same_hidden = fp_w.get("hidden_size") == fp_f.get("hidden_size")
    same_layers = fp_w.get("num_hidden_layers") == fp_f.get("num_hidden_layers")
    # Heuristic: if mean conf differs by >0.05 absolute or families differ, not comparable
    mean_delta = abs(stats_w.get("mean", 0) - stats_f.get("mean", 0))
    std_ratio = (
        max(stats_w.get("std", 1e-9), 1e-9) / max(stats_f.get("std", 1e-9), 1e-9)
    )
    comparable = bool(same_family and same_hidden and same_layers and mean_delta < 0.03)
    return {
        "comparable": comparable,
        "same_family": same_family,
        "same_hidden_size": same_hidden,
        "same_num_layers": same_layers,
        "mean_conf_delta_abs": float(mean_delta),
        "std_ratio_waxal_over_ft": float(std_ratio),
        "reason": (
            "same architecture family and conf scales close"
            if comparable
            else "different families (waxal-300m vs mms-1b-ft) and/or conf scale mismatch — do not conf-mix without calibration"
        ),
    }


def write_md(payload: dict, path: Path) -> None:
    w = payload["waxal_300m_oracle"]
    f = payload["ft_v2"]
    d = payload["delta_ft_minus_waxal"]
    cc = payload["conf_comparability"]
    lines = [
        "# Phase-2 proxy: FT-v2 lug vs waxal-300m lug",
        "",
        f"**Proxy:** language=`lug`, n={payload['n']}, split=`validation`, source=`data/proxy_val_index.csv`.",
        "",
        "## Conclusion",
        "",
    ]
    if d["zindi_est"] > 1e-6:
        lines.append(
            f"**YES — FT-v2 beats waxal-300m on lug proxy.** "
            f"Δzindi_est = **{d['zindi_est']:+.4f}** "
            f"(FT {f['metrics']['zindi_est']:.4f} vs waxal {w['metrics']['zindi_est']:.4f})."
        )
    elif d["zindi_est"] < -1e-6:
        lines.append(
            f"**NO — FT-v2 does NOT beat waxal-300m on lug proxy.** "
            f"Δzindi_est = **{d['zindi_est']:+.4f}** "
            f"(FT {f['metrics']['zindi_est']:.4f} vs waxal {w['metrics']['zindi_est']:.4f})."
        )
        lines.append("")
        lines.append(
            "Honest gate: keep waxal-300m for lug-oracle / multi-hyp family; "
            "do not burn public submissions swapping lug to FT-v2 on this evidence."
        )
    else:
        lines.append(
            f"**TIE within noise** (Δzindi_est ≈ 0). Prefer waxal-300m (champion family)."
        )

    lines += [
        "",
        "## Metrics (greedy CTC, oracle language = lug)",
        "",
        "| system | zindi_est (S) | WER | CER | n |",
        "|--------|--------------:|----:|----:|--:|",
        f"| waxal-300m lug | {w['metrics']['zindi_est']:.4f} | {w['metrics']['wer']:.4f} | {w['metrics']['cer']:.4f} | {w['metrics']['n']} |",
        f"| mms-lug-ft-v2 | {f['metrics']['zindi_est']:.4f} | {f['metrics']['wer']:.4f} | {f['metrics']['cer']:.4f} | {f['metrics']['n']} |",
        f"| **Δ (FT − waxal)** | **{d['zindi_est']:+.4f}** | {d['wer']:+.4f} | {d['cer']:+.4f} | — |",
        "",
        "Score: `zindi_est = 1 - 0.5*WER - 0.5*CER` (higher better).",
        "",
        "## Architectures",
        "",
        f"| field | waxal-300m | ft-v2 |",
        f"|-------|------------|-------|",
        f"| path | `{w['fingerprint']['path']}` | `{f['fingerprint']['path']}` |",
        f"| family | {w['fingerprint']['family']} | {f['fingerprint']['family']} |",
        f"| hidden_size | {w['fingerprint']['hidden_size']} | {f['fingerprint']['hidden_size']} |",
        f"| num_hidden_layers | {w['fingerprint']['num_hidden_layers']} | {f['fingerprint']['num_hidden_layers']} |",
        f"| vocab_size | {w['fingerprint']['vocab_size']} | {f['fingerprint']['vocab_size']} |",
        f"| num_params | {w['fingerprint']['num_params']:,} | {f['fingerprint']['num_params']:,} |",
        "",
        "## CTC confidence (mean argmax log-prob)",
        "",
        f"| system | mean | std | p10 | p50 | p90 |",
        f"|--------|-----:|----:|----:|----:|----:|",
        f"| waxal-300m | {w['conf_stats']['mean']:.4f} | {w['conf_stats']['std']:.4f} | {w['conf_stats']['p10']:.4f} | {w['conf_stats']['p50']:.4f} | {w['conf_stats']['p90']:.4f} |",
        f"| ft-v2 | {f['conf_stats']['mean']:.4f} | {f['conf_stats']['std']:.4f} | {f['conf_stats']['p10']:.4f} | {f['conf_stats']['p50']:.4f} | {f['conf_stats']['p90']:.4f} |",
        "",
        f"**Conf comparable for multi-hyp mix?** **{cc['comparable']}** — {cc['reason']}",
        "",
    ]

    mh = payload.get("multi_hyp_conf_diagnostic")
    if mh:
        lines += [
            "### Multi-hyp by raw CTC conf (diagnostic only)",
            "",
            f"- zindi_est={mh['zindi_est']:.4f} WER={mh['wer']:.4f} CER={mh['cer']:.4f}",
            f"- picks: {mh['picks']}",
            f"- **recommended_for_upload: {mh['recommended_for_upload']}**",
            f"- {mh['note']}",
            "",
        ]

    oc = payload.get("oracle_pick_upper_bound")
    if oc:
        lines += [
            "### Oracle pick upper bound (not deployable)",
            "",
            f"- zindi_est={oc['zindi_est']:.4f} (picks ft={oc['picks']['ft']}, waxal={oc['picks']['waxal']}, ties→waxal={oc['picks']['tie_waxal']})",
            f"- {oc['note']}",
            "",
        ]

    lines += [
        "## Scope notes",
        "",
        "- **sna / lin** have FT-v2 checkpoints but are **not** in the Phase-2 proxy gate "
        f"(`proxy langs = {payload['proxy_langs_note']}`). Not scored here.",
        "- Checkpoint constraint: `checkpoints/mms-lug-ft-v2` only (no 1B cross-family submission mix).",
        "- Sequential, memory-safe decode on device; no submission CSV written.",
        "",
        f"Wall times: waxal={payload.get('seconds_waxal', float('nan')):.1f}s, "
        f"ft={payload.get('seconds_ft', float('nan')):.1f}s, device=`{payload.get('device')}`.",
        "",
    ]
    path.write_text("\n".join(lines))
    logger.info("WROTE %s", path)


def main() -> int:
    device = pick_device()
    logger.info("device=%s", device)

    proxy = load_proxy_lug_rows()
    ids = set(proxy["id"].astype(str))
    audio_map = load_lug_val_by_ids(ids)

    rows = []
    for _, r in proxy.iterrows():
        eid = str(r["id"])
        if eid not in audio_map:
            continue
        a = audio_map[eid]
        ref = a["ref"] or normalize_text(r.get("transcription") or "")
        if not ref:
            continue
        rows.append({"id": eid, "array": a["array"], "sr": a["sr"], "ref": ref})

    if not rows:
        raise RuntimeError("No lug proxy rows with audio+ref")
    logger.info("proxy rows ready: %d", len(rows))

    # ---- waxal-300m ----
    t0 = time.time()
    logger.info("Loading waxal %s", WAXAL_LUG)
    try:
        w_proc = AutoProcessor.from_pretrained(WAXAL_LUG, local_files_only=True)
        w_model = Wav2Vec2ForCTC.from_pretrained(WAXAL_LUG, local_files_only=True)
    except Exception:
        w_proc = AutoProcessor.from_pretrained(WAXAL_LUG)
        w_model = Wav2Vec2ForCTC.from_pretrained(WAXAL_LUG)
    w_model.to(device).eval()
    fp_w = model_fingerprint(w_model, w_proc, "waxal-300m-lug", WAXAL_LUG)
    wax_dec = decode_rows(w_model, w_proc, device, rows, "waxal-lug")
    m_w = metrics_from_rows(wax_dec)
    c_w = conf_stats([d["conf"] for d in wax_dec])
    del w_model, w_proc
    free_mem(device)
    sec_w = time.time() - t0
    logger.info("waxal done S=%.4f in %.1fs", m_w["zindi_est"], sec_w)

    # ---- FT-v2 ----
    t1 = time.time()
    if not (FT_LUG / "model.safetensors").exists() and not (FT_LUG / "pytorch_model.bin").exists():
        raise FileNotFoundError(FT_LUG)
    logger.info("Loading FT %s", FT_LUG)
    f_proc = AutoProcessor.from_pretrained(str(FT_LUG), local_files_only=True)
    f_model = Wav2Vec2ForCTC.from_pretrained(str(FT_LUG), local_files_only=True)
    try:
        from scripts.mms_adapter_ft import fix_mms_tokenizer

        fix_mms_tokenizer(f_proc, "lug")
    except Exception as e:
        logger.warning("fix_mms_tokenizer: %s", e)
        try:
            f_proc.tokenizer.set_target_lang("lug")
        except Exception as e2:
            logger.warning("set_target_lang: %s", e2)
    f_model.to(device).eval()
    fp_f = model_fingerprint(f_model, f_proc, "mms-lug-ft-v2", str(FT_LUG))
    ft_dec = decode_rows(f_model, f_proc, device, rows, "ft-v2-lug")
    m_f = metrics_from_rows(ft_dec)
    c_f = conf_stats([d["conf"] for d in ft_dec])
    del f_model, f_proc
    free_mem(device)
    sec_f = time.time() - t1
    logger.info("ft done S=%.4f in %.1fs", m_f["zindi_est"], sec_f)

    cc = conf_comparable(fp_w, fp_f, c_w, c_f)
    mh = multi_hyp_conf(wax_dec, ft_dec)
    if not cc["comparable"]:
        mh["recommended_for_upload"] = False
    else:
        # only recommend if conf mix beats both singles
        mh["recommended_for_upload"] = bool(
            mh["zindi_est"] > max(m_w["zindi_est"], m_f["zindi_est"]) + 1e-6
        )
    oc = per_clip_oracle(wax_dec, ft_dec)

    delta = {
        "zindi_est": float(m_f["zindi_est"] - m_w["zindi_est"]),
        "wer": float(m_f["wer"] - m_w["wer"]),
        "cer": float(m_f["cer"] - m_w["cer"]),
    }
    ft_beats = bool(delta["zindi_est"] > 1e-6)

    # per-clip agreement
    n_same_hyp = sum(
        1
        for a, b in zip(wax_dec, ft_dec)
        if a["id"] == b["id"] and a["hyp"] == b["hyp"]
    )
    n_shared = len(set(d["id"] for d in wax_dec) & set(d["id"] for d in ft_dec))

    payload = {
        "task": "phase2_proxy_ft_lug",
        "language": "lug",
        "n": len(rows),
        "proxy_csv": str(PROXY_CSV),
        "proxy_langs_note": "ach,nyn,lug,sog,mas (sna/lin NOT in proxy gate)",
        "device": str(device),
        "seconds_waxal": sec_w,
        "seconds_ft": sec_f,
        "ft_beats_waxal": ft_beats,
        "conclusion": (
            "YES — FT-v2 beats waxal-300m on lug proxy"
            if ft_beats
            else "NO — FT-v2 does not beat waxal-300m on lug proxy"
        ),
        "delta_ft_minus_waxal": delta,
        "waxal_300m_oracle": {
            "metrics": m_w,
            "conf_stats": c_w,
            "fingerprint": fp_w,
        },
        "ft_v2": {
            "metrics": m_f,
            "conf_stats": c_f,
            "fingerprint": fp_f,
            "checkpoint": str(FT_LUG),
        },
        "conf_comparability": cc,
        "multi_hyp_conf_diagnostic": mh,
        "oracle_pick_upper_bound": oc,
        "hyp_agreement_rate": float(n_same_hyp / max(n_shared, 1)),
        "per_clip": [
            {
                "id": a["id"],
                "ref": a["ref"],
                "waxal_hyp": a["hyp"],
                "waxal_conf": a["conf"],
                "ft_hyp": b["hyp"],
                "ft_conf": b["conf"],
            }
            for a, b in zip(wax_dec, ft_dec)
            if a["id"] == b["id"]
        ],
        "constraints": {
            "checkpoint": "checkpoints/mms-lug-ft-v2 only",
            "no_submission_csv": True,
            "no_cross_family_1b_upload_mix": True,
            "sequential_memory_safe": True,
        },
    }

    out_json = OUTPUT_DIR / "phase2_proxy_ft_lug.json"
    out_md = OUTPUT_DIR / "phase2_proxy_ft_lug.md"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Keep full per_clip in json for audit
    out_json.write_text(json.dumps(payload, indent=2))
    logger.info("WROTE %s", out_json)
    write_md(payload, out_md)

    print(
        json.dumps(
            {
                "conclusion": payload["conclusion"],
                "ft_beats_waxal": ft_beats,
                "delta_zindi": delta["zindi_est"],
                "waxal_S": m_w["zindi_est"],
                "ft_S": m_f["zindi_est"],
                "conf_comparable": cc["comparable"],
                "n": len(rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
