#!/usr/bin/env python3
"""Phase-2 proxy: oracle waxal-300m vs expanded same-family multi-hyp.

Uses data/proxy_val_index.csv (200 val rows, seed 42, ach/nyn/lug/sog/mas).
Waxal-300m only — never mixes MMS-1b conf. Sequential model loads for MPS memory.

Methods:
  A) oracle_waxal     — decode with true-language waxal-300m only
  B) expand_multihyp  — language-conditioned expanded same-family candidates, max CTC conf
  B2) expand_5way     — always {ach,lug,nyn,sog,mas}, max CTC conf
  C) champion_routing — openset champion candidate sets (oracle-LID / true-lang mapped)

Deliverables under outputs/:
  phase2_proxy_expand.json
  phase2_proxy_expand_detail.csv
  phase2_proxy_expand.md
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DATA_DIR,
    FORBIDDEN_TRAIN_SPLITS,
    HF_DATASET,
    OUTPUT_DIR,
    SEED,
    TARGET_SR,
)
from src.metrics import score_by_language, score_pairs  # noqa: E402
from src.mms_infer import pick_device, transcribe_waveform  # noqa: E402
from src.text_norm import normalize_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("proxy_expand")

PROXY_INDEX = DATA_DIR / "proxy_val_index.csv"
PROXY_LANGS = ("ach", "nyn", "lug", "sog", "mas")

WAXAL300 = {
    lang: f"waxal-benchmarking/mms-300m-waxal-{lang}"
    for lang in ("ach", "lug", "nyn", "sog", "mas", "sna", "lin")
}

# Language-conditioned expanded same-family sets (task B)
EXPAND_CANDS: dict[str, list[str]] = {
    "ach": ["ach", "lug", "nyn", "sog", "mas"],
    "lug": ["lug", "nyn", "sog", "ach", "mas"],
    "nyn": ["nyn", "lug", "sog", "ach"],
    "sog": ["sog", "lug", "nyn"],
    "mas": ["mas", "lug", "sna"],  # sna if available (cached)
}

EXPAND_5WAY = ["ach", "lug", "nyn", "sog", "mas"]

# Champion openset candidate sets, mapped from true lang (oracle-LID proxy)
# portal: luo→ach|lug|sog, lug→lug|nyn|sog, else single if in waxal300
CHAMPION_CANDS: dict[str, list[str]] = {
    "ach": ["ach", "lug", "sog"],  # ach is luo-proxy domain on portal
    "lug": ["lug", "nyn", "sog"],
    "nyn": ["nyn"],
    "sog": ["sog"],
    "mas": ["mas"],
}


def set_seed(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parquet_paths_for_lang(lang: str) -> list[str]:
    """Resolve validation parquet paths for lang from proxy summary or HF cache."""
    summary = DATA_DIR / "proxy_val_index.summary.json"
    if summary.exists():
        try:
            s = json.loads(summary.read_text())
            paths = s.get("per_lang", {}).get(lang, {}).get("parquet_files") or []
            paths = [p for p in paths if Path(p).is_file()]
            if paths:
                return paths
        except Exception as e:
            logger.warning("summary parquet resolve failed: %s", e)
    # Fallback: hub snapshot
    hub_ds = HF_DATASET.replace("/", "--")
    cache_root = Path(
        os.environ.get("HF_HUB_CACHE") or (Path.home() / ".cache/huggingface/hub")
    )
    snap_root = cache_root / f"datasets--{hub_ds}" / "snapshots"
    needle = f"{lang}-validation-"
    if snap_root.is_dir():
        for snap in sorted(snap_root.iterdir(), reverse=True):
            asr_dir = snap / "data" / "ASR" / lang
            if not asr_dir.is_dir():
                continue
            found = sorted(
                str(p.resolve())
                for p in asr_dir.glob("*.parquet")
                if needle in p.name and p.resolve().is_file()
            )
            if found:
                return found
    return []


def load_proxy_clips(index_csv: Path) -> list[dict]:
    """Load HF validation audio for IDs in proxy index. Never uses test split."""
    if not index_csv.exists():
        raise SystemExit(f"Missing {index_csv}")
    idx = pd.read_csv(index_csv)
    for col in ("id", "language", "split", "transcription"):
        if col not in idx.columns:
            raise SystemExit(f"proxy index missing column {col}")
    if (idx["split"].astype(str) != "validation").any():
        raise SystemExit("proxy index must be validation-only (no test gold)")

    from datasets import Audio
    from src.dataset import _decode_audio_item

    clips: list[dict] = []
    for lang, g in idx.groupby("language"):
        lang = str(lang)
        if lang in FORBIDDEN_TRAIN_SPLITS:
            raise SystemExit(f"forbidden lang/split key {lang}")
        want_ids = set(g["id"].astype(str))
        ref_by_id = {
            str(r.id): normalize_text(r.transcription)
            for r in g.itertuples(index=False)
        }
        logger.info("Loading validation audio for %s (need %d ids)", lang, len(want_ids))
        paths = _parquet_paths_for_lang(lang)
        if paths:
            logger.info("%s: parquet %s", lang, paths)
            ds = load_dataset("parquet", data_files={"validation": paths}, split="validation")
        else:
            config = f"{lang}_asr"
            logger.info("%s: fallback HF config %s", lang, config)
            ds = load_dataset(HF_DATASET, config, split="validation")
        try:
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception:
            pass

        found = 0
        for i in tqdm(range(len(ds)), desc=f"load-{lang}"):
            ex = ds[i]
            uid = str(ex.get("id") or ex.get("ID") or "")
            if uid not in want_ids:
                continue
            audio = _decode_audio_item(ex["audio"], TARGET_SR)
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            ref = ref_by_id.get(uid) or normalize_text(
                ex.get("transcription") or ex.get("text") or ""
            )
            if not ref:
                continue
            clips.append(
                {
                    "id": uid,
                    "language": lang,
                    "array": arr,
                    "sr": sr,
                    "ref": ref,
                }
            )
            found += 1
            if found >= len(want_ids):
                break
        missing = want_ids - {c["id"] for c in clips if c["language"] == lang}
        if missing:
            logger.warning("%s: %d ids missing from HF val: %s", lang, len(missing), list(missing)[:5])
        logger.info("%s: loaded %d/%d", lang, found, len(want_ids))
    clips.sort(key=lambda c: (c["language"], c["id"]))
    return clips


def load_waxal(lang: str, device: torch.device):
    mid = WAXAL300[lang]
    logger.info("Loading %s on %s (local_files_only)", mid, device)
    try:
        proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
    except Exception as e:
        logger.warning("local load failed (%s); retry with network", e)
        proc = AutoProcessor.from_pretrained(mid)
        model = Wav2Vec2ForCTC.from_pretrained(mid)
    model.to(device).eval()
    return model, proc


def free_model(model) -> None:
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


@torch.inference_mode()
def score_all_models(
    clips: list[dict],
    model_langs: list[str],
    device: torch.device,
) -> dict[str, dict[str, tuple[str, float]]]:
    """For each model lang, decode every clip once. Sequential load.

    Returns: results[model_lang][clip_id] = (hyp, conf)
    """
    results: dict[str, dict[str, tuple[str, float]]] = {}
    for mlang in model_langs:
        if mlang not in WAXAL300:
            logger.warning("skip unknown model lang %s", mlang)
            continue
        model, proc = load_waxal(mlang, device)
        bucket: dict[str, tuple[str, float]] = {}
        for c in tqdm(clips, desc=f"decode-{mlang}"):
            hyp, conf = transcribe_waveform(
                model,
                proc,
                c["array"],
                c["sr"],
                device=device,
                return_confidence=True,
            )
            bucket[c["id"]] = (hyp, float(conf))
        results[mlang] = bucket
        free_model(model)
        del proc
        gc.collect()
        logger.info("done model %s n=%d", mlang, len(bucket))
    return results


def pick_max(
    clip_id: str,
    candidates: list[str],
    results: dict[str, dict[str, tuple[str, float]]],
) -> tuple[str, str, float]:
    best_hyp, best_lang, best_conf = ".", candidates[0], -1e9
    for lang in candidates:
        if lang not in results or clip_id not in results[lang]:
            continue
        hyp, conf = results[lang][clip_id]
        if conf > best_conf:
            best_hyp, best_lang, best_conf = hyp, lang, conf
    return best_hyp, best_lang, best_conf


def evaluate_method(
    name: str,
    clips: list[dict],
    results: dict[str, dict[str, tuple[str, float]]],
    cand_fn,
) -> tuple[dict, list[dict]]:
    detail_rows = []
    refs, hyps, langs = [], [], []
    for c in clips:
        cands = cand_fn(c["language"])
        hyp, dlang, conf = pick_max(c["id"], cands, results)
        refs.append(c["ref"])
        hyps.append(hyp)
        langs.append(c["language"])
        detail_rows.append(
            {
                "id": c["id"],
                "true_lang": c["language"],
                "method": name,
                "candidates": "|".join(cands),
                "decode_lang": dlang,
                "confidence": conf,
                "ref": c["ref"],
                "hyp": hyp,
                "route_match": int(dlang == c["language"]),
            }
        )
    metrics = score_by_language(refs, hyps, langs)
    overall = metrics["overall"]
    per_lang = {
        k: {
            "wer": v["wer"],
            "cer": v["cer"],
            "score": v["score"],
            "S": 1.0 - v["score"],
            "n": int(v["n"]),
        }
        for k, v in metrics.items()
        if k != "overall"
    }
    route_acc = float(np.mean([r["route_match"] for r in detail_rows])) if detail_rows else 0.0
    decode_mass = dict(Counter(r["decode_lang"] for r in detail_rows))
    out = {
        "method": name,
        "n": len(detail_rows),
        "wer": overall["wer"],
        "cer": overall["cer"],
        "score": overall["score"],  # 0.5*WER+0.5*CER lower better
        "S": 1.0 - overall["score"],  # higher better
        "route_acc": route_acc,
        "decode_mass": decode_mass,
        "per_lang": per_lang,
    }
    logger.info(
        "RESULT %s S=%.4f score=%.4f wer=%.3f cer=%.3f route=%.3f n=%d",
        name,
        out["S"],
        out["score"],
        out["wer"],
        out["cer"],
        out["route_acc"],
        out["n"],
    )
    return out, detail_rows


def maybe_load_champion_json() -> dict | None:
    candidates = [
        OUTPUT_DIR / "phase2_proxy_champion.json",
        OUTPUT_DIR / "phase2_champion_proxy.json",
        OUTPUT_DIR / "phase2_offline_ab.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return {"path": str(p), "data": json.loads(p.read_text())}
            except Exception:
                continue
    return None


def write_report(
    methods: list[dict],
    out_md: Path,
    upload_rec: str,
    notes: list[str],
    champion_cmp: dict | None,
) -> None:
    by = {m["method"]: m for m in methods}
    oracle = by.get("oracle_waxal")
    expand = by.get("expand_multihyp")
    expand5 = by.get("expand_5way")
    champ = by.get("champion_routing")

    lines = [
        "# Phase-2 proxy: expanded same-family multi-hyp vs oracle waxal",
        "",
        "Proxy: `data/proxy_val_index.csv` (200 validation rows, seed 42, ach/nyn/lug/sog/mas).",
        "Decode family: **waxal-300m only** (no MMS-1b conf mix). Sequential model loads.",
        "Score: official error = 0.5·WER + 0.5·CER (lower better); S = 1 − error (higher better).",
        "",
        "## Methods",
        "",
        "| method | description |",
        "|--------|-------------|",
        "| oracle_waxal | true-lang waxal-300m only |",
        "| expand_multihyp | language-conditioned expanded same-family, max CTC conf |",
        "| expand_5way | always {ach,lug,nyn,sog,mas}, max CTC conf |",
        "| champion_routing | openset champion cand sets (true-lang mapped / oracle-LID) |",
        "",
        "### expand_multihyp candidate map",
        "",
        "```",
        "ach → ach,lug,nyn,sog,mas",
        "lug → lug,nyn,sog,ach,mas",
        "nyn → nyn,lug,sog,ach",
        "sog → sog,lug,nyn",
        "mas → mas,lug,sna",
        "```",
        "",
        "### champion_routing candidate map",
        "",
        "```",
        "ach → ach,lug,sog   # portal luo multi-hyp (ach is luo-proxy)",
        "lug → lug,nyn,sog",
        "nyn → nyn",
        "sog → sog",
        "mas → mas",
        "```",
        "",
        "## Overall results",
        "",
        "| method | n | WER | CER | score↓ | S↑ | route_acc |",
        "|--------|--:|----:|----:|-------:|---:|----------:|",
    ]
    for m in sorted(methods, key=lambda x: x["score"]):
        lines.append(
            f"| {m['method']} | {m['n']} | {m['wer']:.4f} | {m['cer']:.4f} | "
            f"{m['score']:.4f} | {m['S']:.4f} | {m['route_acc']:.3f} |"
        )
    lines += ["", "## Per-language (score = 0.5 WER + 0.5 CER, lower better)", ""]

    # table per method per lang
    all_langs = sorted({lang for m in methods for lang in m["per_lang"]})
    header = "| lang | " + " | ".join(m["method"] for m in methods) + " |"
    sep = "|------|" + "|".join(["------:"] * len(methods)) + "|"
    lines.append(header)
    lines.append(sep)
    for lang in all_langs:
        cells = []
        for m in methods:
            pl = m["per_lang"].get(lang)
            if not pl:
                cells.append("—")
            else:
                cells.append(f"{pl['score']:.3f} (S={pl['S']:.3f}, n={pl['n']})")
        lines.append(f"| {lang} | " + " | ".join(cells) + " |")

    lines += ["", "## Decode mass", ""]
    for m in methods:
        lines.append(f"- **{m['method']}**: `{m['decode_mass']}`")

    lines += ["", "## Head-to-head", ""]
    if oracle and expand:
        dS = expand["S"] - oracle["S"]
        dE = oracle["score"] - expand["score"]
        lines.append(
            f"- expand_multihyp vs oracle_waxal: ΔS = {dS:+.4f}, "
            f"Δerror = {dE:+.4f} (positive ΔS / positive Δerror reduction = expand better)"
        )
        if expand["score"] < oracle["score"]:
            lines.append("- **expand beats oracle** on proxy overall error.")
        else:
            lines.append("- **expand does NOT beat oracle** on proxy overall error.")
    if oracle and expand5:
        dS = expand5["S"] - oracle["S"]
        lines.append(f"- expand_5way vs oracle_waxal: ΔS = {dS:+.4f}")
    if expand and champ:
        dS = expand["S"] - champ["S"]
        lines.append(f"- expand_multihyp vs champion_routing: ΔS = {dS:+.4f}")
    if oracle and champ:
        dS = champ["S"] - oracle["S"]
        lines.append(f"- champion_routing vs oracle_waxal: ΔS = {dS:+.4f}")

    if champion_cmp:
        lines += ["", "## External champion JSON (if present)", ""]
        lines.append(f"- path: `{champion_cmp['path']}`")
        data = champion_cmp["data"]
        # best-effort extract
        if isinstance(data, dict):
            if "best_S" in data:
                lines.append(f"- best_S: {data['best_S']}")
            if "results" in data:
                for r in data["results"]:
                    if isinstance(r, dict) and "name" in r:
                        lines.append(
                            f"  - {r.get('name')}: S={r.get('S')} score/error={r.get('error', r.get('score'))}"
                        )

    lines += ["", "## Notes", ""]
    for n in notes:
        lines.append(f"- {n}")

    lines += [
        "",
        "## Upload recommendation",
        "",
        f"**{upload_rec}**",
        "",
        "Rule: YES only if expand clearly better than oracle by a meaningful margin on proxy; else NO.",
        "",
    ]
    out_md.write_text("\n".join(lines))
    logger.info("wrote %s", out_md)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=PROXY_INDEX)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["oracle_waxal", "expand_multihyp", "expand_5way", "champion_routing"],
    )
    p.add_argument("--max-clips", type=int, default=None, help="debug: cap clips after load")
    p.add_argument("--out-json", type=Path, default=OUTPUT_DIR / "phase2_proxy_expand.json")
    p.add_argument("--out-csv", type=Path, default=OUTPUT_DIR / "phase2_proxy_expand_detail.csv")
    p.add_argument("--out-md", type=Path, default=OUTPUT_DIR / "phase2_proxy_expand.md")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    logger.info("device=%s seed=%d", device, args.seed)

    t0 = time.time()
    clips = load_proxy_clips(args.index)
    if args.max_clips:
        clips = clips[: args.max_clips]
    logger.info("clips loaded: %d  mass=%s", len(clips), dict(Counter(c["language"] for c in clips)))

    # models needed across methods
    need: set[str] = set()
    for c in clips:
        lang = c["language"]
        if "oracle_waxal" in args.methods:
            need.add(lang)
        if "expand_multihyp" in args.methods:
            need.update(EXPAND_CANDS.get(lang, [lang]))
        if "expand_5way" in args.methods:
            need.update(EXPAND_5WAY)
        if "champion_routing" in args.methods:
            need.update(CHAMPION_CANDS.get(lang, [lang]))
    # only keep available waxal models
    model_langs = [m for m in sorted(need) if m in WAXAL300]
    # verify sna for mas expand
    if "sna" in need and "sna" not in WAXAL300:
        EXPAND_CANDS["mas"] = ["mas", "lug"]
    logger.info("models to run sequentially: %s", model_langs)

    results = score_all_models(clips, model_langs, device)

    method_specs = {
        "oracle_waxal": lambda true_lang: [true_lang],
        "expand_multihyp": lambda true_lang: [
            x for x in EXPAND_CANDS.get(true_lang, [true_lang]) if x in results
        ],
        "expand_5way": lambda true_lang: [x for x in EXPAND_5WAY if x in results],
        "champion_routing": lambda true_lang: [
            x for x in CHAMPION_CANDS.get(true_lang, [true_lang]) if x in results
        ],
    }

    method_metrics: list[dict] = []
    all_detail: list[dict] = []
    for name in args.methods:
        if name not in method_specs:
            logger.warning("unknown method %s", name)
            continue
        met, det = evaluate_method(name, clips, results, method_specs[name])
        method_metrics.append(met)
        all_detail.extend(det)

    by = {m["method"]: m for m in method_metrics}
    oracle = by.get("oracle_waxal")
    expand = by.get("expand_multihyp")
    expand5 = by.get("expand_5way")
    champ = by.get("champion_routing")

    # Meaningful margin: absolute error reduction of at least ~0.01 (1 point of 0.5WER+0.5CER)
    # and expand better than oracle; also prefer expand beating champion_routing if present.
    notes = [
        "Waxal-300m only; CTC conf = mean frame logprob of argmax path (higher better).",
        "No MMS-1b / FT-v2 conf mixing.",
        f"Runtime wall_s={time.time() - t0:.1f}",
        f"Models: {model_langs}",
    ]
    upload_rec = "NO"
    if oracle and expand:
        err_red = oracle["score"] - expand["score"]
        notes.append(
            f"expand_multihyp vs oracle: Δerror={err_red:+.4f} (ΔS={expand['S'] - oracle['S']:+.4f})"
        )
        if expand5:
            notes.append(
                f"expand_5way vs oracle: Δerror={oracle['score'] - expand5['score']:+.4f}"
            )
        if champ:
            notes.append(
                f"expand_multihyp vs champion_routing: Δerror={champ['score'] - expand['score']:+.4f}"
            )
        # YES only if clearly better than oracle by meaningful margin
        meaningful = 0.01
        if err_red >= meaningful:
            upload_rec = "YES"
            notes.append(
                f"Upload YES: expand beats oracle by {err_red:.4f} >= {meaningful} absolute error."
            )
        else:
            upload_rec = "NO"
            if err_red > 0:
                notes.append(
                    f"Upload NO: expand slightly better ({err_red:.4f}) but below meaningful margin {meaningful}."
                )
            elif err_red == 0:
                notes.append("Upload NO: expand ties oracle.")
            else:
                notes.append(
                    f"Upload NO: expand worse than oracle by {-err_red:.4f}."
                )
    else:
        notes.append("Missing oracle or expand metrics; default NO upload.")

    champion_cmp = maybe_load_champion_json()
    if champion_cmp:
        notes.append(f"Compared against existing file {champion_cmp['path']}")

    payload = {
        "seed": args.seed,
        "n_clips": len(clips),
        "langs": list(PROXY_LANGS),
        "index": str(args.index),
        "device": str(device),
        "models": model_langs,
        "expand_cands": EXPAND_CANDS,
        "champion_cands": CHAMPION_CANDS,
        "methods": method_metrics,
        "upload_recommendation": upload_rec,
        "notes": notes,
        "wall_s": time.time() - t0,
        "external_champion": (
            {"path": champion_cmp["path"]} if champion_cmp else None
        ),
    }
    # pairwise deltas for convenience
    if oracle and expand:
        payload["expand_vs_oracle"] = {
            "delta_S": expand["S"] - oracle["S"],
            "delta_error": oracle["score"] - expand["score"],
            "expand_better": expand["score"] < oracle["score"],
        }
    if expand and champ:
        payload["expand_vs_champion"] = {
            "delta_S": expand["S"] - champ["S"],
            "delta_error": champ["score"] - expand["score"],
            "expand_better": expand["score"] < champ["score"],
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", args.out_json)

    pd.DataFrame(all_detail).to_csv(args.out_csv, index=False)
    logger.info("wrote %s", args.out_csv)

    write_report(method_metrics, args.out_md, upload_rec, notes, champion_cmp)

    print(json.dumps({
        "upload_recommendation": upload_rec,
        "methods": [
            {
                "method": m["method"],
                "S": round(m["S"], 4),
                "score": round(m["score"], 4),
                "wer": round(m["wer"], 4),
                "cer": round(m["cer"], 4),
            }
            for m in method_metrics
        ],
        "wall_s": round(time.time() - t0, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
