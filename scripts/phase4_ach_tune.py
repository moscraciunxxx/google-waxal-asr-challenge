#!/usr/bin/env python3
"""Phase-4: adapter-tune MMS-1B for Acholi (ach) on HF train only; score proxy ach val.

Option A pattern from scripts/mms_adapter_ft.py:
  - train split only (never test)
  - adapter_layer + lm_head (~2.2M params)
  - seed 42

Compares on data/proxy_val_index.csv language=ach (n=40):
  1) waxal-300m-ach (champion family for Phase-2 luo→ach)
  2) mms-1b ZS adapter ach
  3) mms-ach-ft (this run)

Outputs:
  checkpoints/mms-ach-ft-v2/
  outputs/phase4_ach_tune.json
  outputs/phase4_ach_tune.md  (upload note only if Δzindi_est FT−waxal ≥ 0.01)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
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

from scripts.mms_adapter_ft import (  # noqa: E402
    MMS_MODEL_ID,
    eval_loss,
    fix_mms_tokenizer,
    pick_device,
    prep_example,
    set_trainable_adapters,
)
from src.config import CHECKPOINT_DIR, DATA_DIR, HF_DATASET, OUTPUT_DIR, TARGET_SR  # noqa: E402
from src.dataset import _decode_audio_item, load_hf_asr_split  # noqa: E402
from src.metrics import score_pairs  # noqa: E402
from src.mms_infer import transcribe_waveform  # noqa: E402
from src.text_norm import normalize_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase4_ach_tune")

WAXAL_ACH = "waxal-benchmarking/mms-300m-waxal-ach"
PROXY_CSV = DATA_DIR / "proxy_val_index.csv"
SEED = 42
UPLOAD_DELTA_THRESHOLD = 0.01


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_mem(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train_ach_adapter(
    *,
    device: str | None,
    max_train: int,
    max_val: int,
    steps: int,
    lr: float,
    max_seconds: float,
    output_dir: Path,
    skip_if_exists: bool,
) -> dict:
    """Adapter FT on ach train only. Returns train_meta."""
    output_dir = Path(output_dir)
    meta_path = output_dir / "train_meta.json"
    weights_ok = (output_dir / "model.safetensors").exists() or (
        output_dir / "pytorch_model.bin"
    ).exists()
    if skip_if_exists and weights_ok and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        logger.info("Reusing existing checkpoint %s", output_dir)
        return meta

    seed_all(SEED)
    device_t = pick_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base MMS on %s for lang=ach", device_t)
    processor = AutoProcessor.from_pretrained(MMS_MODEL_ID, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(MMS_MODEL_ID, local_files_only=True)
    fix_mms_tokenizer(processor, "ach")
    model.load_adapter("ach")
    model.to(device_t)

    n_train = set_trainable_adapters(model)
    logger.info("Trainable params: %.3fM", n_train / 1e6)
    if n_train == 0:
        raise RuntimeError("No trainable params")

    train_ds = load_hf_asr_split("ach", "train", max_samples=max_train)
    val_ds = load_hf_asr_split("ach", "validation", max_samples=max_val)
    n = len(train_ds)
    logger.info(
        "train=%d val=%d steps=%d lr=%s max_seconds=%.1f seed=%d",
        n,
        len(val_ds),
        steps,
        lr,
        max_seconds,
        SEED,
    )

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    losses: list[float] = []
    t0 = time.time()
    step = 0
    seen = 0
    while step < steps and seen < steps * 8:
        ex = train_ds[seen % n]
        seen += 1
        packed = prep_example(ex, processor, max_seconds=max_seconds)
        if packed is None:
            continue
        iv, labels = packed
        out = model(iv.to(device_t), labels=labels.to(device_t))
        loss = out.loss
        if loss is None or (not torch.isfinite(loss)):
            logger.warning("skip bad loss at attempt %d (step %d)", seen, step)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        step += 1
        losses.append(float(loss.item()))
        if step % 25 == 0 or step == 1 or step == steps:
            logger.info(
                "step %d/%d loss=%.4f avg25=%.4f elapsed=%.1fs",
                step,
                steps,
                losses[-1],
                float(np.mean(losses[-25:])),
                time.time() - t0,
            )
        if device_t.type == "mps" and step % 50 == 0:
            free_mem(device_t)

    vloss = eval_loss(model, processor, val_ds, device_t, max_n=min(32, len(val_ds)))
    logger.info("val_loss=%.4f", vloss)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    try:
        torch.save(model._get_adapters(), output_dir / "adapter_layers.pt")
    except Exception as e:
        logger.warning("adapter dump failed: %s", e)

    meta = {
        "lang": "ach",
        "steps": steps,
        "max_train": max_train,
        "max_seconds": max_seconds,
        "lr": lr,
        "seed": SEED,
        "device": str(device_t),
        "trainable_m": n_train / 1e6,
        "final_train_loss": losses[-1] if losses else None,
        "avg_last50_loss": float(np.mean(losses[-50:])) if losses else None,
        "val_loss": vloss,
        "wall_seconds": float(time.time() - t0),
        "n_successful_steps": step,
        "rule": "train split only; never test gold",
        "output": str(output_dir),
        "base_model": MMS_MODEL_ID,
        "init_adapter": "ach",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved %s in %.1fs", output_dir, time.time() - t0)

    del model, processor, opt
    free_mem(device_t)
    return meta


def _resolve_val_parquets(lang: str) -> list[str]:
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
            p
            for p in asr_dir.glob("*.parquet")
            if needle in p.name and p.resolve().is_file()
        )
        if found:
            return [str(p.resolve()) for p in found]
    return []


def load_proxy_ach_audio() -> list[dict]:
    df = pd.read_csv(PROXY_CSV)
    sub = df[df["language"] == "ach"].copy()
    if len(sub) == 0:
        raise RuntimeError("No ach rows in proxy_val_index.csv")
    ids = set(sub["id"].astype(str))
    logger.info("Loading ach validation audio for %d proxy ids", len(ids))

    from datasets import Audio, load_dataset

    files = _resolve_val_parquets("ach")
    if files:
        ds = load_dataset("parquet", data_files={"validation": files}, split="validation")
    else:
        ds = load_hf_asr_split("ach", "validation", max_samples=None)

    all_ids = [str(x) for x in ds["id"]]
    positions = [i for i, uid in enumerate(all_ids) if uid in ids]
    if not positions:
        raise RuntimeError("No matching proxy ids in ach validation")
    sub_ds = ds.select(positions)
    try:
        sub_ds = sub_ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass

    by_id: dict[str, dict] = {}
    for i in tqdm(range(len(sub_ds)), desc="decode-ach-val"):
        ex = sub_ds[i]
        eid = str(ex.get("id") or ex.get("ID"))
        arr_info = _decode_audio_item(ex["audio"], TARGET_SR)
        ref = normalize_text(ex.get("transcription") or ex.get("text") or "")
        by_id[eid] = {
            "array": np.asarray(arr_info["array"], dtype=np.float32),
            "sr": int(arr_info["sampling_rate"]),
            "ref": ref,
        }

    rows = []
    for _, r in sub.iterrows():
        eid = str(r["id"])
        if eid not in by_id:
            continue
        a = by_id[eid]
        ref = a["ref"] or normalize_text(r.get("transcription") or "")
        if not ref:
            continue
        rows.append({"id": eid, "array": a["array"], "sr": a["sr"], "ref": ref})
    logger.info("proxy ach rows ready: %d", len(rows))
    return rows


def model_fingerprint(model, processor, name: str, path: str) -> dict:
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    vocab = getattr(processor.tokenizer, "vocab_size", None)
    if vocab is None:
        try:
            vocab = len(processor.tokenizer)
        except Exception:
            vocab = None
    hidden = int(getattr(cfg, "hidden_size", -1))
    return {
        "name": name,
        "path": path,
        "architectures": list(getattr(cfg, "architectures", None) or []),
        "hidden_size": hidden,
        "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", -1)),
        "vocab_size": int(getattr(cfg, "vocab_size", -1)),
        "tokenizer_vocab_size": vocab,
        "num_params": int(n_params),
        "family": "waxal-300m" if hidden <= 1024 else "mms-1b",
    }


@torch.inference_mode()
def decode_rows(model, processor, device, rows, desc: str) -> list[dict]:
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


def load_waxal_ach(device):
    try:
        proc = AutoProcessor.from_pretrained(WAXAL_ACH, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(WAXAL_ACH, local_files_only=True)
    except Exception:
        proc = AutoProcessor.from_pretrained(WAXAL_ACH)
        model = Wav2Vec2ForCTC.from_pretrained(WAXAL_ACH)
    model.to(device).eval()
    return model, proc


def load_mms_zs_ach(device):
    proc = AutoProcessor.from_pretrained(MMS_MODEL_ID, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(MMS_MODEL_ID, local_files_only=True)
    fix_mms_tokenizer(proc, "ach")
    model.load_adapter("ach")
    model.to(device).eval()
    return model, proc


def load_ft_ach(ckpt: Path, device):
    proc = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
    try:
        fix_mms_tokenizer(proc, "ach")
    except Exception as e:
        logger.warning("fix_mms_tokenizer: %s", e)
    model.to(device).eval()
    return model, proc


def write_md(payload: dict, path: Path) -> None:
    w = payload["waxal_300m_ach"]
    zs = payload.get("mms1b_zs_ach")
    ft = payload.get("mms_ach_ft")
    d = payload.get("delta_ft_minus_waxal") or {}
    dz = float(d.get("zindi_est", float("nan")))
    upload = bool(payload.get("upload_relevant", False))

    lines = [
        "# Phase-4: Acholi (ach) adapter tune vs waxal-ach",
        "",
        f"**Proxy:** language=`ach`, n={payload['n']}, split=`validation`, "
        f"source=`data/proxy_val_index.csv`, seed={payload.get('seed', SEED)}.",
        "",
        f"**Train rule:** HF `ach` **train only** (never test). Checkpoint: `{payload.get('checkpoint', '')}`.",
        "",
        "## Upload relevance",
        "",
    ]
    if upload:
        lines.append(
            f"**YES — upload-relevant on ach proxy.** "
            f"Δzindi_est (FT − waxal) = **{dz:+.4f}** ≥ {UPLOAD_DELTA_THRESHOLD:.2f}."
        )
        lines.append("")
        lines.append(
            "Candidate use: Phase-2 luo mass (~785) multi-hyp / ach branch if FT is "
            "wired carefully (same-family or calibrated; do not raw conf-mix 1B vs 300m)."
        )
    else:
        if np.isnan(dz):
            lines.append(
                "**NO upload gate** — FT metrics unavailable (training incomplete or skipped)."
            )
        else:
            lines.append(
                f"**NO — not upload-relevant on ach proxy alone.** "
                f"Δzindi_est (FT − waxal) = **{dz:+.4f}** "
                f"(threshold ≥ {UPLOAD_DELTA_THRESHOLD:.2f})."
            )
        lines.append("")
        lines.append(
            "Keep champion waxal-300m multi-hyp for luo→ach|lug|sog until a ≥0.01 "
            "proxy lift is shown (or a safe calibrated hybrid)."
        )

    lines += [
        "",
        "## Metrics (greedy CTC, oracle language = ach)",
        "",
        "| system | zindi_est (S) | WER | CER | n |",
        "|--------|--------------:|----:|----:|--:|",
        f"| waxal-300m-ach | {w['metrics']['zindi_est']:.4f} | "
        f"{w['metrics']['wer']:.4f} | {w['metrics']['cer']:.4f} | {w['metrics']['n']} |",
    ]
    if zs:
        lines.append(
            f"| mms-1b ZS ach | {zs['metrics']['zindi_est']:.4f} | "
            f"{zs['metrics']['wer']:.4f} | {zs['metrics']['cer']:.4f} | {zs['metrics']['n']} |"
        )
    if ft:
        lines.append(
            f"| mms-ach-ft | {ft['metrics']['zindi_est']:.4f} | "
            f"{ft['metrics']['wer']:.4f} | {ft['metrics']['cer']:.4f} | {ft['metrics']['n']} |"
        )
    if d:
        lines.append(
            f"| **Δ (FT − waxal)** | **{d['zindi_est']:+.4f}** | "
            f"{d['wer']:+.4f} | {d['cer']:+.4f} | — |"
        )
    lines += [
        "",
        "Score: `zindi_est = 1 - 0.5*WER - 0.5*CER` (higher better).",
        "",
        "## Training",
        "",
    ]
    tm = payload.get("train_meta") or {}
    if tm:
        lines += [
            f"- base: `{tm.get('base_model', MMS_MODEL_ID)}` + adapter `ach`",
            f"- steps: {tm.get('steps')} | max_train: {tm.get('max_train')} | "
            f"lr: {tm.get('lr')} | max_seconds: {tm.get('max_seconds')}",
            f"- device: `{tm.get('device')}` | wall: {tm.get('wall_seconds', float('nan')):.1f}s",
            f"- final_train_loss: {tm.get('final_train_loss')} | "
            f"avg_last50: {tm.get('avg_last50_loss')} | val_loss: {tm.get('val_loss')}",
            f"- trainable: {tm.get('trainable_m')}M params (adapter + lm_head)",
            "",
        ]
    else:
        lines.append("- (no train_meta — eval-only / ZS comparison)")
        lines.append("")

    lines += [
        "## Why ach matters for Phase-2",
        "",
        "- Phase-2 LID mass: **~785 luo** → multi-hyp among ach/lug/sog (ach is primary luo proxy).",
        "- High ROI if ach quality improves without conf-sink vs waxal family.",
        "",
        f"Wall decode times: {payload.get('seconds_decode', {})}; device=`{payload.get('device')}`.",
        "",
    ]
    path.write_text("\n".join(lines))
    logger.info("WROTE %s", path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None, help="mps|cpu|cuda")
    p.add_argument("--max-train", type=int, default=4000)
    p.add_argument("--max-val", type=int, default=64)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-seconds", type=float, default=12.0)
    p.add_argument(
        "--out-dir",
        default=str(CHECKPOINT_DIR / "mms-ach-ft-v2"),
        help="checkpoint directory",
    )
    p.add_argument("--skip-train", action="store_true", help="eval only if ckpt exists")
    p.add_argument(
        "--reuse-ckpt",
        action="store_true",
        help="skip training when checkpoint already present",
    )
    p.add_argument("--skip-zs", action="store_true", help="skip mms-1b zero-shot decode")
    args = p.parse_args()

    seed_all(SEED)
    device = pick_device(args.device)
    logger.info("device=%s seed=%d", device, SEED)
    ckpt = Path(args.out_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_meta = None
    if not args.skip_train:
        try:
            train_meta = train_ach_adapter(
                device=args.device,
                max_train=args.max_train,
                max_val=args.max_val,
                steps=args.steps,
                lr=args.lr,
                max_seconds=args.max_seconds,
                output_dir=ckpt,
                skip_if_exists=args.reuse_ckpt,
            )
        except Exception as e:
            logger.exception("Training failed: %s — continuing with ZS/waxal score if possible", e)
            train_meta = {"error": str(e), "partial": True}

    rows = load_proxy_ach_audio()
    if not rows:
        raise RuntimeError("No proxy ach audio rows")

    seconds_decode: dict[str, float] = {}
    payload: dict = {
        "task": "phase4_ach_tune",
        "language": "ach",
        "n": len(rows),
        "proxy_csv": str(PROXY_CSV),
        "seed": SEED,
        "device": str(device),
        "checkpoint": str(ckpt),
        "train_meta": train_meta,
        "rule": "train split only; never test gold",
        "phase2_note": "~785 LID=luo mapped to ach multi-hyp — ach quality is high ROI",
    }

    # ---- waxal-300m ach ----
    t0 = time.time()
    logger.info("Scoring waxal %s", WAXAL_ACH)
    w_model, w_proc = load_waxal_ach(device)
    fp_w = model_fingerprint(w_model, w_proc, "waxal-300m-ach", WAXAL_ACH)
    wax_dec = decode_rows(w_model, w_proc, device, rows, "waxal-ach")
    m_w = metrics_from_rows(wax_dec)
    c_w = conf_stats([d["conf"] for d in wax_dec])
    del w_model, w_proc
    free_mem(device)
    seconds_decode["waxal"] = time.time() - t0
    payload["waxal_300m_ach"] = {
        "metrics": m_w,
        "conf_stats": c_w,
        "fingerprint": fp_w,
    }
    logger.info("waxal zindi_est=%.4f in %.1fs", m_w["zindi_est"], seconds_decode["waxal"])

    # ---- mms-1b ZS ach ----
    if not args.skip_zs:
        t1 = time.time()
        logger.info("Scoring mms-1b ZS ach")
        z_model, z_proc = load_mms_zs_ach(device)
        fp_z = model_fingerprint(z_model, z_proc, "mms-1b-zs-ach", MMS_MODEL_ID)
        zs_dec = decode_rows(z_model, z_proc, device, rows, "mms1b-zs-ach")
        m_z = metrics_from_rows(zs_dec)
        c_z = conf_stats([d["conf"] for d in zs_dec])
        del z_model, z_proc
        free_mem(device)
        seconds_decode["mms1b_zs"] = time.time() - t1
        payload["mms1b_zs_ach"] = {
            "metrics": m_z,
            "conf_stats": c_z,
            "fingerprint": fp_z,
        }
        logger.info(
            "zs zindi_est=%.4f in %.1fs", m_z["zindi_est"], seconds_decode["mms1b_zs"]
        )
        payload["delta_zs_minus_waxal"] = {
            "zindi_est": float(m_z["zindi_est"] - m_w["zindi_est"]),
            "wer": float(m_z["wer"] - m_w["wer"]),
            "cer": float(m_z["cer"] - m_w["cer"]),
        }

    # ---- FT ach ----
    weights_ok = (ckpt / "model.safetensors").exists() or (
        ckpt / "pytorch_model.bin"
    ).exists()
    if weights_ok:
        t2 = time.time()
        logger.info("Scoring FT %s", ckpt)
        f_model, f_proc = load_ft_ach(ckpt, device)
        fp_f = model_fingerprint(f_model, f_proc, "mms-ach-ft-v2", str(ckpt))
        ft_dec = decode_rows(f_model, f_proc, device, rows, "mms-ach-ft")
        m_f = metrics_from_rows(ft_dec)
        c_f = conf_stats([d["conf"] for d in ft_dec])
        del f_model, f_proc
        free_mem(device)
        seconds_decode["ft"] = time.time() - t2
        payload["mms_ach_ft"] = {
            "metrics": m_f,
            "conf_stats": c_f,
            "fingerprint": fp_f,
            "checkpoint": str(ckpt),
        }
        delta = {
            "zindi_est": float(m_f["zindi_est"] - m_w["zindi_est"]),
            "wer": float(m_f["wer"] - m_w["wer"]),
            "cer": float(m_f["cer"] - m_w["cer"]),
        }
        payload["delta_ft_minus_waxal"] = delta
        payload["ft_beats_waxal"] = bool(delta["zindi_est"] > 1e-6)
        payload["upload_relevant"] = bool(delta["zindi_est"] >= UPLOAD_DELTA_THRESHOLD)
        logger.info(
            "ft zindi_est=%.4f Δ=%.4f upload_relevant=%s",
            m_f["zindi_est"],
            delta["zindi_est"],
            payload["upload_relevant"],
        )
        # light per-id detail for debugging
        detail = []
        by_w = {d["id"]: d for d in wax_dec}
        for d in ft_dec:
            wr = by_w.get(d["id"])
            if not wr:
                continue
            detail.append(
                {
                    "id": d["id"],
                    "ref": d["ref"],
                    "hyp_waxal": wr["hyp"],
                    "hyp_ft": d["hyp"],
                    "conf_waxal": wr["conf"],
                    "conf_ft": d["conf"],
                }
            )
        payload["n_detail"] = len(detail)
        # keep sample only (full detail in separate csv if needed)
        payload["detail_sample"] = detail[:5]
    else:
        logger.warning("No FT checkpoint at %s — reporting ZS vs waxal only", ckpt)
        payload["mms_ach_ft"] = None
        payload["delta_ft_minus_waxal"] = None
        payload["ft_beats_waxal"] = False
        payload["upload_relevant"] = False
        payload["partial"] = True

    payload["seconds_decode"] = seconds_decode
    out_json = OUTPUT_DIR / "phase4_ach_tune.json"
    out_md = OUTPUT_DIR / "phase4_ach_tune.md"
    out_json.write_text(json.dumps(payload, indent=2))
    write_md(payload, out_md)
    logger.info("WROTE %s", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
