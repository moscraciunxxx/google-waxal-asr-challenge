#!/usr/bin/env python3
"""Build a BadrEx Shona overlay on the expanded Phase-2 route set.

This is an experiment only: BadrEx won the matched WAXAL validation gate but
has not yet been public-scored on Phase 2.  It changes only routed Sna rows
and never reads Phase-2 targets.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2BertForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_phase2_openset import load_wav
from src.config import TARGET_SR
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sna_badrex")

MODEL_ID = "badrex/w2v-bert-2.0-shona-asr"
ROUTES = ROOT / "outputs" / "next_iter" / "new_routes.csv"
AUDIO = ROOT / "newaudios"
BASE = ROOT / "submission_phase2_beat075_primary_lug_splitjoin.csv"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def decode(model, processor, array: np.ndarray, device: torch.device) -> str:
    inputs = processor(array, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    kwargs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    logits = model(**kwargs).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return normalize_text(processor.decode(ids)) or "."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "submission_phase2_badrex_sna_lug_splitjoin.csv")
    ap.add_argument("--hyps", type=Path, default=ROOT / "outputs/goal_2026_08_06/hyps_sna_badrex.csv")
    ap.add_argument("--reuse-hyps", action="store_true")
    args = ap.parse_args()

    routes = list(csv.DictReader(ROUTES.open()))
    ids = [r["ID"] for r in routes if r["decode_lang"] == "sna"]
    hyps: dict[str, str] = {}
    if args.reuse_hyps and args.hyps.exists():
        for row in csv.DictReader(args.hyps.open()):
            hyps[row["ID"]] = normalize_text(row.get("Target") or "") or "."

    missing = [uid for uid in ids if uid not in hyps]
    if missing:
        device = pick_device()
        logger.info("loading %s on %s; decoding %d Sna clips", MODEL_ID, device, len(missing))
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Wav2Vec2BertForCTC.from_pretrained(MODEL_ID).to(device).eval()
        for n, uid in enumerate(missing, 1):
            array, sr = load_wav(AUDIO / f"{uid}.wav")
            array = np.asarray(array, dtype=np.float32)
            if sr != TARGET_SR:
                import librosa

                array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
            peak = float(np.max(np.abs(array)) + 1e-9)
            hyps[uid] = decode(model, processor, array / peak, device)
            if n % 25 == 0 or n == len(missing):
                logger.info("decoded %d/%d", n, len(missing))
        args.hyps.parent.mkdir(parents=True, exist_ok=True)
        with args.hyps.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
            writer.writeheader()
            writer.writerows({"ID": uid, "Target": hyps[uid]} for uid in ids)
        del model, processor
        if device.type == "mps":
            torch.mps.empty_cache()

    base = list(csv.DictReader(BASE.open()))
    if len(base) != 2392:
        raise SystemExit(f"unexpected base rows: {len(base)}")
    changed = 0
    rows = []
    for row in base:
        uid = row["ID"]
        target = hyps.get(uid, row["Target"])
        if target != row["Target"]:
            changed += 1
        rows.append({"ID": uid, "Target": target})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s rows=%d changed=%d", args.out, len(rows), changed)


if __name__ == "__main__":
    main()
