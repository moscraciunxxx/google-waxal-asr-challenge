#!/usr/bin/env python3
"""A/B WAXAL lug validation: mms-lug-ft-v3 (shipped) vs candidate checkpoints.

n=150 seed 42 (larger than the historical n=40 to cut noise; decision gate for
shipping a lug-route redecode is >= +0.01 zindi over ft-v3 on these same rows).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import pick_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lug_ab")


def prep_audio(ex) -> np.ndarray:
    arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
    sr = int(ex["audio"].get("sampling_rate") or TARGET_SR)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak


@torch.inference_mode()
def greedy(model, processor, arr, device) -> str:
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return normalize_text(processor.decode(ids)) or "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="checkpoint dirs (first is baseline)")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "next_iter" / "lug_ab.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    val = load_hf_asr_split("lug", "validation")
    idx = list(range(len(val)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]
    logger.info("lug validation total=%d using n=%d device=%s", len(val), len(idx), device)

    refs, audios = [], []
    for i in idx:
        ex = val[i]
        refs.append(normalize_text(ex.get("transcription") or "") or ".")
        audios.append(prep_audio(ex))

    results = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ck in args.ckpts:
        name = Path(ck).name
        logger.info("== %s ==", name)
        proc = AutoProcessor.from_pretrained(ck, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(ck, local_files_only=True).to(device).eval()
        hyps = []
        t0 = time.time()
        for k, arr in enumerate(audios):
            hyps.append(greedy(model, proc, arr, device))
            if (k + 1) % 30 == 0:
                logger.info("%s %d/%d %.1fs", name, k + 1, len(audios), time.time() - t0)
        sc = score_pairs(refs, hyps)
        results[name] = {"n": len(refs), **sc, "zindi": 1.0 - sc["score"]}
        logger.info("%s -> %s", name, results[name])
        args.out.write_text(json.dumps({"seed": args.seed, "results": results}, indent=2))
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
