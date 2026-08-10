#!/usr/bin/env python3
"""Continue-FT waxal-300m-waxal-ach on WAXAL-ach train + Sunbird SALT multispeaker-ach.

Goal: beat the floor's ach decoder (waxal-ach + KenLM beam a0.2 = 0.7389 zindi on
WAXAL ach val) by adding NEW in-language data (SALT) the base model never saw.
Gate afterwards with the same recipe (FT + beam) on WAXAL ach val n=120 seed 42;
ship only if >= +0.02.

Data rules: WAXAL train split + SALT train split only; no test gold anywhere.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, TARGET_SR
from src.dataset import load_hf_asr_split
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ach_salt")

BASE = "waxal-benchmarking/mms-300m-waxal-ach"


def salt_examples(limit: int | None = None):
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("Sunbird/salt", "multispeaker-ach/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    logger.info("SALT ach train rows: %d cols=%s", len(df), list(df.columns)[:8])
    n = 0
    for _, r in df.iterrows():
        if limit and n >= limit:
            break
        aud = r.get("audio")
        txt = r.get("text") or r.get("transcription") or r.get("sentence")
        if aud is None or not txt:
            continue
        try:
            if isinstance(aud, dict) and aud.get("bytes") is not None:
                arr, sr = sf.read(io.BytesIO(aud["bytes"]), dtype="float32")
            else:
                continue
        except Exception:
            continue
        yield np.asarray(arr, dtype=np.float32), int(sr), str(txt)
        n += 1


def prep(arr, sr, text, processor, max_seconds=16.0):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)  # stereo -> mono (2-D input explodes the batch dim)
    if not (4000 <= sr <= 48000):
        return None
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    max_len = int(max_seconds * TARGET_SR)
    if arr.shape[0] > max_len:
        arr = arr[:max_len]
    if arr.shape[0] < TARGET_SR // 4:
        return None
    peak = float(np.max(np.abs(arr)) + 1e-9)
    arr = arr / peak
    bucket = int(np.ceil(arr.shape[0] / TARGET_SR)) * TARGET_SR
    if bucket > arr.shape[0]:
        arr = np.pad(arr, (0, bucket - arr.shape[0]))
    text = normalize_text(text) or "."
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    # standard Wav2Vec2CTCTokenizer: char-level, spaces -> word_delimiter natively
    ids = processor.tokenizer(text, add_special_tokens=False).input_ids
    unk = processor.tokenizer.unk_token_id
    ids = [i for i in ids if i is not None and i != unk]
    n_frames = max(1, int(inputs.input_values.shape[-1] // 320))
    if not ids or len(ids) > n_frames:
        return None
    return inputs.input_values, torch.tensor([ids], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--salt-limit", type=int, default=4000)
    ap.add_argument("--waxal-limit", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=CHECKPOINT_DIR / "waxal-ach-salt-mix-v1")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    processor = AutoProcessor.from_pretrained(BASE)
    model = Wav2Vec2ForCTC.from_pretrained(BASE).to(device)
    model.freeze_feature_encoder()
    model.train()

    logger.info("loading SALT…")
    salt = [(a, s, t) for a, s, t in salt_examples(limit=args.salt_limit)]
    logger.info("SALT usable: %d", len(salt))
    logger.info("loading WAXAL ach train…")
    wx = load_hf_asr_split("ach", "train", max_samples=args.waxal_limit)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = np.random.default_rng(42)
    losses = []
    t0 = time.time()
    step = seen = 0
    while step < args.steps and seen < args.steps * 5:
        seen += 1
        if rng.random() < 0.3 and salt:
            a, s, t = salt[int(rng.integers(len(salt)))]
        else:
            ex = wx[int(rng.integers(len(wx)))]
            a = np.asarray(ex["audio"]["array"], dtype=np.float32)
            s = int(ex["audio"].get("sampling_rate") or TARGET_SR)
            t = ex.get("transcription") or "."
        try:
            packed = prep(a, s, t, processor)
            if packed is None:
                continue
            iv, labels = packed
            if iv.numel() > 20 * TARGET_SR:  # hard belt: never feed >20s equivalent
                continue
            out = model(iv.to(device), labels=labels.to(device))
            if out.loss is None or not torch.isfinite(out.loss):
                continue
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        except RuntimeError as e:
            logger.warning("skip sample (runtime): %s", str(e)[:120])
            opt.zero_grad(set_to_none=True)
            continue
        step += 1
        losses.append(float(out.loss.item()))
        if step % 25 == 0 or step == 1:
            logger.info("step %d/%d loss=%.3f avg25=%.3f %.1fs", step, args.steps,
                        losses[-1], float(np.mean(losses[-25:])), time.time() - t0)
        if step % 1000 == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            logger.info("periodic checkpoint at step %d", step)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    (args.out / "train_meta.json").write_text(json.dumps({
        "base": BASE, "steps": args.steps, "lr": args.lr,
        "salt_n": len(salt), "waxal_n": len(wx), "mix": "50/50",
        "final_loss_avg25": float(np.mean(losses[-25:])) if losses else None,
        "wall_s": time.time() - t0, "seed": 42,
        "data": "WAXAL ach train + Sunbird/salt multispeaker-ach train (no test gold)",
    }, indent=2))
    logger.info("saved %s", args.out)


if __name__ == "__main__":
    main()
