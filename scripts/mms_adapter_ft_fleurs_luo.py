#!/usr/bin/env python3
"""First Luo acoustic FT: MMS-1B adapter.luo fine-tuned on FLEURS luo_ke TRAIN.

Trains adapter_layer + lm_head only (~2.3M params), mirroring the recipe that
produced mms-lug-ft-v3 (the shipped lug decoder). Data: google/fleurs luo_ke
train split only (open, CC-BY; never Phase-1/2 test gold).

Gate afterwards on FLEURS luo_ke validation (n=80 seed 42) with
scripts/eval_luo_ft_gate.py — promote only if it beats zero-shot 0.8551 zindi.
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
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, TARGET_SR
from src.text_norm import normalize_text

# Reuse the exact tokenizer fix + label encoding from the WAXAL adapter FT.
from scripts.mms_adapter_ft import (
    MMS_MODEL_ID,
    fix_mms_tokenizer,
    pick_device,
    set_trainable_adapters,
    text_to_ctc_labels,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mms_ft_fleurs_luo")


def decode_audio(aud) -> tuple[np.ndarray, int]:
    if isinstance(aud, dict) and aud.get("bytes") is not None:
        src = io.BytesIO(aud["bytes"])
    else:
        src = str(aud.get("path") if isinstance(aud, dict) else aud)
    arr, sr = sf.read(src, dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


def prep_example(ex, processor, max_seconds: float = 12.0):
    arr, sr = decode_audio(ex["audio"])
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    max_len = int(max_seconds * TARGET_SR)
    if arr.shape[0] > max_len:
        arr = arr[:max_len]
    if arr.shape[0] < TARGET_SR // 4:
        return None
    # Pad to 1s buckets so MPS reuses compiled kernels (avoids per-length JIT stalls)
    bucket = int(np.ceil(arr.shape[0] / TARGET_SR)) * TARGET_SR
    if bucket > arr.shape[0]:
        arr = np.pad(arr, (0, bucket - arr.shape[0]))
    text = normalize_text(ex.get("transcription") or "") or "."
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    labels = text_to_ctc_labels(processor.tokenizer, text)
    n_frames = max(1, int(inputs.input_values.shape[-1] // 320))
    if len(labels) > n_frames:
        labels = labels[:n_frames]
    if not labels:
        return None
    return inputs.input_values, torch.tensor([labels], dtype=torch.long)


@torch.no_grad()
def eval_loss(model, processor, ds, device, max_n: int = 32) -> float:
    model.eval()
    losses = []
    for i in range(min(max_n, len(ds))):
        packed = prep_example(ds[i], processor)
        if packed is None:
            continue
        iv, labels = packed
        out = model(iv.to(device), labels=labels.to(device))
        if out.loss is not None and torch.isfinite(out.loss):
            losses.append(float(out.loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=CHECKPOINT_DIR / "mms-luo-ft-fleurs-v1")
    args = ap.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base MMS on %s for lang=luo", device)
    processor = AutoProcessor.from_pretrained(MMS_MODEL_ID, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(MMS_MODEL_ID, local_files_only=True)
    fix_mms_tokenizer(processor, "luo")
    model.load_adapter("luo")
    model.to(device)

    n_train_params = set_trainable_adapters(model)
    logger.info("Trainable params: %.3fM", n_train_params / 1e6)

    train_ds = load_dataset("google/fleurs", "luo_ke", split="train")
    train_ds = train_ds.cast_column("audio", Audio(decode=False))
    val_ds = load_dataset("google/fleurs", "luo_ke", split="validation")
    val_ds = val_ds.cast_column("audio", Audio(decode=False))
    n = len(train_ds) if args.max_train is None else min(args.max_train, len(train_ds))
    logger.info("fleurs luo train=%d (using %d) val=%d steps=%d lr=%s", len(train_ds), n, len(val_ds), args.steps, args.lr)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()

    losses: list[float] = []
    t0 = time.time()
    step = 0
    seen = 0
    order = list(range(n))
    rng = np.random.default_rng(42)
    rng.shuffle(order)
    while step < args.steps and seen < args.steps * 5:
        ex = train_ds[order[seen % n]]
        seen += 1
        packed = prep_example(ex, processor)
        if packed is None:
            continue
        iv, labels = packed
        out = model(iv.to(device), labels=labels.to(device))
        loss = out.loss
        if loss is None or (not torch.isfinite(loss)):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        step += 1
        losses.append(float(loss.item()))
        if step % 10 == 0 or step == 1:
            logger.info(
                "step %d/%d loss=%.4f avg50=%.4f elapsed=%.1fs",
                step, args.steps, losses[-1], float(np.mean(losses[-50:])), time.time() - t0,
            )

    vloss = eval_loss(model, processor, val_ds, device)
    logger.info("val_loss=%.4f", vloss)

    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    try:
        torch.save(model._get_adapters(), out_dir / "adapter_layers.pt")
    except Exception as e:
        logger.warning("adapter dump failed: %s", e)

    meta = {
        "lang": "luo",
        "data": "google/fleurs luo_ke train (open, CC-BY); no test gold",
        "steps": args.steps,
        "lr": args.lr,
        "n_train_pool": n,
        "device": str(device),
        "trainable_m": n_train_params / 1e6,
        "final_train_loss": losses[-1] if losses else None,
        "avg_last50_loss": float(np.mean(losses[-50:])) if losses else None,
        "val_loss": vloss,
        "wall_seconds": time.time() - t0,
        "seed": 42,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved %s", out_dir)


if __name__ == "__main__":
    main()
