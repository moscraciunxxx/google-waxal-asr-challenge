#!/usr/bin/env python3
"""Continued full FT of WAXALNet MMS-300m specialists on train only.

Produces own domain CTC checkpoints that can beat frozen waxal-300m on val
(same architecture, more train). Never loads test gold.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Wav2Vec2ForCTC, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, FORBIDDEN_TRAIN_SPLITS, SEED, TARGET_SR
from src.dataset import load_hf_asr_split
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mms300_continue")

WAXAL300 = {
    "lin": "waxal-benchmarking/mms-300m-waxal-lin",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate(batch, processor):
    arrays = []
    labels = []
    for ex in batch:
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        if sr != TARGET_SR:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        # cap 20s
        max_len = TARGET_SR * 20
        if arr.shape[0] > max_len:
            arr = arr[:max_len]
        arrays.append(arr)
        text = normalize_text(ex.get("transcription") or "")
        labels.append(text)
    inputs = processor(arrays, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with processor.as_target_processor() if hasattr(processor, "as_target_processor") else nullctx():
        pass
    # encode labels via tokenizer
    tok = processor.tokenizer
    label_ids = []
    for t in labels:
        ids = tok(t, return_tensors="pt").input_ids[0]
        label_ids.append(ids)
    # pad labels with -100
    max_l = max(len(x) for x in label_ids)
    lab = torch.full((len(label_ids), max_l), -100, dtype=torch.long)
    for i, ids in enumerate(label_ids):
        lab[i, : len(ids)] = ids
    inputs["labels"] = lab
    return inputs


class nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def train_lang(
    lang: str,
    max_samples: int | None,
    max_steps: int,
    lr: float,
    out_root: Path,
    seed: int,
) -> Path:
    assert "test" not in ("train",)  # train only
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device()
    mid = WAXAL300[lang]
    out = out_root / lang
    out.mkdir(parents=True, exist_ok=True)
    logger.info("continue FT %s from %s device=%s", lang, mid, device)

    try:
        processor = AutoProcessor.from_pretrained(mid, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(mid)
        model = Wav2Vec2ForCTC.from_pretrained(mid)
    model.to(device)
    model.train()

    train_ds = load_hf_asr_split(lang, "train", max_samples=max_samples)
    # map-style list for simple loader
    rows = [train_ds[i] for i in range(len(train_ds))]

    def _collate(batch):
        arrays = []
        texts = []
        for ex in batch:
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"]["sampling_rate"])
            if sr != TARGET_SR:
                import librosa

                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
            max_len = TARGET_SR * 18
            if arr.shape[0] > max_len:
                arr = arr[:max_len]
            peak = float(np.max(np.abs(arr)) + 1e-9)
            arrays.append(arr / peak)
            texts.append(normalize_text(ex.get("transcription") or ""))
        batch_in = processor(
            arrays, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        )
        # CTC labels
        with processor.as_target_processor() if hasattr(processor, "as_target_processor") else nullctx():
            label_feats = processor.tokenizer(
                texts, return_tensors="pt", padding=True
            )
        labels = label_feats.input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch_in["labels"] = labels
        return batch_in

    loader = DataLoader(rows, batch_size=2, shuffle=True, collate_fn=_collate)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(optim, 20, max_steps)
    step = 0
    model.train()
    losses = []
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out_m = model(**batch)
            loss = out_m.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            losses.append(float(loss.item()))
            step += 1
            if step % 10 == 0:
                logger.info("%s step=%d loss=%.4f", lang, step, losses[-1])
            if step >= max_steps:
                break

    best = out / "best"
    best.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best)
    processor.save_pretrained(best)
    meta = {
        "lang": lang,
        "init": mid,
        "max_steps": max_steps,
        "max_samples": max_samples,
        "lr": lr,
        "seed": seed,
        "mean_loss_last10": float(np.mean(losses[-10:])) if losses else None,
        "rule": "train split only; never test gold",
        "forbidden": list(FORBIDDEN_TRAIN_SPLITS),
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("saved %s", best)
    return best


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=["lin", "sna", "lug"])
    p.add_argument("--max-samples", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-root", type=Path, default=CHECKPOINT_DIR / "mms300-continue-legit")
    args = p.parse_args(argv)
    results = []
    for lang in args.languages:
        best = train_lang(
            lang, args.max_samples, args.max_steps, args.lr, args.out_root, args.seed
        )
        results.append({"lang": lang, "best": str(best)})
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "summary.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
