#!/usr/bin/env python3
"""Predict a language test/val shard in parallel-friendly slices.

Usage:
  python scripts/predict_shard.py --lang sna --split test --start 0 --count 400 --out outputs/shards/sna_test_0.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import load_hf_asr_split
from src.infer import load_model, transcribe_array
from src.text_norm import normalize_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["lin", "sna", "lug"])
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=None, help="Max rows from start; default all remaining")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/whisper-waxal/best")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--audio-only", action="store_true")
    args = p.parse_args()

    if not args.checkpoint.exists():
        # fall back to smoke
        alt = ROOT / "checkpoints/whisper-waxal-smoke/best"
        args.checkpoint = alt if alt.exists() else args.checkpoint

    model, processor, device = load_model(args.checkpoint)
    # Only materialize rows we will score (avoids multi-GB full-split decode for small shards).
    need = None if args.count is None else max(0, args.start) + args.count
    ds = load_hf_asr_split(
        args.lang,
        args.split,
        max_samples=need,
        allow_test=(args.split == "test"),
    )
    n = len(ds)
    start = max(0, args.start)
    end = n if args.count is None else min(n, start + args.count)
    print(f"lang={args.lang} split={args.split} n={n} range=[{start},{end}) ckpt={args.checkpoint}", flush=True)

    rows = []
    for i in tqdm(range(start, end), desc=f"{args.lang}-{args.split}-{start}"):
        ex = ds[i]
        array = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        pred = transcribe_array(
            model, processor, array, sr, device=device, language=None, num_beams=args.num_beams
        )
        row = {
            "ID": ex["id"],
            "language": args.lang,
            "prediction": pred if pred else ".",
            "shard_start": start,
            "index": i,
        }
        if "transcription" in ex and ex["transcription"] is not None:
            row["reference"] = normalize_text(ex["transcription"])
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"WROTE {args.out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
