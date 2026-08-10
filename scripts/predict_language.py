#!/usr/bin/env python3
"""Predict full language split once (loads model+data once). Parallelize across languages externally."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_hf_asr_split
from src.infer import load_model, transcribe_array
from src.text_norm import normalize_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/whisper-waxal/best")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--device", default=None, help="cpu|mps|cuda — default auto")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    if not args.checkpoint.exists():
        args.checkpoint = ROOT / "checkpoints/whisper-waxal-smoke/best"

    import torch
    if args.device:
        device = torch.device(args.device)
        model, processor, _ = load_model(args.checkpoint, device=device)
        model.to(device)
    else:
        model, processor, device = load_model(args.checkpoint)

    ds = load_hf_asr_split(
        args.lang,
        args.split,
        max_samples=args.max_samples,
        allow_test=(args.split == "test"),
    )
    n = len(ds)
    print(f"predict lang={args.lang} split={args.split} n={n} device={device}", flush=True)

    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    for i in tqdm(range(n), desc=f"{args.lang}-{args.split}"):
        ex = ds[i]
        array = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        pred = transcribe_array(
            model, processor, array, sr, device=device, language=None, num_beams=args.num_beams
        )
        row = {"ID": ex["id"], "language": args.lang, "prediction": pred if pred else "."}
        if ex.get("transcription") is not None:
            row["reference"] = normalize_text(ex["transcription"])
        rows.append(row)
        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f"checkpoint rows={len(rows)} → {partial}", flush=True)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    if partial.exists():
        partial.unlink(missing_ok=True)
    print(f"WROTE {args.out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
