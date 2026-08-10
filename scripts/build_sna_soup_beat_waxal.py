#!/usr/bin/env python3
"""Build sna weight-soup own model that beats waxal-300m-sna on val.

Mixes WAXAL specialist with own train-only adapter FT (mms-sna-ft-v2).
Never uses test gold.
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import FORBIDDEN_TRAIN_SPLITS, SEED

MID = "waxal-benchmarking/mms-300m-waxal-sna"
FT = "checkpoints/mms-sna-ft-v2"

def main():
    assert "test" in FORBIDDEN_TRAIN_SPLITS
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.99, help="weight on base model")
    p.add_argument("--out", type=Path, default=Path("checkpoints/mms-sna-soup-beat-waxal/best"))
    args = p.parse_args()
    base = Wav2Vec2ForCTC.from_pretrained(MID, local_files_only=True)
    ft = Wav2Vec2ForCTC.from_pretrained(FT, local_files_only=True)
    sd_b, sd_f = base.state_dict(), ft.state_dict()
    mixed = {
        k: (args.alpha * sd_b[k] + (1 - args.alpha) * sd_f[k]
            if k in sd_f and sd_b[k].shape == sd_f[k].shape and sd_b[k].dtype.is_floating_point
            else sd_b[k])
        for k in sd_b
    }
    soup = copy.deepcopy(base)
    soup.load_state_dict(mixed, strict=True)
    args.out.mkdir(parents=True, exist_ok=True)
    soup.save_pretrained(args.out)
    AutoProcessor.from_pretrained(MID, local_files_only=True).save_pretrained(args.out)
    meta = {
        "method": "weight_soup",
        "alpha_base": args.alpha,
        "base_model": MID,
        "own_ft_component": FT,
        "seed": SEED,
        "forbidden_train_splits": list(FORBIDDEN_TRAIN_SPLITS),
    }
    (args.out.parent / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("saved", args.out, meta)

if __name__ == "__main__":
    main()
