#!/usr/bin/env python3
"""Repair only true placeholder targets in current expanded Phase-2 finals.

The expanded-route cache contains one clip where the specialized Luganda FT
decoder emitted only punctuation. Two independent WAXAL-300M/open-set paths
emit ``e`` for that same audio, so use that conservative non-placeholder
fallback. No other target is changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text_norm import normalize_text


def main() -> None:
    root = ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path, default=[
        root / "submission_phase2_v2_full.csv",
        root / "submission_phase2_v3_final.csv",
        root / "submission_phase2_private_swing_luo_precision.csv",
    ])
    args = ap.parse_args()
    fallback = {}
    openset = root / "outputs/next_iter/new_openset.csv"
    if openset.exists():
        for row in pd.read_csv(openset).itertuples(index=False):
            text = normalize_text(getattr(row, "openset_text", ""))
            if text and text != ".":
                fallback[str(row.ID)] = text
    changed = []
    for path in args.files:
        df = pd.read_csv(path)
        df["ID"] = df["ID"].astype(str)
        df["Target"] = df["Target"].astype(str)
        for i, row in df.iterrows():
            if normalize_text(row.Target) in {"", ".", "nan", "null", "none"}:
                new = fallback.get(row.ID)
                if not new:
                    raise SystemExit(f"No non-placeholder fallback for {row.ID} in {path}")
                df.at[i, "Target"] = new
                changed.append({"file": str(path), "ID": row.ID, "Target": new})
        df.to_csv(path, index=False)
    print({"changed": changed, "n_files": len(args.files)})


if __name__ == "__main__":
    main()
