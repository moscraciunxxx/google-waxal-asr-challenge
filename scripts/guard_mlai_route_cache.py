#!/usr/bin/env python3
"""Revert obvious generation artifacts in the full MLAI Luganda route cache."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def artifact_stats(text: str, incumbent: str) -> tuple[int, int, float]:
    words = str(text).split()
    old_words = str(incumbent).split()
    trigrams = Counter(tuple(words[i : i + 3]) for i in range(len(words) - 2))
    repeated_trigrams = sum(count - 1 for count in trigrams.values() if count > 1)
    repeated_bigrams = sum(words[i] == words[i + 1] for i in range(len(words) - 1))
    ratio = len(words) / max(1, len(old_words))
    return repeated_bigrams, repeated_trigrams, ratio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--incumbent", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cache = pd.read_csv(args.cache, dtype=str, keep_default_na=False)
    incumbent = pd.read_csv(args.incumbent, dtype=str, keep_default_na=False)
    if list(cache.columns) != ["ID", "Target"] or list(incumbent.columns) != ["ID", "Target"]:
        raise RuntimeError("both inputs must have exact ID,Target columns")
    old = dict(zip(incumbent.ID, incumbent.Target))
    if set(cache.ID) - set(old):
        raise RuntimeError("cache contains IDs absent from incumbent")

    reverted = []
    out = cache.copy()
    for i, row in out.iterrows():
        dup2, dup3, ratio = artifact_stats(row.Target, old[row.ID])
        # These thresholds target unmistakable looping/truncation artifacts.
        # Ordinary repeated words and normal length variation remain untouched.
        if dup3 >= 3 or dup2 >= 3 or ratio > 1.6:
            reverted.append({"ID": row.ID, "dup2": dup2, "dup3": dup3, "length_ratio": ratio})
            out.at[i, "Target"] = old[row.ID]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print({"rows": len(out), "reverted_rows": len(reverted), "reverted": reverted})


if __name__ == "__main__":
    main()
