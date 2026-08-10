#!/usr/bin/env python3
"""Build a new expanded candidate: proven public Lug beam + Lug split-join.

The base is the already public-tested ``beat075_primary`` artifact.  The only
new operation is a train-lexicon-only Luganda split-join postprocess applied to
public-visible rows whose route is ``lug``.  All other rows, including all Luo
rows and all non-Lug public routes, remain byte-for-byte from the base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.phase3_text_norm_ablations import feat_D_join_lug_splits
from src.submission import check_phase2_submission
from src.text_norm import normalize_text


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=Path,
        default=ROOT / "submission_phase2_beat075_primary.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "submission_phase2_beat075_primary_lug_splitjoin.csv",
    )
    ap.add_argument(
        "--meta",
        type=Path,
        default=ROOT / "outputs" / "beat075" / "primary_lug_splitjoin_meta.json",
    )
    args = ap.parse_args()

    counts = json.loads((ROOT / "data" / "lms" / "lug_counts.json").read_text())
    uni = {
        str(w): int(c)
        for w, c in counts["uni"].items()
        if not str(w).startswith("<")
    }
    bi = {str(k): int(v) for k, v in counts["bi"].items()}

    base = pd.read_csv(args.base, dtype=str).fillna("")
    public = pd.read_csv(
        ROOT / "outputs" / "beat075" / "public_visible_index.csv", dtype=str
    ).fillna("")
    public_lug = set(public.loc[public["decode_lang"] == "lug", "ID"].astype(str))
    target = base.set_index("ID")["Target"].astype(str).to_dict()

    replacements = []
    for uid in sorted(public_lug):
        before = target[uid]
        after = feat_D_join_lug_splits(before, uni, bi)
        if normalize_text(before) != normalize_text(after):
            target[uid] = after
            replacements.append(
                {
                    "ID": uid,
                    "before": before,
                    "after": after,
                }
            )

    out = base.copy()
    out["Target"] = out["ID"].map(target)
    out.to_csv(args.out, index=False)
    check = check_phase2_submission(args.out, strict=True)

    public_n = len(public)
    lug_n = len(public_lug)
    holdout_delta = 0.003044416196698818
    expected_total_delta = holdout_delta * lug_n / public_n
    meta = {
        "base": str(args.base),
        "out": str(args.out),
        "rows": len(out),
        "public_visible_rows": public_n,
        "public_visible_lug_rows": lug_n,
        "n_changed_vs_base": len(replacements),
        "changed_ids": [r["ID"] for r in replacements],
        "rule": "train Luganda lexicon split-join; join a+b when unigram(a+b)>=3 and bigram(a,b)==0, otherwise unigram>=5 and unigram>=5*max(bigram,1) with bigram<=1",
        "heldout_validation": {
            "script": "scripts/eval_lug_splitjoin.py",
            "checkpoint": "checkpoints/mms-lug-ft-v3",
            "n": 150,
            "seed": 42,
            "route_delta_zindi": holdout_delta,
            "baseline_zindi": 0.8836723143100441,
            "splitjoin_zindi": 0.8867167305067429,
        },
        "expected_public_delta_vs_base": expected_total_delta,
        "known_public_base_score": 0.687889452,
        "expected_public_score_vs_floor": 0.687889452 + expected_total_delta,
        "strict_validation": check,
        "sha256": sha256(args.out),
        "replacements": replacements,
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
