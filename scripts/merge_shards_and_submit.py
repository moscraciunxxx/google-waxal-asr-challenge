#!/usr/bin/env python3
"""Merge shard prediction CSVs → submission.csv + scratch evidence."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT, SAMPLE_SUBMISSION_CSV, TARGET_COL, TEST_CSV
from src.submission import build_submission, check_submission

SCRATCH = Path(os.environ.get("GROK_SCRATCH", ROOT / ".scratch"))


def main() -> None:
    shard_dir = ROOT / "outputs" / "shards"
    files = sorted(shard_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No shard CSVs in {shard_dir}")
    frames = [pd.read_csv(f) for f in files]
    preds = pd.concat(frames, ignore_index=True)
    # Never silently choose one of two conflicting shard predictions.
    if "ID" in preds.columns:
        dup = preds[preds["ID"].duplicated(keep=False)].copy()
        if not dup.empty:
            value_col = "prediction" if "prediction" in dup.columns else TARGET_COL
            conflicts = dup.groupby("ID")[value_col].nunique(dropna=False)
            if bool((conflicts > 1).any()):
                bad = conflicts[conflicts > 1].index.tolist()[:5]
                raise SystemExit(f"Conflicting duplicate shard IDs: {bad}")
            preds = preds.drop_duplicates(subset=["ID"], keep="first")
    out_preds = OUTPUT_DIR / "test_predictions.csv"
    preds.to_csv(out_preds, index=False)
    print(f"merged preds {len(preds)} from {len(files)} shards → {out_preds}")

    sub = build_submission(preds, sample_path=SAMPLE_SUBMISSION_CSV, out_path=PROJECT_ROOT / "submission.csv")
    shutil.copy(PROJECT_ROOT / "submission.csv", OUTPUT_DIR / "submission.csv")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROJECT_ROOT / "submission.csv", SCRATCH / "submission.csv")
    report = check_submission(PROJECT_ROOT / "submission.csv", SAMPLE_SUBMISSION_CSV)
    (SCRATCH / "submission_check.log").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    test_n = len(pd.read_csv(TEST_CSV))
    nonempty = (sub[TARGET_COL].astype(str).str.strip() != "").sum()
    print(f"test_rows={test_n} sub_rows={len(sub)} nonempty={nonempty}")
    if not report["ok"]:
        raise SystemExit(1)
    print("MERGE_OK")


if __name__ == "__main__":
    main()
