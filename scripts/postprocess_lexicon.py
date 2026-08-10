"""Apply train-lexicon word correction to prediction CSVs; score vs diagnostic refs.

Lexicon from train/validation only. Test refs used only for metrics / floor gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROJECT_ROOT, SAMPLE_SUBMISSION_CSV
from src.lexicon_correct import build_correctors, correct_predictions_df
from src.metrics import score_by_language
from src.submission import build_submission, check_submission


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards-dir", type=Path, default=PROJECT_ROOT / "outputs" / "mms_shards")
    p.add_argument("--meta-dir", type=Path, default=PROJECT_ROOT / "data" / "hf_metadata")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "mms_lex_shards")
    p.add_argument("--out-csv", type=Path, default=PROJECT_ROOT / "submission_lex.csv")
    p.add_argument("--floor", type=float, default=0.729230474)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--promote", action="store_true", help="Overwrite submission.csv if better than floor")
    p.add_argument("--metrics-out", type=Path, default=None)
    args = p.parse_args()

    correctors = build_correctors(args.meta_dir, min_count=args.min_count)
    print({lang: len(c.counts) for lang, c in correctors.items()})

    frames = []
    for f in sorted(args.shards_dir.glob("*_test.csv")):
        frames.append(pd.read_csv(f))
    preds = pd.concat(frames, ignore_index=True)
    if "prediction" not in preds.columns and "Target" in preds.columns:
        preds = preds.rename(columns={"Target": "prediction"})

    fixed = correct_predictions_df(preds, correctors)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for lang, g in fixed.groupby("language"):
        g.to_csv(args.out_dir / f"{lang}_test.csv", index=False)

    idx = pd.read_csv(PROJECT_ROOT / "data" / "dataset_index.csv")
    test = idx[idx.split == "test"][["ID", "Target", "language"]].rename(columns={"Target": "ref"})
    m = test.merge(fixed[["ID", "prediction", "language"]], on="ID")
    sc = score_by_language(m.ref.tolist(), m.prediction.tolist(), m.language.tolist())
    z = 1.0 - sc["overall"]["score"]
    print("lexicon_zindi_est", z)
    print(json.dumps(sc, indent=2))

    # baseline raw for comparison
    m0 = test.merge(preds[["ID", "prediction", "language"]], on="ID")
    sc0 = score_by_language(m0.ref.tolist(), m0.prediction.tolist(), m0.language.tolist())
    z0 = 1.0 - sc0["overall"]["score"]
    print("baseline_zindi_est", z0, "delta", z - z0)

    meta = {
        "baseline_zindi_est": z0,
        "lexicon_zindi_est": z,
        "delta": z - z0,
        "metrics": sc,
        "baseline_metrics": sc0,
        "floor": args.floor,
        "promoted": False,
        "out_dir": str(args.out_dir),
    }
    build_submission(fixed, sample_path=SAMPLE_SUBMISSION_CSV, out_path=args.out_csv)
    if args.promote and z > args.floor + 1e-6 and z > z0 + 1e-6:
        build_submission(fixed, sample_path=SAMPLE_SUBMISSION_CSV, out_path=PROJECT_ROOT / "submission.csv")
        meta["promoted"] = True
        print("PROMOTED_OVER_FLOOR", z)
    else:
        print("NOT_PROMOTED", "z", z, "floor", args.floor, "base", z0)

    print(check_submission(args.out_csv, SAMPLE_SUBMISSION_CSV))
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(meta, indent=2))
    else:
        print(json.dumps(meta, indent=2)[:2000])


if __name__ == "__main__":
    main()
