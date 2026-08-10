"""Build Zindi-compatible submission.csv aligned to SampleSubmission.csv."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import (
    ID_COL,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SAMPLE_SUBMISSION_CSV,
    TARGET_COL,
    TEST_CSV,
    PHASE2_NEW_AUDIO_DIR,
    PHASE2_SAMPLE_SUBMISSION_CSV,
)
from src.text_norm import normalize_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("submission")


def build_submission(
    predictions: pd.DataFrame,
    sample_path: Path | None = None,
    test_path: Path | None = None,
    out_path: Path | None = None,
    id_col: str = ID_COL,
    pred_col: str = "prediction",
    target_col: str = TARGET_COL,
    strict: bool = False,
) -> pd.DataFrame:
    """Align predictions to SampleSubmission (or Test) IDs; ensure non-empty strings."""
    sample_path = Path(sample_path or SAMPLE_SUBMISSION_CSV)
    test_path = Path(test_path or TEST_CSV)

    if sample_path.exists():
        template = pd.read_csv(sample_path)
        if id_col not in template.columns:
            # tolerate lowercase
            if "id" in template.columns:
                template = template.rename(columns={"id": id_col})
            else:
                raise ValueError(f"{sample_path} missing {id_col}")
    elif test_path.exists():
        template = pd.read_csv(test_path)
        if id_col not in template.columns and "id" in template.columns:
            template = template.rename(columns={"id": id_col})
        if target_col not in template.columns:
            template[target_col] = ""
    else:
        # Fall back to prediction IDs alone
        template = predictions[[id_col]].copy() if id_col in predictions.columns else predictions[["ID"]].rename(columns={"ID": id_col})
        template[target_col] = ""

    preds = predictions.copy()
    if id_col not in preds.columns:
        if "ID" in preds.columns:
            preds = preds.rename(columns={"ID": id_col})
        elif "id" in preds.columns:
            preds = preds.rename(columns={"id": id_col})
        else:
            raise ValueError("predictions missing ID column")
    if pred_col not in preds.columns:
        if target_col in preds.columns:
            pred_col = target_col
        else:
            raise ValueError(f"predictions missing '{pred_col}' column")

    preds[id_col] = preds[id_col].astype(str)
    template[id_col] = template[id_col].astype(str)

    merged = template[[id_col]].merge(
        preds[[id_col, pred_col]],
        on=id_col,
        how="left",
        validate="one_to_one",
    )
    missing = int(merged[pred_col].isna().sum())
    if strict and missing:
        missing_ids = merged.loc[merged[pred_col].isna(), id_col].astype(str).tolist()
        raise ValueError(f"Missing predictions for {missing} IDs; first={missing_ids[:5]}")
    if missing:
        logger.warning("%d IDs missing predictions — filling with '.' placeholder", missing)
    # Non-empty after strip so Zindi/structural checks accept the cell
    _EMPTY_PLACEHOLDER = "."
    texts = []
    for v in merged[pred_col].tolist():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            if strict:
                raise ValueError("NaN/missing prediction encountered in strict mode")
            texts.append(_EMPTY_PLACEHOLDER)
        else:
            t = normalize_text(str(v))
            if strict and (not t or t in {".", "nan", "null", "none"}):
                raise ValueError(f"Invalid placeholder prediction in strict mode: {v!r}")
            texts.append(t if t else _EMPTY_PLACEHOLDER)
    out = pd.DataFrame({id_col: merged[id_col], target_col: texts})

    # Structural guarantees
    if len(out) != len(template):
        raise RuntimeError(f"Row count mismatch: submission {len(out)} vs template {len(template)}")
    if out[target_col].map(lambda x: not str(x).strip()).any():
        out.loc[out[target_col].map(lambda x: not str(x).strip()), target_col] = _EMPTY_PLACEHOLDER

    out_path = Path(out_path or (OUTPUT_DIR / "submission.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Wrote submission %s rows=%d", out_path, len(out))
    return out


def check_submission(
    submission_path: Path,
    sample_path: Path | None = None,
    *,
    strict: bool = False,
    compare_template: bool = True,
) -> dict:
    """Structural checks for Zindi upload readiness."""
    sub = pd.read_csv(submission_path)
    sample_path = Path(sample_path or SAMPLE_SUBMISSION_CSV)
    result = {
        "path": str(submission_path),
        "n_rows": len(sub),
        "columns": list(sub.columns),
        "ok": True,
        "errors": [],
    }
    if strict and list(sub.columns) != [ID_COL, TARGET_COL]:
        result["ok"] = False
        result["errors"].append(f"Strict schema requires columns [{ID_COL},{TARGET_COL}]")
    if ID_COL not in sub.columns or TARGET_COL not in sub.columns:
        result["ok"] = False
        result["errors"].append(f"Expected columns {ID_COL},{TARGET_COL}; got {list(sub.columns)}")
    if compare_template and sample_path.exists():
        sample = pd.read_csv(sample_path)
        if len(sub) != len(sample):
            result["ok"] = False
            result["errors"].append(f"Row count {len(sub)} != sample {len(sample)}")
        if list(sub.columns) != list(sample.columns) and set(sub.columns) >= {ID_COL, TARGET_COL}:
            # column order may differ; still note
            result["column_order_note"] = f"sub={list(sub.columns)} sample={list(sample.columns)}"
        sample_ids = set(sample[ID_COL].astype(str)) if ID_COL in sample.columns else set(sample.iloc[:, 0].astype(str))
        sub_ids = set(sub[ID_COL].astype(str))
        if sample_ids != sub_ids:
            result["ok"] = False
            result["errors"].append(
                f"ID set mismatch: missing={len(sample_ids - sub_ids)} extra={len(sub_ids - sample_ids)}"
            )
    empty = sub[TARGET_COL].astype(str).str.strip().eq("").sum() if TARGET_COL in sub.columns else -1
    if empty:
        result["ok"] = False
        result["errors"].append(f"{empty} empty Target values")
    result["n_empty_targets"] = int(empty)
    if ID_COL in sub.columns:
        if sub[ID_COL].astype(str).duplicated().any():
            result["ok"] = False
            result["errors"].append("Duplicate IDs")
    if strict and TARGET_COL in sub.columns:
        bad = sub[TARGET_COL].map(lambda x: normalize_text(x) in {"", ".", "nan", "null", "none"})
        if bool(bad.any()):
            result["ok"] = False
            result["errors"].append(f"{int(bad.sum())} placeholder/empty Target values")
    return result


def phase2_expected_ids() -> list[str]:
    """Return the current expanded Phase-2 ID order from local audio + legacy set."""
    old = pd.read_csv(PHASE2_SAMPLE_SUBMISSION_CSV)[ID_COL].astype(str).tolist()
    old_set = set(old)
    if not PHASE2_NEW_AUDIO_DIR.exists():
        return old
    new = sorted(
        p.stem for p in PHASE2_NEW_AUDIO_DIR.glob("*.wav") if p.stem not in old_set
    )
    return old + new


def check_phase2_submission(submission_path: Path, *, strict: bool = True) -> dict:
    """Strictly validate the expanded Phase-2 upload contract.

    The checked-in Zindi template predates the 892-row extension, so this
    validator derives the expected union from the released legacy IDs and the
    local audio IDs instead of silently validating an obsolete 1,500-row file.
    """
    sub = pd.read_csv(submission_path)
    expected = phase2_expected_ids()
    result = check_submission(
        submission_path,
        sample_path=None,
        strict=strict,
        compare_template=False,
    )
    result["expected_rows"] = len(expected)
    result["phase"] = "phase2-expanded"
    if len(sub) != len(expected):
        result["ok"] = False
        result["errors"].append(f"Row count {len(sub)} != current Phase-2 {len(expected)}")
    if ID_COL in sub.columns:
        got = sub[ID_COL].astype(str).tolist()
        if got != expected:
            result["ok"] = False
            result["errors"].append("ID order/set does not match current Phase-2 union")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build / check Zindi submission")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--sample", type=Path, default=None)
    p.add_argument("--check-only", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    args = parse_args(argv)
    if args.check_only:
        report = check_submission(args.predictions, args.sample)
        print(report)
        if not report["ok"]:
            sys.exit(1)
        return
    preds = pd.read_csv(args.predictions)
    build_submission(preds, sample_path=args.sample, out_path=args.out)


if __name__ == "__main__":
    main()
