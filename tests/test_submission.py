"""Submission schema tests against shipped build_submission / check_submission."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ID_COL, TARGET_COL
from src.submission import build_submission, check_submission


def test_build_submission_aligns_and_nonempty(tmp_path):
    sample = pd.DataFrame({ID_COL: ["lin_1", "sna_2", "lug_3"], TARGET_COL: ["", "", ""]})
    preds = pd.DataFrame(
        {
            ID_COL: ["sna_2", "lin_1", "lug_3"],
            "prediction": ["hello", "world", "test"],
        }
    )
    sp = tmp_path / "SampleSubmission.csv"
    sample.to_csv(sp, index=False)
    out = tmp_path / "submission.csv"
    sub = build_submission(preds, sample_path=sp, out_path=out)
    assert list(sub[ID_COL]) == ["lin_1", "sna_2", "lug_3"]
    assert list(sub[TARGET_COL]) == ["world", "hello", "test"]
    report = check_submission(out, sp)
    assert report["ok"], report["errors"]


def test_empty_prediction_filled(tmp_path):
    sample = pd.DataFrame({ID_COL: ["a"], TARGET_COL: [""]})
    preds = pd.DataFrame({ID_COL: ["a"], "prediction": [""]})
    sp = tmp_path / "s.csv"
    sample.to_csv(sp, index=False)
    out = tmp_path / "sub.csv"
    sub = build_submission(preds, sample_path=sp, out_path=out)
    assert str(sub.iloc[0][TARGET_COL]).strip() != "" or sub.iloc[0][TARGET_COL] == " "


def test_strict_submission_rejects_missing_ids_and_placeholders(tmp_path):
    sample = pd.DataFrame({ID_COL: ["a", "b"], TARGET_COL: ["", ""]})
    preds = pd.DataFrame({ID_COL: ["a"], "prediction": ["."]})
    sp = tmp_path / "s.csv"
    out = tmp_path / "sub.csv"
    sample.to_csv(sp, index=False)
    with pytest.raises(ValueError, match="Missing predictions"):
        build_submission(preds, sample_path=sp, out_path=out, strict=True)

    full = pd.DataFrame({ID_COL: ["a"], TARGET_COL: ["."]})
    full.to_csv(out, index=False)
    report = check_submission(out, sample_path=sp, strict=True)
    assert report["ok"] is False
    assert any("placeholder" in e for e in report["errors"])
