"""Rules: test gold must never enter training frames."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORBIDDEN_TRAIN_SPLITS, ID_COL, TARGET_COL
from src.data_index import assert_no_test_gold_in_training, load_training_frame
from src.dataset import load_hf_asr_split, load_labeled_splits


def test_forbidden_splits_contains_test():
    assert "test" in FORBIDDEN_TRAIN_SPLITS


def test_load_labeled_splits_rejects_test():
    with pytest.raises(ValueError, match="forbidden split"):
        load_labeled_splits(languages=("lin",), splits=("test",), max_per_lang_split=1)


def test_raw_test_split_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="allow_test=True"):
        load_hf_asr_split("lin", "test", max_samples=1)


def test_assert_no_test_gold(tmp_path):
    train = pd.DataFrame({ID_COL: ["lin_1", "sna_2"], TARGET_COL: ["a", "b"]})
    index = pd.DataFrame(
        {
            ID_COL: ["lin_1", "sna_2", "lin_99"],
            "split": ["train", "train", "test"],
            TARGET_COL: ["a", "b", "SECRET"],
        }
    )
    tpath = tmp_path / "Train.csv"
    ipath = tmp_path / "index.csv"
    train.to_csv(tpath, index=False)
    index.to_csv(ipath, index=False)
    assert_no_test_gold_in_training(tpath, ipath)

    # Overlap must raise
    train2 = pd.DataFrame({ID_COL: ["lin_1", "lin_99"], TARGET_COL: ["a", "leak"]})
    train2.to_csv(tpath, index=False)
    with pytest.raises(RuntimeError, match="overlap"):
        assert_no_test_gold_in_training(tpath, ipath)


def test_load_training_frame_excludes_test(tmp_path):
    index = pd.DataFrame(
        {
            ID_COL: ["a", "b", "c"],
            "split": ["train", "validation", "test"],
            TARGET_COL: ["t1", "t2", "SECRET"],
        }
    )
    ipath = tmp_path / "index.csv"
    index.to_csv(ipath, index=False)
    frame = load_training_frame(ipath)
    assert set(frame["split"]) == {"train", "validation"}
    assert "SECRET" not in frame[TARGET_COL].values
