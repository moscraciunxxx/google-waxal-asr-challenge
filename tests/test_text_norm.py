"""Text normalization unit tests (shipped function)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text_norm import normalize_text, tokenize_words


def test_unicode_and_spaces():
    assert normalize_text("  café\t\n") == "café"
    assert tokenize_words("One  TWO") == ["one", "two"]


def test_punctuation_stripped():
    assert normalize_text("hello, world!") == "hello world"


def test_missing_value_sentinels_are_not_predictions():
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
    assert normalize_text("NaN") == ""
    assert normalize_text(" null ") == ""
