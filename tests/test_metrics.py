"""Unit tests for WER/CER/score — exercises shipped src.metrics (no reimplementation)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import compute_cer, compute_wer, score_by_language, score_pairs, weighted_score
from src.text_norm import normalize_text


def test_normalize_lowercase_and_punct():
    assert normalize_text("  Hello, WORLD!  ") == "hello world"
    assert normalize_text(None) == ""


def test_perfect_match_zero_error():
    refs = ["ndaku oyo ezali", "mhoro shamwari"]
    hyps = ["ndaku oyo ezali", "mhoro shamwari"]
    m = score_pairs(refs, hyps)
    assert m["wer"] == 0.0
    assert m["cer"] == 0.0
    assert m["score"] == 0.0


def test_weighted_score_formula():
    assert weighted_score(0.2, 0.4) == pytest.approx(0.3)
    assert weighted_score(0.0, 0.0) == 0.0


def test_substitution_increases_wer():
    refs = ["one two three"]
    hyps = ["one two four"]
    wer = compute_wer(refs, hyps)
    assert wer > 0.0
    assert wer <= 1.0


def test_score_by_language():
    refs = ["a b", "c d", "e f"]
    hyps = ["a b", "c x", "e f"]
    langs = ["lin", "sna", "lin"]
    m = score_by_language(refs, hyps, langs)
    assert "overall" in m and "lin" in m and "sna" in m
    assert m["lin"]["wer"] == 0.0
    assert m["sna"]["wer"] > 0.0


def test_metric_smoke_artifact(tmp_path, monkeypatch):
    """Write metric smoke JSON like the verification plan expects."""
    refs = ["hello world", "foo bar baz"]
    hyps = ["hello word", "foo bar baz"]
    m = score_pairs(refs, hyps)
    out = tmp_path / "metric_smoke.json"
    out.write_text(json.dumps(m, indent=2))
    loaded = json.loads(out.read_text())
    assert "wer" in loaded and "cer" in loaded and "score" in loaded
    assert loaded["score"] == pytest.approx(0.5 * loaded["wer"] + 0.5 * loaded["cer"])
