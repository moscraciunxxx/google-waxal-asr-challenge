"""Unit tests for multi-family fusion + beat baseline helpers (shipped code)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.legit_fusion import beats_baseline, fuse_row, mean_error, pack_fusion_submission_row


def test_mean_error_and_beats():
    assert abs(mean_error(0.2, 0.4) - 0.3) < 1e-9
    own = {"wer": 0.10, "cer": 0.05}
    base = {"wer": 0.20, "cer": 0.10}
    assert beats_baseline(own, base)
    assert not beats_baseline(base, own)


def test_fuse_prefers_mms_strong_lid():
    r = fuse_row(
        "mms text here",
        "whisper text",
        mms_score=-0.05,
        decode_lang="lug",
        lid_lang="lug",
        lid_p1=0.99,
    )
    assert r["fusion_source"] == "mms"
    assert r["fused_hyp"] == "mms text here"


def test_fuse_whisper_when_mms_conf_weak():
    r = fuse_row(
        "mms weak",
        "whisper better path",
        mms_score=-0.9,
        decode_lang="lug",
        lid_lang="lug",
        lid_p1=0.99,
    )
    assert r["fusion_source"] == "whisper"
    assert "whisper" in r["fused_hyp"]


def test_fuse_whisper_on_open_set_luo():
    r = fuse_row(
        "ach route mms",
        "whisper luo hyp",
        mms_score=-0.05,
        decode_lang="ach",
        lid_lang="luo",
        lid_p1=0.99,
    )
    assert r["fusion_source"] == "whisper"
    assert r["fusion_reason"] == "open_set_luo_router_whisper"


def test_pack_submission():
    row = pack_fusion_submission_row("ID_X", " Hello ")
    assert row["ID"] == "ID_X"
    assert row["Target"] == "hello"
