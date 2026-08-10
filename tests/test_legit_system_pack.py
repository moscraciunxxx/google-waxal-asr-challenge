"""Unit tests for legit_system pure packing helpers (no model I/O)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.legit_system_pack import (
    pack_hyp_row,
    resolve_decode_lang,
    rows_to_records,
    whisper_language_name,
)


def test_resolve_decode_lang_direct_and_fallback():
    assert resolve_decode_lang("lug") == "lug"
    assert resolve_decode_lang("lin") == "lin"
    assert resolve_decode_lang("luo") == "ach"  # first fallback with WAXAL specialist
    assert resolve_decode_lang("unknown_xyz") == "lug"


def test_whisper_language_name():
    assert whisper_language_name("lin") == "lingala"
    assert whisper_language_name("sna") == "shona"
    assert whisper_language_name("lug") is None
    assert whisper_language_name("zzz") is None


def test_pack_hyp_row_normalizes_and_schema():
    row = pack_hyp_row(
        "ID_TEST",
        lid_lang="LUG",
        lid_p1=0.99,
        decode_lang="lug",
        mms_hyp="  Hello, World!  ",
        mms_score=-0.5,
        whisper_hyp="Hello world",
        whisper_score=None,
        mms_model_id="waxal-benchmarking/mms-300m-waxal-lug",
        whisper_model_id="openai/whisper-small",
        seed=42,
    )
    assert row["ID"] == "ID_TEST"
    assert row["lid_lang"] == "lug"
    assert row["mms_hyp"] == "hello world"
    assert row["whisper_hyp"] == "hello world"
    assert row["omnilingual"] == "deferred"
    assert row["seed"] == 42


def test_rows_to_records_requires_keys():
    good = pack_hyp_row(
        "a",
        lid_lang="lin",
        lid_p1=0.5,
        decode_lang="lin",
        mms_hyp="x",
        mms_score=None,
        whisper_hyp="y",
        whisper_score=None,
    )
    recs = rows_to_records([good])
    assert len(recs) == 1
    assert recs[0]["mms_hyp"] == "x"
