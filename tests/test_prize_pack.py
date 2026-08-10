"""Unit tests for floor-first prize packing helpers."""

from __future__ import annotations

import pytest

from src.prize_pack import (
    align_ids,
    apply_replace_set,
    char_sim,
    char_sim_ok,
    diff_stats,
    is_banned_mass_rewrite,
    length_guard_ok,
    word_level_merge,
)


def test_apply_replace_set_defaults_to_floor():
    floor = {"A": "hello world", "B": "foo bar", "C": "keep me"}
    replaces = [{"ID": "A", "own_hyp": "hello earth"}]
    out = apply_replace_set(floor, replaces)
    assert out["A"] == "hello earth"
    assert out["B"] == "foo bar"
    assert out["C"] == "keep me"


def test_align_ids_order_and_missing():
    ids = ["B", "A"]
    rows = align_ids(ids, {"A": "a", "B": "b"})
    assert [r["ID"] for r in rows] == ["B", "A"]
    assert rows[0]["Target"] == "b"
    with pytest.raises(KeyError):
        align_ids(["Z"], {"A": "a"})


def test_length_guard():
    assert length_guard_ok("one two three", "one two four")
    assert not length_guard_ok("one two three four five", "x")
    assert not length_guard_ok("hello", "")


def test_diff_stats_and_mass_rewrite():
    a = {f"I{i}": "same" for i in range(100)}
    b = dict(a)
    b["I0"] = "diff"
    st = diff_stats(a, b)
    assert st["n_diff"] == 1 and st["n_same"] == 99
    assert not is_banned_mass_rewrite(10, 1500)
    assert is_banned_mass_rewrite(1479, 1500)  # multifamily-class mass rewrite


def test_char_sim_gate_rejects_total_rewrite():
    close = "omuntu orikwiragura ajwaire esaati"
    close2 = "omuntu olukwiragula ajuyire essaati"
    far = "completely different sentence with no overlap at all xyz"
    assert char_sim(close, close2) > 0.6
    assert char_sim_ok(close, close2, min_sim=0.6)
    assert char_sim(close, far) < 0.3
    assert not char_sim_ok(close, far, min_sim=0.6)
    assert not char_sim_ok(close, "", min_sim=0.6)


def test_word_level_merge_keeps_floor_on_low_sim_and_swaps_variants():
    floor = "omuntu orikwiragura ajwaire esaati ya mutare"
    # high-sim orthography variants on two tokens
    specialist = "omuntu olukwiragula ajuyire essaati ya mutale"
    merged = word_level_merge(floor, specialist, token_min_sim=0.72)
    # should change something (variant swaps) but stay close to floor length
    assert len(merged.split()) == len(floor.split())
    assert char_sim(floor, merged) >= 0.75
    # total rewrite → keep floor tokens (no insert of unrelated mass)
    far = "completely different words with zero orthography overlap here"
    assert word_level_merge(floor, far, token_min_sim=0.72) == floor
