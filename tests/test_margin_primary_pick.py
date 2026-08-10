"""Unit tests for margin-primary multi-hyp language pick (no model I/O)."""

from __future__ import annotations


def pick(scored, cands, thr):
    scored = sorted(scored, key=lambda x: x[2], reverse=True)
    best_L, best_h, best_c = scored[0]
    second_c = scored[1][2] if len(scored) > 1 else -1e9
    margin = best_c - second_c
    if thr is None or margin >= thr:
        return best_L, best_h, margin, "maxconf" if thr is None else "margin_ok"
    for L, h, c in scored:
        if L == cands[0]:
            return L, h, margin, "primary_fb"
    return best_L, best_h, margin, "primary_missing"


def test_maxconf_when_margin_large():
    cands = ["ach", "lug", "sog"]
    scored = [("ach", "a", -0.05), ("lug", "l", -0.02), ("sog", "s", -0.08)]
    L, h, m, note = pick(scored, cands, thr=0.01)
    assert L == "lug" and h == "l" and note == "margin_ok"
    assert m == pytest_approx(0.03)


def pytest_approx(x, rel=1e-6):
    class A:
        def __eq__(self, other):
            return abs(other - x) <= rel * max(1.0, abs(x))

    return A()


def test_primary_fallback_when_margin_small():
    cands = ["ach", "lug", "sog"]
    scored = [("ach", "a", -0.030), ("lug", "l", -0.025), ("sog", "s", -0.08)]
    L, h, m, note = pick(scored, cands, thr=0.01)
    assert L == "ach" and h == "a" and note == "primary_fb"
    assert abs(m - 0.005) < 1e-9


def test_pure_multihyp_ignores_primary():
    cands = ["ach", "lug", "sog"]
    scored = [("ach", "a", -0.030), ("lug", "l", -0.025), ("sog", "s", -0.08)]
    L, h, m, note = pick(scored, cands, thr=None)
    assert L == "lug" and note == "maxconf"


def test_luo_and_lug_cand_sets():
    # policy mirror of build_phase2_margin_selective.cand_set
    def cand_set(lid_lang: str):
        if lid_lang == "luo":
            return ["ach", "lug", "sog"]
        if lid_lang == "lug":
            return ["lug", "nyn", "sog"]
        return [lid_lang] if lid_lang else ["lug"]

    assert cand_set("luo")[0] == "ach"
    assert cand_set("lug")[0] == "lug"
