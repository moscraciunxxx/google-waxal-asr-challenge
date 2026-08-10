"""Prove zindi_est scoring path used by proxy A/B is real (not hardcoded)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import score_pairs


def test_score_pairs_perfect_match_is_one_minus_zero():
    refs = ["omwana alina emmere", "abantu babiri"]
    hyps = ["omwana alina emmere", "abantu babiri"]
    s = score_pairs(refs, hyps)
    assert s["n"] == 2
    assert s["wer"] == 0.0
    assert s["cer"] == 0.0
    assert s["score"] == 0.0
    zindi = 1.0 - s["score"]
    assert zindi == 1.0


def test_score_pairs_total_mismatch_has_positive_error():
    refs = ["aaaa bbbb"]
    hyps = ["xxxx yyyy"]
    s = score_pairs(refs, hyps)
    assert s["score"] > 0.5
    zindi = 1.0 - s["score"]
    assert zindi < 0.5


def test_proxy_ab_gate_json_schema_if_present():
    p = ROOT / "outputs" / "proxy_ab_gate.json"
    if not p.exists():
        return
    import json

    d = json.loads(p.read_text())
    assert "results" in d
    assert "openset_multihyp_conf" in d["results"]
    base = d["results"]["openset_multihyp_conf"]["zindi_est"]
    assert 0.0 <= base <= 1.0
    # winners must beat baseline by gate_delta
    for w in d.get("winners", []):
        assert w["delta"] >= d["gate_delta"] - 1e-9
        assert abs(w["zindi_est"] - (base + w["delta"])) < 1e-6
