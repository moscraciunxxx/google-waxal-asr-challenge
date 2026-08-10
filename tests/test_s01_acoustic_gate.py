"""Unit tests for S01 pure thr/feature helpers — real shipped path."""

from __future__ import annotations

from src.s01_acoustic_gate import (
    AcousticMass,
    S01Thresholds,
    calibrate_thresholds,
    eval_binary,
    mass_from_mapping,
    s01_accept,
)


def test_s01_accept_and_fail_closed():
    thr = S01Thresholds(min_mass_luo=0.55, max_mass_ach=0.25, min_margin_luo_ach=0.20, max_mass_bantu=0.45)
    ok, reason = s01_accept(AcousticMass(0.70, 0.10, 0.15), thr)
    assert ok and reason == "s01_acoustic_accept"
    bad, r2 = s01_accept(AcousticMass(0.40, 0.10, 0.10), thr)
    assert not bad and r2 == "mass_luo_low"
    bad2, r3 = s01_accept(AcousticMass(0.70, 0.40, 0.10), thr)
    assert not bad2 and r3 == "mass_ach_high"
    bad3, r4 = s01_accept(AcousticMass(0.55, 0.40, 0.05), thr)
    assert not bad3  # margin 0.15 < 0.20


def test_calibrate_prefers_low_fpr():
    # clear separation: luo high mass_luo, ach high mass_ach
    luo = [AcousticMass(0.8, 0.05, 0.1) for _ in range(20)]
    ach = [AcousticMass(0.2, 0.6, 0.1) for _ in range(20)]
    thr, stats = calibrate_thresholds(luo, ach, max_fpr=0.05)
    assert stats["fpr"] <= 0.05 + 1e-9
    assert stats["tpr"] >= 0.9
    # apply thr: luo accepts, ach rejects
    assert s01_accept(luo[0], thr)[0]
    assert not s01_accept(ach[0], thr)[0]


def test_calibrate_fail_closed_when_inseparable():
    # mid-mass identical → any FPR-safe thr has TPR=0 (both classes rejected)
    same = [AcousticMass(0.5, 0.5, 0.0) for _ in range(10)]
    thr, stats = calibrate_thresholds(same, same, max_fpr=0.05)
    assert stats["fpr"] <= 0.05 + 1e-9
    assert stats["tpr"] == 0.0  # inseparable → no useful accepts
    assert not s01_accept(AcousticMass(0.5, 0.5, 0.0), thr)[0]


def test_calibrate_fail_closed_high_mass_inseparable_residual_regime():
    """Real open-val / residual regime: mass_luo≈0.99 both classes (LID confuses Ach).

    Fail-closed must accept NOTHING — including mass_luo=1.0 mass_ach=0.
    """
    luo = [AcousticMass(0.999, 0.0, 0.001) for _ in range(20)]
    ach = [AcousticMass(0.999, 0.0, 0.001) for _ in range(20)]  # same LID mass as residual Ach
    thr, stats = calibrate_thresholds(luo, ach, max_fpr=0.05)
    assert stats.get("zero_accept") is True
    assert stats["candidates_ok"] == 0
    assert "fail_closed" in stats.get("note", "")
    assert stats["fpr"] == 0.0 and stats["tpr"] == 0.0
    # residual operating point must not accept
    ok, reason = s01_accept(AcousticMass(1.0, 0.0, 0.0), thr)
    assert not ok, reason
    ok2, _ = s01_accept(AcousticMass(0.999, 0.0, 0.0), thr)
    assert not ok2


def test_mass_from_mapping_and_eval_binary():
    m = mass_from_mapping({"mass_luo": 0.9, "mass_ach": 0.05, "mass_bantu": 0.05})
    assert m.mass_luo == 0.9
    thr = S01Thresholds(0.5, 0.3, 0.2, 0.5)
    st = eval_binary([m, AcousticMass(0.1, 0.8, 0.1)], [True, False], thr)
    assert st["tp"] == 1.0 and st["tn"] == 1.0
