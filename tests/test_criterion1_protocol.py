"""Unit tests for criterion-1 honest select/report protocol (no report leak)."""

from __future__ import annotations

import pytest

from src.criterion1_protocol import (
    REPORT_SLICE_LABEL,
    SELECT_SLICE_LABEL,
    CheckpointCandidate,
    ValProtocol,
    all_languages_beat,
    assert_honest_sna_meta,
    finalize_report,
    select_checkpoint_by_slice,
    split_val_protocol,
)


def test_slices_disjoint_nonempty():
    proto = split_val_protocol()
    assert len(proto.report_indices) == 50
    assert len(proto.select_indices) == 40
    assert set(proto.report_indices).isdisjoint(proto.select_indices)
    assert proto.report_slice != proto.select_slice
    assert proto.report_slice == REPORT_SLICE_LABEL
    assert proto.select_slice == SELECT_SLICE_LABEL


def test_overlap_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        ValProtocol(
            report_indices=tuple(range(0, 50)),
            select_indices=tuple(range(40, 90)),  # overlaps 40-49
        )


def test_select_ignores_report_fields():
    """Best by select ME; fake better report metrics must not change pick."""
    c_good_select = CheckpointCandidate(
        step=10, select_mean_error=0.18, select_beats_baseline=True, state_key="a"
    )
    c_bad_select = CheckpointCandidate(
        step=20, select_mean_error=0.25, select_beats_baseline=True, state_key="b"
    )
    # Even if we claim report is better for b, selection must pick a
    pick = select_checkpoint_by_slice(
        [c_bad_select, c_good_select],
        report_mean_error=0.01,  # would favor b if leaked
        report_beats=True,
        report_metrics={"wer": 0.0, "cer": 0.0},
    )
    assert pick is not None
    assert pick.state_key == "a"
    assert pick.step == 10


def test_select_prefers_select_beaters():
    beater = CheckpointCandidate(step=5, select_mean_error=0.20, select_beats_baseline=True)
    non = CheckpointCandidate(step=1, select_mean_error=0.15, select_beats_baseline=False)
    pick = select_checkpoint_by_slice([non, beater])
    assert pick is beater


def test_select_empty():
    assert select_checkpoint_by_slice([]) is None


def test_finalize_report_uses_report_only():
    proto = split_val_protocol()
    entry = finalize_report(
        {"wer": 0.30, "cer": 0.08, "score": 0.19, "n": 50},
        {"wer": 0.31, "cer": 0.08, "score": 0.195, "n": 50},
        beats=True,
        own_model="checkpoints/x",
        baseline_model="waxal-benchmarking/mms-300m-waxal-sna",
        lang="sna",
        n=50,
        protocol=proto,
        own_kind="pure_ft",
        train_meta={"early_stop_slice": SELECT_SLICE_LABEL, "report_slice": REPORT_SLICE_LABEL},
    )
    assert entry["beats"] is True
    assert entry["report_slice"] == REPORT_SLICE_LABEL
    assert entry["selection_slice"] == SELECT_SLICE_LABEL
    assert entry["mean_error_own"] == pytest.approx(0.19)


def test_finalize_report_rejects_inconsistent_beats_claim():
    proto = split_val_protocol()
    with pytest.raises(ValueError, match="beats disagrees"):
        finalize_report(
            {"wer": 0.40, "cer": 0.20},
            {"wer": 0.30, "cer": 0.10},
            beats=True,
            own_model="checkpoints/x",
            baseline_model="waxal-benchmarking/mms-300m-waxal-sna",
            lang="sna",
            n=50,
            protocol=proto,
            own_kind="pure_ft",
        )


def test_assert_honest_sna_meta_ok():
    assert_honest_sna_meta(
        {
            "early_stop_slice": SELECT_SLICE_LABEL,
            "report_slice": REPORT_SLICE_LABEL,
            "pure_own_checkpoint": True,
            "no_baseline_blend": True,
            "method": "pure_micro_ft_last1_lm_select_only",
        }
    )


def test_assert_honest_rejects_same_slice():
    with pytest.raises(AssertionError, match="differ"):
        assert_honest_sna_meta(
            {
                "early_stop_slice": REPORT_SLICE_LABEL,
                "report_slice": REPORT_SLICE_LABEL,
                "pure_own_checkpoint": True,
                "no_baseline_blend": True,
            }
        )


def test_assert_honest_rejects_baseline_soup_method():
    with pytest.raises(AssertionError, match="baseline"):
        assert_honest_sna_meta(
            {
                "early_stop_slice": SELECT_SLICE_LABEL,
                "report_slice": REPORT_SLICE_LABEL,
                "pure_own_checkpoint": True,
                "no_baseline_blend": True,
                "method": "baseline_weight_soup",
            }
        )


def test_assert_honest_rejects_baseline_alpha_field():
    with pytest.raises(AssertionError):
        assert_honest_sna_meta(
            {
                "early_stop_slice": SELECT_SLICE_LABEL,
                "report_slice": REPORT_SLICE_LABEL,
                "pure_own_checkpoint": True,
                "no_baseline_blend": True,
                "method": "pure_ft",
                "alpha_base": 0.99,
                "own_ft_component": "checkpoints/mms-sna-ft-v2",
            }
        )


def test_assert_honest_allows_pure_own_ensemble():
    assert_honest_sna_meta(
        {
            "early_stop_slice": SELECT_SLICE_LABEL,
            "report_slice": REPORT_SLICE_LABEL,
            "pure_own_checkpoint": True,
            "no_baseline_blend": True,
            "method": "pure_ft_ensemble_avg_select_weight",
            "ensemble_weight": 0.5,
            "components": ["track_a/best", "track_b/best"],
        }
    )


def test_all_languages_beat():
    assert all_languages_beat(
        [{"beats": True}, {"beats": True}, {"beats": True}]
    )
    assert not all_languages_beat(
        [{"beats": True}, {"beats": False}, {"beats": True}]
    )
    assert not all_languages_beat([{"beats": True}, {"beats": True}])
