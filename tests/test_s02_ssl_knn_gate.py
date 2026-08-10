"""Unit tests for pure S02 SSL kNN gate — real shipped path."""

from __future__ import annotations

import numpy as np

from src.s02_ssl_knn_gate import (
    S02Thresholds,
    calibrate_threshold,
    eval_binary_scores,
    knn_margin_scores,
    l2_normalize,
    margin_score,
    mean_prototype,
    s02_accept,
)


def test_margin_and_accept():
    luo = l2_normalize(np.array([1.0, 0.0, 0.0]))
    ach = l2_normalize(np.array([0.0, 1.0, 0.0]))
    x_luo = l2_normalize(np.array([0.9, 0.1, 0.0]))
    x_ach = l2_normalize(np.array([0.1, 0.9, 0.0]))
    s_l = margin_score(x_luo, luo, ach)
    s_a = margin_score(x_ach, luo, ach)
    assert s_l > s_a
    thr = S02Thresholds(min_score=0.0)
    assert s02_accept(s_l, thr)[0]
    # high thr rejects both
    thr_hi = S02Thresholds(min_score=10.0)
    assert not s02_accept(s_l, thr_hi)[0]


def test_calibrate_separable_fpr_at_most_5pct():
    # separable: luo scores high, ach low
    scores_luo = [1.0, 1.2, 0.9, 1.1, 0.95, 1.05, 1.15, 0.85, 1.0, 1.3]
    scores_ach = [-1.0, -0.8, -1.2, -0.5, -0.9, -1.1, -0.7, -0.6, -1.0, -0.85]
    thr, stats = calibrate_threshold(scores_luo, scores_ach, max_fpr=0.05)
    assert stats["fpr"] <= 0.05 + 1e-9
    assert stats["n_pos"] == 10.0 and stats["n_neg"] == 10.0
    assert stats["tpr"] >= 0.8
    assert "calibrated" in stats.get("note", "")
    # thr should accept most luo, reject ach
    st = eval_binary_scores(scores_luo + scores_ach, [True] * 10 + [False] * 10, thr)
    assert st["fpr"] <= 0.05 + 1e-9


def test_calibrate_inseparable_documented():
    # identical scores both classes → no thr with TPR>0 and FPR≤5% unless TPR=0
    same = [0.0] * 20
    thr, stats = calibrate_threshold(same, same, max_fpr=0.05)
    # either zero TPR with FPR ok, or fail_closed
    assert stats["fpr"] <= 0.05 + 1e-9
    # accepting mid score should not create high FPR under thr
    st = eval_binary_scores(same + same, [True] * 20 + [False] * 20, thr)
    assert st["fpr"] <= 0.05 + 1e-9


def test_mean_prototype_and_knn_scores():
    a = np.array([1.0, 0.0])
    b = np.array([0.9, 0.1])
    p = mean_prototype([a, b])
    assert abs(np.linalg.norm(p) - 1.0) < 1e-6
    X = np.stack([a, np.array([0.0, 1.0])])
    luo_p = l2_normalize(np.array([1.0, 0.0]))
    ach_p = l2_normalize(np.array([0.0, 1.0]))
    sc = knn_margin_scores(X, luo_p, ach_p)
    assert sc[0] > sc[1]
