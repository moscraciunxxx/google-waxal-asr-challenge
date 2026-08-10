"""S02 SSL kNN / margin gate — pure score→accept and open-val FPR helpers.

Decision features are **SSL embedding distances** (not MMS-LID mass/p1).
Score = d(x, ach_proto) − d(x, luo_proto)  (larger ⇒ more Luo-like).
Accept if score ≥ thr. Calibrate thr to max TPR s.t. FPR ≤ max_fpr on open val.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class S02Thresholds:
    """Accept when margin score ≥ min_score."""

    min_score: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"min_score": float(self.min_score)}


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalize 1d or 2d array."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / max(n, eps)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b))


def margin_score(x: np.ndarray, luo_proto: np.ndarray, ach_proto: np.ndarray) -> float:
    """Larger score ⇒ closer to Luo than Ach (d_ach − d_luo)."""
    return euclidean(x, ach_proto) - euclidean(x, luo_proto)


def mean_prototype(embeddings: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Mean prototype over a list/stack of embeddings (L2-normalized after mean)."""
    if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
        mat = embeddings
    else:
        mat = np.stack([np.asarray(e, dtype=np.float64).ravel() for e in embeddings], axis=0)
    if mat.size == 0:
        raise ValueError("empty embeddings for prototype")
    proto = mat.mean(axis=0)
    return l2_normalize(proto)


def s02_accept(score: float, thr: S02Thresholds) -> tuple[bool, str]:
    """Return (accept, reason). Fail closed if score < thr."""
    if float(score) >= float(thr.min_score):
        return True, "s02_ssl_knn_accept"
    return False, "score_below_thr"


def eval_binary_scores(
    scores: Sequence[float],
    labels_is_luo: Sequence[bool],
    thr: S02Thresholds,
) -> dict[str, float]:
    """TPR/FPR: labels_is_luo True = true Luo positive."""
    if len(scores) != len(labels_is_luo) or not scores:
        return {"n": 0.0, "tpr": 0.0, "fpr": 0.0, "tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels_is_luo):
        pred, _ = s02_accept(float(s), thr)
        if y and pred:
            tp += 1
        elif y and not pred:
            fn += 1
        elif (not y) and pred:
            fp += 1
        else:
            tn += 1
    n_pos = tp + fn
    n_neg = tn + fp
    return {
        "n": float(len(scores)),
        "tpr": (tp / n_pos) if n_pos else 0.0,
        "fpr": (fp / n_neg) if n_neg else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
    }


def calibrate_threshold(
    scores_luo: Sequence[float],
    scores_ach: Sequence[float],
    *,
    max_fpr: float = 0.05,
) -> tuple[S02Thresholds, dict[str, Any]]:
    """Maximize TPR s.t. FPR ≤ max_fpr by scanning thr over unique scores + midpoints.

    If no thr meets max_fpr with any TPR, return thr = +inf (zero accepts) with note.
    """
    scores = list(scores_luo) + list(scores_ach)
    labels = [True] * len(scores_luo) + [False] * len(scores_ach)
    if not scores_luo or not scores_ach:
        thr = S02Thresholds(min_score=float("inf"))
        return thr, {
            "note": "empty_class_fail_closed",
            "fpr": 0.0,
            "tpr": 0.0,
            "n_pos": float(len(scores_luo)),
            "n_neg": float(len(scores_ach)),
            "thr": thr.as_dict(),
            "candidates_ok": 0,
        }

    # Candidate thresholds: just above each ach score (so that ach is rejected)
    # and midpoints of sorted unique scores
    uniq = sorted(set(float(s) for s in scores))
    cands: list[float] = []
    for i, u in enumerate(uniq):
        cands.append(u)
        if i + 1 < len(uniq):
            cands.append(0.5 * (u + uniq[i + 1]))
    # also slightly above max ach score
    max_ach = max(float(s) for s in scores_ach)
    cands.append(max_ach + 1e-9)
    cands.append(max_ach + 1e-3)
    cands = sorted(set(cands))

    best_thr: S02Thresholds | None = None
    best_tpr = -1.0
    best_stats: dict[str, Any] = {}
    ok_count = 0

    for t in cands:
        thr = S02Thresholds(min_score=t)
        st = eval_binary_scores(scores, labels, thr)
        if st["fpr"] <= max_fpr + 1e-12:
            ok_count += 1
            if st["tpr"] > best_tpr + 1e-15 or (
                abs(st["tpr"] - best_tpr) < 1e-15
                and st["fpr"] < best_stats.get("fpr", 1.0)
            ):
                best_tpr = st["tpr"]
                best_thr = thr
                best_stats = {**st, "thr": thr.as_dict()}

    if best_thr is None or best_tpr < 0:
        thr = S02Thresholds(min_score=float("inf"))
        st = eval_binary_scores(scores, labels, thr)
        return thr, {
            **st,
            "thr": thr.as_dict(),
            "note": "no_thr_met_max_fpr_fail_closed",
            "candidates_ok": 0,
            "max_fpr_target": max_fpr,
        }

    best_stats["note"] = "calibrated_open_val"
    best_stats["candidates_ok"] = ok_count
    best_stats["max_fpr_target"] = max_fpr
    return best_thr, best_stats


def knn_margin_scores(
    X: np.ndarray,
    luo_proto: np.ndarray,
    ach_proto: np.ndarray,
) -> np.ndarray:
    """Vector of margin scores for rows of X."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    out = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        out[i] = margin_score(X[i], luo_proto, ach_proto)
    return out
