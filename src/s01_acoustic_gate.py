"""S01 acoustic accept gate — pure thr/feature logic (no model I/O).

Acoustic features (mass_luo / mass_ach / mass_bantu) come from MMS-LID-126
waveform softmax bucketed by language family (see scripts/luo_acoustic_router.py).
Calibration chooses thresholds on **open** labeled sets only (FLEURS luo, WAXAL ach).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AcousticMass:
    """Family mass features from one waveform LID forward."""

    mass_luo: float
    mass_ach: float
    mass_bantu: float = 0.0
    mass_other: float = 0.0
    top1_lang: str = ""
    top1_p: float = 0.0


@dataclass(frozen=True)
class S01Thresholds:
    """High-precision accept thresholds (fail closed when not met)."""

    min_mass_luo: float = 0.55
    max_mass_ach: float = 0.25
    min_margin_luo_ach: float = 0.20  # mass_luo - mass_ach
    max_mass_bantu: float = 0.45

    def as_dict(self) -> dict[str, float]:
        return {
            "min_mass_luo": self.min_mass_luo,
            "max_mass_ach": self.max_mass_ach,
            "min_margin_luo_ach": self.min_margin_luo_ach,
            "max_mass_bantu": self.max_mass_bantu,
        }


def s01_accept(mass: AcousticMass, thr: S01Thresholds) -> tuple[bool, str]:
    """Return (accept, reason). Fail closed."""
    ml = float(mass.mass_luo)
    ma = float(mass.mass_ach)
    mb = float(mass.mass_bantu)
    if ml < thr.min_mass_luo:
        return False, "mass_luo_low"
    if ma > thr.max_mass_ach:
        return False, "mass_ach_high"
    if (ml - ma) < thr.min_margin_luo_ach:
        return False, "margin_luo_ach_low"
    if mb > thr.max_mass_bantu:
        return False, "mass_bantu_high"
    return True, "s01_acoustic_accept"


def eval_binary(
    masses: Sequence[AcousticMass],
    labels_is_luo: Sequence[bool],
    thr: S01Thresholds,
) -> dict[str, float]:
    """TPR/FPR on open labeled set. labels_is_luo True = true Luo."""
    if len(masses) != len(labels_is_luo) or not masses:
        return {"n": 0.0, "tpr": 0.0, "fpr": 0.0, "tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}
    tp = fp = tn = fn = 0
    for m, y in zip(masses, labels_is_luo):
        pred, _ = s01_accept(m, thr)
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
        "n": float(len(masses)),
        "tpr": (tp / n_pos) if n_pos else 0.0,
        "fpr": (fp / n_neg) if n_neg else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def calibrate_thresholds(
    luo_masses: Sequence[AcousticMass],
    ach_masses: Sequence[AcousticMass],
    *,
    max_fpr: float = 0.05,
    grid_luo: Sequence[float] | None = None,
    grid_ach: Sequence[float] | None = None,
    grid_margin: Sequence[float] | None = None,
    grid_bantu: Sequence[float] | None = None,
) -> tuple[S01Thresholds, dict[str, Any]]:
    """Pick thr maximizing TPR on Luo subject to FPR on Ach ≤ max_fpr.

    Open val only. If no thr meets FPR, returns ultra-strict thr with note.
    """
    grid_luo = list(grid_luo or [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    grid_ach = list(grid_ach or [0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    grid_margin = list(grid_margin or [0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    grid_bantu = list(grid_bantu or [0.30, 0.40, 0.50, 0.60, 0.99])

    labels = [True] * len(luo_masses) + [False] * len(ach_masses)
    masses = list(luo_masses) + list(ach_masses)

    best: S01Thresholds | None = None
    best_tpr = -1.0
    best_stats: dict[str, Any] = {}
    candidates_ok = 0

    for ml in grid_luo:
        for ma in grid_ach:
            for mg in grid_margin:
                for mb in grid_bantu:
                    thr = S01Thresholds(
                        min_mass_luo=ml,
                        max_mass_ach=ma,
                        min_margin_luo_ach=mg,
                        max_mass_bantu=mb,
                    )
                    st = eval_binary(masses, labels, thr)
                    if st["fpr"] <= max_fpr + 1e-12:
                        candidates_ok += 1
                        # prefer higher TPR; tie-break lower FPR then higher margin
                        key = (st["tpr"], -st["fpr"], mg)
                        if st["tpr"] > best_tpr or (
                            abs(st["tpr"] - best_tpr) < 1e-12
                            and best is not None
                            and st["fpr"] < best_stats.get("fpr", 1.0)
                        ):
                            best_tpr = st["tpr"]
                            best = thr
                            best_stats = {**st, "thr": thr.as_dict()}

    if best is None:
        # Fail closed for real: thr that no finite mass in [0,1] can satisfy.
        # (min_mass_luo > 1.0 ⇒ s01_accept always False for valid AcousticMass.)
        best = S01Thresholds(
            min_mass_luo=1.01,
            max_mass_ach=-0.01,
            min_margin_luo_ach=1.01,
            max_mass_bantu=-0.01,
        )
        st = eval_binary(masses, labels, best)
        best_stats = {
            **st,
            "thr": best.as_dict(),
            "note": "no_thr_met_max_fpr_used_fail_closed_zero_accept",
            "candidates_ok": 0,
            "zero_accept": True,
        }
        # Honesty: inseparable open-val must not ship residual replaces
        assert st["tp"] == 0.0 and st["fp"] == 0.0, "fail_closed thr must reject all open-val masses"
    else:
        best_stats["note"] = "calibrated_open_val"
        best_stats["candidates_ok"] = candidates_ok
        best_stats["zero_accept"] = False

    best_stats["max_fpr_target"] = max_fpr
    best_stats["n_luo"] = len(luo_masses)
    best_stats["n_ach"] = len(ach_masses)
    return best, best_stats


def mass_from_mapping(row: Mapping[str, Any]) -> AcousticMass:
    """Build AcousticMass from a dict/CSV row."""
    return AcousticMass(
        mass_luo=float(row.get("mass_luo") or 0),
        mass_ach=float(row.get("mass_ach") or 0),
        mass_bantu=float(row.get("mass_bantu") or 0),
        mass_other=float(row.get("mass_other") or 0),
        top1_lang=str(row.get("top1_lang") or ""),
        top1_p=float(row.get("top1_p") or 0),
    )
