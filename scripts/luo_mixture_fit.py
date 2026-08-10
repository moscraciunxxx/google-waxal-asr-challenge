#!/usr/bin/env python3
"""Estimate the true-Dholuo fraction pi among the 785 lid=luo phase-2 rows.

Why: the shipped hybrid accepts 35 rows at agreement thr 0.3, and UPLOAD_DECISION.md
concludes from that count that "the true-Dholuo mass is SMALL". That is circular — a
strict threshold accepts few rows by construction, whatever the underlying mixture.

Proper test: the agreement statistic cer_pm has a measured distribution under each
hypothesis (FLEURS Dholuo probe = true luo, WAXAL ach probe = true Acholi). The 785
test rows are a mixture of the two. Fit

    F_test(t) ~= pi * F_luo(t) + (1 - pi) * F_ach(t)

over a grid of thresholds t by least squares on the CDFs, with a bootstrap CI over
the (n=40, n=40) probe samples. pi is what decides whether the 31-row hybrid or a
much looser gate is right on the private 70%.

Caveat carried into the report: both probes are n=40, and neither is spontaneous
long-form Dholuo (the ANV request that would supply that is still pending). The fit
bounds pi under the domains we can actually measure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "next_iter" / "luo_mixture_fit.json"

# CDF evaluation grid: dense where the mass is, log-spaced into the tail
GRID = np.concatenate([np.arange(0.05, 1.0, 0.05), np.array([1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 60.0, 120.0])])


def cdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.array([(x <= t).mean() for t in grid])


def fit_pi(f_test: np.ndarray, f_luo: np.ndarray, f_ach: np.ndarray) -> tuple[float, float]:
    """Least-squares pi on the CDF grid, clipped to [0,1]. Returns (pi, rmse)."""
    d = f_luo - f_ach
    y = f_test - f_ach
    denom = float(d @ d)
    if denom <= 0:
        return float("nan"), float("nan")
    pi = float(np.clip((d @ y) / denom, 0.0, 1.0))
    rmse = float(np.sqrt(np.mean((f_test - (pi * f_luo + (1 - pi) * f_ach)) ** 2)))
    return pi, rmse


def main() -> None:
    luo = pd.read_csv("/tmp/probe_luo.csv")
    ach = pd.read_csv("/tmp/probe_ach.csv")
    test = pd.read_csv(ROOT / "outputs" / "next_iter" / "hybrid_agreement_785.csv")

    x_luo = luo.cer_pm.to_numpy(float)
    x_ach = ach.cer_pm.to_numpy(float)
    x_test = test.cer_pm.to_numpy(float)

    f_luo, f_ach, f_test = cdf(x_luo, GRID), cdf(x_ach, GRID), cdf(x_test, GRID)
    pi, rmse = fit_pi(f_test, f_luo, f_ach)

    # bootstrap over probe resamples (the dominant uncertainty: n=40 each)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        bl = cdf(rng.choice(x_luo, len(x_luo), replace=True), GRID)
        ba = cdf(rng.choice(x_ach, len(x_ach), replace=True), GRID)
        bt = cdf(rng.choice(x_test, len(x_test), replace=True), GRID)
        p, _ = fit_pi(bt, bl, ba)
        if np.isfinite(p):
            boot.append(p)
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # How well does a pure-Acholi model explain the test rows? (pi = 0 null)
    rmse_null = float(np.sqrt(np.mean((f_test - f_ach) ** 2)))

    print("=== mixture fit on the 785 lid=luo rows ===")
    print(f"pi (true-Dholuo fraction) = {pi:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"implied true-Dholuo rows  = {pi * len(x_test):.0f}   CI [{lo * len(x_test):.0f}, {hi * len(x_test):.0f}]")
    print(f"fit RMSE = {rmse:.4f}   vs pure-Acholi (pi=0) RMSE = {rmse_null:.4f}")
    print(f"currently converted by the hybrid: 35 rows ({35 / len(x_test):.1%})")
    print()
    print("CDF comparison (fraction of rows at or below each threshold):")
    print(f"{'thr':>6} {'F_luo':>7} {'F_ach':>7} {'F_test':>7} {'F_mix':>7}")
    for i, t in enumerate(GRID):
        if t in (0.1, 0.2, 0.3, 0.5, 0.7, 0.95, 1.5, 3.0, 8.0, 30.0, 120.0):
            mix = pi * f_luo[i] + (1 - pi) * f_ach[i]
            print(f"{t:>6.2f} {f_luo[i]:>7.2f} {f_ach[i]:>7.2f} {f_test[i]:>7.2f} {mix:>7.2f}")

    OUT.write_text(json.dumps({
        "pi": pi, "ci95": [float(lo), float(hi)],
        "implied_dholuo_rows": pi * len(x_test),
        "ci95_rows": [float(lo * len(x_test)), float(hi * len(x_test))],
        "rmse": rmse, "rmse_pure_acholi_null": rmse_null,
        "n_test": int(len(x_test)), "n_probe_luo": int(len(x_luo)), "n_probe_ach": int(len(x_ach)),
        "hybrid_accepts": 35,
        "caveat": "probes are n=40; FLEURS Dholuo is read speech, test rows are ~22s spontaneous",
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
