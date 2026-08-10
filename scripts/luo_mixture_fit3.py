#!/usr/bin/env python3
"""Three-component mixture fit for the 785 lid=luo rows: Dholuo / Acholi / Luganda.

The two-component (Dholuo vs Acholi) fit was misspecified: the openset router assigns
308 of these 785 rows to the LUGANDA decoder, and true Luganda has its own, very
different agreement-statistic distribution (median cer_pm 6.6 vs Acholi 0.96) which
was measured only after the fact by probe_lug_falsepos.py.

Fit  F_test(t) ~= w_luo*F_luo(t) + w_ach*F_ach(t) + w_lug*F_lug(t)
subject to w >= 0, sum(w) = 1, by non-negative least squares on the CDF grid.

Then compute the decision value of the luo-swap gate under the fitted mixture, using
the measured loss matrix:
    G      = +0.569  zindi per correctly swapped (true Dholuo) row
    L_ach  = -0.314  zindi per wrongly swapped true-Acholi row
    L_lug  = -0.177  zindi per wrongly swapped true-Luganda row
and the measured per-class accept rates at each threshold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "next_iter" / "luo_mixture_fit3.json"

GRID = np.concatenate([np.arange(0.05, 1.0, 0.05),
                       np.array([1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 60.0, 120.0])])

G = 0.5692        # gain per correct swap (probe_ach_on_luo.py)
L_ACH = 0.3137    # loss per wrong swap, true Acholi
L_LUG = 0.1767    # loss per wrong swap, true Luganda
N = 785
THRS = [0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]


def cdf(x, grid):
    return np.array([(x <= t).mean() for t in grid])


def fit(f_test, comps):
    """NNLS with a sum-to-one constraint enforced by a heavy penalty row."""
    A = np.column_stack(comps)
    big = 50.0
    A_aug = np.vstack([A, big * np.ones((1, A.shape[1]))])
    b_aug = np.concatenate([f_test, [big]])
    w, _ = nnls(A_aug, b_aug)
    s = w.sum()
    if s > 0:
        w = w / s
    resid = float(np.sqrt(np.mean((f_test - A @ w) ** 2)))
    return w, resid


def main() -> None:
    luo = pd.read_csv("/tmp/probe_luo.csv").cer_pm.to_numpy(float)
    ach = pd.read_csv("/tmp/probe_ach.csv").cer_pm.to_numpy(float)
    lug = pd.read_csv(ROOT / "outputs" / "next_iter" / "lug_falsepos_probe.csv").cer_pm.to_numpy(float)
    test = pd.read_csv(ROOT / "outputs" / "next_iter" / "hybrid_agreement_785.csv").cer_pm.to_numpy(float)

    f = [cdf(x, GRID) for x in (luo, ach, lug)]
    f_test = cdf(test, GRID)
    w, resid = fit(f_test, f)

    # bootstrap over all four samples
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        fb = [cdf(rng.choice(x, len(x), replace=True), GRID) for x in (luo, ach, lug)]
        tb = cdf(rng.choice(test, len(test), replace=True), GRID)
        wb, _ = fit(tb, fb)
        boot.append(wb)
    boot = np.array(boot)
    ci = np.percentile(boot, [2.5, 97.5], axis=0)

    print("=== 3-component mixture fit on the 785 lid=luo rows ===")
    names = ["Dholuo", "Acholi", "Luganda"]
    for i, nm in enumerate(names):
        print(f"  {nm:>8}: w = {w[i]:.3f}  ({w[i]*N:5.0f} rows)   95% CI [{ci[0][i]:.3f}, {ci[1][i]:.3f}]")
    print(f"  fit RMSE = {resid:.4f}")
    print(f"  router assigns: 476 ach-route / 308 lug-route  (= 0.606 / 0.392)")
    print()

    # per-class accept rates at each threshold, and the resulting decision value
    print("=== decision value of the swap gate under the fitted mixture ===")
    print(f"{'thr':>6} {'TPR':>6} {'FPRach':>7} {'FPRlug':>7} {'accept':>7} {'pred_acc':>9} {'P(luo|A)':>9} {'EV/row':>8} {'total':>8}")
    rows = []
    for t in THRS:
        tpr = float((luo <= t).mean())
        fa = float((ach <= t).mean())
        fl = float((lug <= t).mean())
        n_luo = w[0] * N * tpr
        n_ach = w[1] * N * fa
        n_lug = w[2] * N * fl
        pred_acc = n_luo + n_ach + n_lug
        obs_acc = int((test <= t).sum())
        p_luo = n_luo / pred_acc if pred_acc > 0 else 0.0
        ev_row = (p_luo * G) - ((n_ach * L_ACH + n_lug * L_LUG) / pred_acc if pred_acc > 0 else 0.0)
        total = n_luo * G - n_ach * L_ACH - n_lug * L_LUG
        rows.append({"thr": t, "tpr": tpr, "fpr_ach": fa, "fpr_lug": fl,
                     "observed_accepts": obs_acc, "predicted_accepts": pred_acc,
                     "p_dholuo_given_accept": p_luo, "ev_per_row": ev_row, "ev_total_zindi_rows": total})
        print(f"{t:>6.2f} {tpr:>6.2f} {fa:>7.3f} {fl:>7.3f} {obs_acc:>7d} {pred_acc:>9.1f} {p_luo:>9.3f} {ev_row:>8.3f} {total:>8.2f}")

    print()
    print("total = summed zindi-row units gained; divide by 1674 private rows for private-score delta")
    for r in rows:
        r["private_score_delta"] = r["ev_total_zindi_rows"] / 1674.0
    print()
    print(f"{'thr':>6} {'private delta':>15}")
    for r in rows:
        print(f"{r['thr']:>6.2f} {r['private_score_delta']:>+15.5f}")

    OUT.write_text(json.dumps({
        "weights": {"dholuo": w[0], "acholi": w[1], "luganda": w[2]},
        "weights_ci95": {"dholuo": list(ci[:, 0]), "acholi": list(ci[:, 1]), "luganda": list(ci[:, 2])},
        "implied_dholuo_rows": w[0] * N, "rmse": resid,
        "loss_matrix": {"G": G, "L_ach": L_ACH, "L_lug": L_LUG},
        "by_threshold": rows,
    }, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
