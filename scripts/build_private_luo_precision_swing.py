#!/usr/bin/env python3
"""Build private-swing CSV with Anv-calibrated Luo detector + precision overlays.

Overlays (fail-closed, private lid=luo island only for public impact):
  1) **Anv-ke/Dholuo** FPR-capped gate decisions
     (`outputs/goal_2026_08_06/anv_luo_gate_decisions.csv`) from
     `scripts/anv_luo_calibrate_and_ft.py` — required unless --allow-missing-anv.
  2) Dholuo PAZA∩MMS agreement thr <= 0.25 (hybrid_agreement_785)
  3) Supported lm_lug <= -8 lug→ach flips (achlug_785_decisions)

Primary supervision is NOT Phase-2 self-pseudo FT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANV_DEC = ROOT / "outputs" / "goal_2026_08_06" / "anv_luo_gate_decisions.csv"
ANV_CAL = ROOT / "outputs" / "goal_2026_08_06" / "anv_luo_calibration.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=ROOT / "submission_phase2_v2_full.csv")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "submission_phase2_private_swing_luo_precision.csv",
    )
    ap.add_argument("--dholuo-thr", type=float, default=0.25)
    ap.add_argument("--lm-lug-cut", type=float, default=-8.0)
    ap.add_argument(
        "--allow-missing-anv",
        action="store_true",
        help="Skip Anv overlay if decisions missing (default: require Anv artifacts)",
    )
    ap.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help="Optional meta JSON path (default: private_swing_meta.json under goal outputs)",
    )
    args = ap.parse_args()

    sub = pd.read_csv(args.base)
    sub["ID"] = sub["ID"].astype(str)
    sub["Target"] = sub["Target"].astype(str)
    orig = sub.set_index("ID")["Target"].copy()
    tgt = orig.copy()

    n_anv = 0
    anv_thr = None
    anv_fpr = None
    if ANV_DEC.exists():
        anv = pd.read_csv(ANV_DEC)
        if "accept" not in anv.columns:
            raise SystemExit(f"Anv decisions missing accept column: {ANV_DEC}")
        acc = anv[anv["accept"].astype(str).str.lower().isin(["true", "1"])]
        for r in acc.itertuples(index=False):
            uid = str(r.ID)
            new = str(getattr(r, "overlay_text", "") or getattr(r, "mms1b_luo", "") or "").strip() or "."
            if uid in tgt.index and tgt.loc[uid].strip() != new:
                tgt.loc[uid] = new
                n_anv += 1
        if ANV_CAL.exists():
            cal = json.loads(ANV_CAL.read_text())
            anv_thr = cal.get("thr")
            anv_fpr = cal.get("fpr_at_thr")
    elif not args.allow_missing_anv:
        raise SystemExit(
            f"Missing required Anv gate decisions at {ANV_DEC}. "
            "Run scripts/anv_luo_calibrate_and_ft.py first, or pass --allow-missing-anv."
        )

    n_dholuo = 0
    agree_path = ROOT / "outputs" / "next_iter" / "hybrid_agreement_785.csv"
    if agree_path.exists():
        agree = pd.read_csv(agree_path)
        acc = agree[agree.cer_pm <= args.dholuo_thr]
        for r in acc.itertuples(index=False):
            uid = str(r.ID)
            new = str(r.mms1b_luo).strip() or "."
            if uid in tgt.index and tgt.loc[uid].strip() != new:
                tgt.loc[uid] = new
                n_dholuo += 1

    n_ach = 0
    dec_path = ROOT / "outputs" / "next_iter" / "achlug_785_decisions.csv"
    if dec_path.exists():
        dec = pd.read_csv(dec_path)
        flip = dec[(dec.router_lang == "lug") & (dec.lm_lug <= args.lm_lug_cut)]
        for r in flip.itertuples(index=False):
            uid = str(r.ID)
            new = str(r.hyp_ach).strip() or "."
            if uid in tgt.index and tgt.loc[uid].strip() != new:
                tgt.loc[uid] = new
                n_ach += 1

    sub["Target"] = sub["ID"].map(tgt)
    assert len(sub) == 2392
    assert sub["Target"].notna().all() and (sub["Target"].str.strip() != "").all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)

    changed = int((orig.loc[sub.ID].values != sub.Target.values).sum())
    meta = {
        "base": str(args.base.name),
        "out": str(args.out.name),
        "rows": len(sub),
        "anv_decisions": str(ANV_DEC.relative_to(ROOT)) if ANV_DEC.exists() else None,
        "anv_rows_changed": n_anv,
        "anv_thr": anv_thr,
        "anv_fpr_at_thr": anv_fpr,
        "dholuo_thr": args.dholuo_thr,
        "dholuo_rows_changed": n_dholuo,
        "lm_lug_cut": args.lm_lug_cut,
        "lug_to_ach_rows_changed": n_ach,
        "total_rows_changed_vs_base": changed,
        "public_impact": "none expected (old lid=luo private-only)",
        "recipe": (
            "Anv-ke unscripted FT detector (FPR-capped) + hybrid thr0.25 + lm_lug<=-8; "
            "no Phase-2 self-pseudo primary"
        ),
    }
    meta_path = args.meta_out or (
        ROOT / "outputs" / "goal_2026_08_06" / "private_swing_meta.json"
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
