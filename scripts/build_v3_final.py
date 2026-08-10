#!/usr/bin/env python3
"""Build the evidence-backed final candidate: v2_full + two small, measured changes.

Change 1 -- tighten the Dholuo agreement gate from 0.30 to 0.25.
  The 3-component mixture fit (Dholuo/Acholi/Luganda) puts the true-Dholuo weight at
  0.015 and gives the decision value of the gate at each threshold:
      thr 0.30 (shipped hybrid): P(Dholuo|accept)=0.36, private delta +0.00006
      thr 0.25                 : P(Dholuo|accept)=1.00, private delta +0.00276
  At 0.25 the measured false-positive rate is 0.000 on BOTH contaminant classes
  (Acholi and Luganda probes), so the tighter gate is simultaneously safer and
  higher-expected-value. The shipped hybrid sits at break-even.

Change 2 -- reassign lug-routed rows whose Luganda decode is gibberish under the
  Luganda LM (lm_lug <= -8) to the Acholi decode.
  On validation lm_lug <= -8 has FPR 0.000 on true Luganda and TPR 0.63 on true
  Acholi. Only 8 of the 308 lug-routed rows meet it -- confirming the router is
  right almost everywhere. Worth ~+0.0016 private; small, but positive and safe.

Deliberately NOT done: the d_lm > 1.7344 discriminator that proposed 229 flips. Its
threshold was calibrated where the classes sit at d_lm 0.10 vs 15.0, but the test rows
sit at 2.0-3.7 -- outside its support. Domain shift alone moves lm_lug from -1 to -3,
manufacturing false Acholi calls. See UPLOAD_DECISION.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = ROOT / "submission_phase2_v2_full.csv"
OUT = ROOT / "submission_phase2_v3_final.csv"
CHECK = ROOT / "outputs" / "next_iter" / "v3_final_check.json"

GATE_THR = 0.25
LM_LUG_CUT = -8.0


def main() -> None:
    sub = pd.read_csv(BASE)
    sub["Target"] = sub["Target"].astype(str)
    orig = sub.set_index("ID")["Target"].copy()
    tgt = sub.set_index("ID")["Target"].copy()

    # --- change 1: Dholuo gate at 0.25 ---
    agree = pd.read_csv(ROOT / "outputs" / "next_iter" / "hybrid_agreement_785.csv")
    acc = agree[agree.cer_pm <= GATE_THR]
    n_luo = 0
    for r in acc.itertuples(index=False):
        new = str(r.mms1b_luo).strip() or "."
        if r.ID in tgt.index and tgt.loc[r.ID].strip() != new:
            tgt.loc[r.ID] = new
            n_luo += 1

    # --- change 2: lug-routed rows whose Luganda decode is gibberish -> Acholi ---
    dec = pd.read_csv(ROOT / "outputs" / "next_iter" / "achlug_785_decisions.csv")
    flip = dec[(dec.router_lang == "lug") & (dec.lm_lug <= LM_LUG_CUT)]
    n_ach = 0
    for r in flip.itertuples(index=False):
        new = str(r.hyp_ach).strip() or "."
        if r.ID in tgt.index and tgt.loc[r.ID].strip() != new:
            tgt.loc[r.ID] = new
            n_ach += 1

    sub["Target"] = sub["ID"].map(tgt)
    assert len(sub) == 2392, len(sub)
    assert sub["Target"].notna().all() and (sub["Target"].str.strip() != "").all()
    sub.to_csv(OUT, index=False)

    changed = int((orig.loc[sub.ID].values != sub.Target.values).sum())
    meta = {
        "base": BASE.name, "out": OUT.name, "rows": len(sub),
        "dholuo_gate_thr": GATE_THR, "dholuo_rows_changed": n_luo,
        "lm_lug_cut": LM_LUG_CUT, "lug_to_ach_rows_changed": n_ach,
        "total_rows_changed_vs_v2_full": changed,
        "expected_private_delta": {
            "dholuo_gate_0.25_vs_0.30": 0.00276 - 0.00006,
            "lug_to_ach_8_rows": 8 * 0.3247 / 1674,
        },
    }
    CHECK.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
