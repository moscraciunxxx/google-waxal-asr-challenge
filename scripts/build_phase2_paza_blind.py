#!/usr/bin/env python3
"""Build PAZA blind-swap candidates on top of the public best (wMjTA7Dz).

Candidate A (paza_ach437): replace lid=luo ∩ decode=ach rows EXCEPT the 39
  publicly-validated dual15 island rows with cleaned PAZA Dholuo decodes.
Candidate B (paza_all746): replace ALL lid=luo rows except the island.

Rationale: top-12 public cluster at 0.72 implies the luo mass is decodable at
~0.75 by a real Dholuo model; the repo's ban on Luo replacement came from
confounded experiments (conf-gated swaps / mms1b-ach downgrades / broken PAZA
harness). Public LB is the only oracle for spontaneous Dholuo. Base submission
stays selected; these are upload experiments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text

BASE = PROJECT_ROOT / "submission_phase2_nextiter_nyn.csv"
FLOOR = PROJECT_ROOT / "submission_phase2_selective_v3_dual15.csv"
V3_DETAIL = OUTPUT_DIR / "phase2_selective_v3_detail.csv"
OPENSET_DETAIL = OUTPUT_DIR / "phase2_openset_detail.csv"
PAZA = OUTPUT_DIR / "next_iter" / "paza_luo_hyps.csv"
SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"


def main():
    base = pd.read_csv(BASE)
    base["ID"] = base["ID"].astype(str)
    base = base.set_index("ID")["Target"].astype(str).map(lambda s: normalize_text(s) or ".")

    floor = pd.read_csv(FLOOR)
    floor["ID"] = floor["ID"].astype(str)
    floor = floor.set_index("ID")["Target"].astype(str).map(lambda s: normalize_text(s) or ".")
    v3 = pd.read_csv(V3_DETAIL)
    v3["ID"] = v3["ID"].astype(str)
    v3 = v3.set_index("ID")["prediction"].astype(str).map(lambda s: normalize_text(s) or ".")
    island = {uid for uid in floor.index if floor[uid] != v3.get(uid, floor[uid])}
    print(f"dual15 island preserved: {len(island)} rows")

    det = pd.read_csv(OPENSET_DETAIL)
    det["ID"] = det["ID"].astype(str)
    det = det.set_index("ID")

    paza = pd.read_csv(PAZA)
    paza["ID"] = paza["ID"].astype(str)
    paza = paza.set_index("ID")

    from scripts.paza_decode import clean_hyp

    def usable(uid, cap: float) -> str | None:
        if uid not in paza.index:
            return None
        raw = str(paza.loc[uid, "paza_raw"]) if "paza_raw" in paza.columns else str(paza.loc[uid, "paza_luo"])
        dur = float(paza.loc[uid, "dur"]) if "dur" in paza.columns else 15.0
        h = clean_hyp(normalize_text(raw) or "", dur, wps_cap=cap)
        if not h or h == "." or len(h.split()) < 1:
            return None
        return h

    ach_filter = lambda uid: det.loc[uid, "lid_lang"] == "luo" and det.loc[uid, "decode_lang"] == "ach"
    all_filter = lambda uid: det.loc[uid, "lid_lang"] == "luo"
    for tag, (pool_filter, cap) in {
        "paza_ach437_c16": (ach_filter, 1.6),
        "paza_all746_c16": (all_filter, 1.6),
        "paza_all746_c20": (all_filter, 2.0),
    }.items():
        pred = dict(base)
        n_replaced = n_skipped = 0
        for uid in base.index:
            if uid in island or not pool_filter(uid):
                continue
            h = usable(uid, cap)
            if h is None:
                n_skipped += 1
                continue
            if h != pred[uid]:
                pred[uid] = h
                n_replaced += 1
        out = PROJECT_ROOT / f"submission_phase2_{tag}.csv"
        df = pd.DataFrame({"ID": list(pred.keys()), "prediction": list(pred.values())})
        build_submission(df, sample_path=SAMPLE, out_path=out)
        rep = check_submission(out, SAMPLE)
        rep.update({"tag": tag, "n_replaced": n_replaced, "n_skipped": n_skipped, "island_kept": len(island), "base": str(BASE)})
        (OUTPUT_DIR / "next_iter" / f"{tag}_check.json").write_text(json.dumps(rep, indent=2, default=str))
        print(tag, "->", out.name, f"replaced={n_replaced} skipped={n_skipped} ok={rep['ok']}")


if __name__ == "__main__":
    main()
