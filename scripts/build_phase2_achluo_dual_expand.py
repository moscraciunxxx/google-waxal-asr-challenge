#!/usr/bin/env python3
"""Ach-route Luo dual-agree expansion on floor selective_v3_dual15.

Rules:
- Never rewrite decode_lang==lug (frozen public FT-v3 path)
- Only lid_lang==luo & decode_lang==ach
- Overlay = MMS-1B luo when CLEAR∩MMS char-CER <= thr and p1 >= min_p1
- Base = public floor CSV
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def cer(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(la, lb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.18)
    ap.add_argument("--min-p1", type=float, default=0.99)
    ap.add_argument("--top-k", type=int, default=0, help="If >0, take best-k NEW by cer_mc among thr pool")
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument(
        "--floor",
        type=Path,
        default=ROOT / "submission_phase2_selective_v3_dual15.csv",
    )
    args = ap.parse_args()

    floor = pd.read_csv(args.floor)
    v3 = pd.read_csv(ROOT / "submission_phase2_selective_v3.csv")
    det = pd.read_csv(ROOT / "outputs/phase2_selective_v3_detail.csv")
    mms = pd.read_csv(ROOT / "outputs/phase2_luo_mms1b_detail.csv")
    clr = pd.read_csv(ROOT / "outputs/phase2_selective_clear_allluo_detail.csv")
    frozen = set(det.loc[det.decode_lang == "lug", "ID"])

    m = (
        det.merge(mms[["ID", "prediction"]].rename(columns={"prediction": "mms"}), on="ID")
        .merge(
            clr[["ID", "prediction", "source"]].rename(
                columns={"prediction": "clr", "source": "clr_src"}
            ),
            on="ID",
        )
        .merge(floor.rename(columns={"Target": "floor"}), on="ID")
        .merge(v3.rename(columns={"Target": "sel_v3"}), on="ID")
    )
    pool = m[(m.lid_lang == "luo") & (m.decode_lang == "ach") & (m.clr_src == "clear_luo")].copy()
    pool["cer_mc"] = [cer(a, b) for a, b in zip(pool.mms, pool.clr)]
    pool["already_dual"] = pool.floor != pool.sel_v3
    cand = pool[~pool.already_dual]
    sub = cand[(cand.cer_mc <= args.thr) & (cand.p1 >= args.min_p1)].sort_values(
        ["cer_mc", "p1"], ascending=[True, False]
    )
    if args.top_k > 0:
        sub = sub.head(args.top_k)
    ids = set(sub.ID)

    out = floor.copy()
    mmap = pool.set_index("ID")["mms"]
    for i, row in out.iterrows():
        if row.ID in ids:
            out.at[i, "Target"] = mmap[row.ID]

    path = ROOT / f"submission_phase2_selective_v3_dual15_achluo_{args.name}.csv"
    out.to_csv(path, index=False)

    ft = floor.set_index("ID")["Target"]
    ch = {row.ID for _, row in out.iterrows() if row.Target != ft[row.ID]}
    assert not (ch & frozen), "frozen lug touched"
    rep = {
        "path": str(path),
        "n_changed_vs_floor": len(ch),
        "thr": args.thr,
        "min_p1": args.min_p1,
        "top_k": args.top_k,
        "touched_frozen_lug": 0,
        "ok": True,
    }
    (ROOT / "outputs" / f"phase2_achluo_{args.name}_check.json").write_text(
        json.dumps(rep, indent=2)
    )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
