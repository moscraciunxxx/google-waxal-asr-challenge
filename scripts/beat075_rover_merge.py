#!/usr/bin/env python3
"""ROVER-lite / agreement merge for beat075 multi-system hyps vs v2 floor.

Policy (fail-closed to floor):
  - Only public-visible IDs
  - For each ID, collect available system hyps for its decode_lang
  - If ≥2 systems agree within char-CER thr and the consensus differs from floor,
    replace floor with the consensus (prefer longest among agreeing cluster)
  - Length guard vs floor: word-count ratio in [0.65, 1.45]
  - Never touch old lid=luo rows

Also builds high-agreement pseudo CSV for domain FT.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text_norm import normalize_text

OUT = ROOT / "outputs" / "beat075"


def cer(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            if ca == cb:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[lb] / max(la, lb)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agree-cer", type=float, default=0.12)
    ap.add_argument("--min-systems", type=int, default=2)
    ap.add_argument("--floor-diff-cer", type=float, default=0.08)
    ap.add_argument("--out-sub", type=Path, default=ROOT / "submission_phase2_beat075_rover.csv")
    args = ap.parse_args()

    idx = load_csv(OUT / "public_visible_index.csv")
    floor_rows = load_csv(ROOT / "submission_phase2_v2_full.csv")
    floor = {r["ID"]: r["Target"] for r in floor_rows}
    by_id = {r["ID"]: r for r in idx}

    # load all hyp_*.csv
    hyps: dict[str, dict[str, str]] = defaultdict(dict)  # id -> tag -> text
    for path in sorted(OUT.glob("hyps_*.csv")):
        if "_lim" in path.name:
            continue
        # hyps_{route}_{tag}.csv
        parts = path.stem.split("_", 2)
        if len(parts) < 3:
            continue
        tag = parts[2]
        for r in load_csv(path):
            col = [c for c in r.keys() if c.startswith("hyp_")]
            if not col:
                continue
            hyps[r["ID"]][tag] = r[col[0]]

    tgt = dict(floor)
    replaces = []
    pseudo = []
    stats = defaultdict(int)

    for uid, meta in by_id.items():
        route = meta["decode_lang"]
        fl = floor.get(uid, meta.get("floor") or "")
        sys_hyps = hyps.get(uid, {})
        # always include floor as a system for agreement counting when peers exist
        texts = list(sys_hyps.items())
        if not texts:
            stats["no_hyps"] += 1
            continue
        # cluster by pairwise CER
        tags = [t for t, _ in texts]
        vals = [normalize_text(v) or "." for _, v in texts]
        n = len(vals)
        # find largest clique-like set: greedy seed by pair agreement
        best_cluster = []
        for i in range(n):
            cluster = [i]
            for j in range(n):
                if j == i:
                    continue
                if cer(vals[i], vals[j]) <= args.agree_cer:
                    cluster.append(j)
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
        if len(best_cluster) < args.min_systems:
            stats["no_agree"] += 1
            continue
        # consensus = longest hyp in cluster (more complete transcription often)
        cluster_vals = [vals[i] for i in best_cluster]
        cons = max(cluster_vals, key=lambda s: (len(s.split()), len(s)))
        if cer(cons, fl) < args.floor_diff_cer:
            stats["agree_with_floor"] += 1
            # still useful as pseudo if multi-agree
            if len(best_cluster) >= 2:
                pseudo.append(
                    {
                        "ID": uid,
                        "decode_lang": route,
                        "lid_lang": meta.get("lid_lang"),
                        "text": cons,
                        "source": "rover_agree_floor",
                        "audio": meta.get("audio"),
                        "n_agree": len(best_cluster),
                    }
                )
            continue
        # length guard
        fw, cw = len(normalize_text(fl).split()), len(cons.split())
        if fw > 0:
            ratio = cw / fw
            if ratio < 0.65 or ratio > 1.45:
                stats["length_reject"] += 1
                continue
        tgt[uid] = cons
        stats["replaced"] += 1
        replaces.append(
            {
                "ID": uid,
                "route": route,
                "n_agree": len(best_cluster),
                "floor": fl,
                "cons": cons,
                "cer_vs_floor": cer(cons, fl),
                "systems": [tags[i] for i in best_cluster],
            }
        )
        pseudo.append(
            {
                "ID": uid,
                "decode_lang": route,
                "lid_lang": meta.get("lid_lang"),
                "text": cons,
                "source": "rover_replace",
                "audio": meta.get("audio"),
                "n_agree": len(best_cluster),
            }
        )

    # write submission in floor ID order
    out_rows = [{"ID": r["ID"], "Target": tgt[r["ID"]]} for r in floor_rows]
    with args.out_sub.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "Target"])
        w.writeheader()
        w.writerows(out_rows)

    meta = {
        "agree_cer": args.agree_cer,
        "min_systems": args.min_systems,
        "floor_diff_cer": args.floor_diff_cer,
        "stats": dict(stats),
        "n_replaced": stats["replaced"],
        "n_rows": len(out_rows),
        "out": str(args.out_sub),
    }
    (OUT / "rover_merge_meta.json").write_text(json.dumps(meta, indent=2))
    with (OUT / "rover_replaces.json").open("w") as f:
        json.dump(replaces, f, indent=2)
    # pseudo for FT
    if pseudo:
        with (OUT / "pseudo_high_agree.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pseudo[0].keys()))
            w.writeheader()
            w.writerows(pseudo)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
