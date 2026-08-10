#!/usr/bin/env python3
"""Build conservative BadrEx Shona overlays on the current public-best spine.

The full BadrEx new-Sna swap is an untested high-risk candidate.  This script
creates similarity-gated tiers from the 2,392-row public-best CSV and records
their exact public-visible overlap.  Similarity is computed after the same
competition text normalization used by the decode pipeline; exact agreement
with the independent Mubarak Shona decode is tracked as a second signal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(text: str) -> str:
    return re.sub(r"[^a-z ]", "", str(text).lower())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_map(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return {r["ID"]: r["Target"] for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=Path,
        default=ROOT / "submission_phase2_beat075_primary_lug_splitjoin.csv",
    )
    ap.add_argument(
        "--routes",
        type=Path,
        default=ROOT / "outputs" / "next_iter" / "new_routes.csv",
    )
    ap.add_argument(
        "--badrex",
        type=Path,
        default=ROOT / "outputs" / "goal_2026_08_06" / "hyps_sna_badrex.csv",
    )
    ap.add_argument(
        "--mubarak",
        type=Path,
        default=ROOT
        / "outputs"
        / "goal_2026_08_06"
        / "hyps_sna_mubarak_whisper.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "goal_2026_08_07" / "badrex_tiers",
    )
    args = ap.parse_args()

    base_rows = list(csv.DictReader(args.base.open(newline="")))
    base = {r["ID"]: r["Target"] for r in base_rows}
    bad = read_map(args.badrex)
    mub = read_map(args.mubarak)
    routes = {
        r["ID"]: r
        for r in csv.DictReader(args.routes.open(newline=""))
        if r.get("decode_lang") == "sna"
    }
    public = {
        r["ID"]
        for r in csv.DictReader(
            (ROOT / "outputs" / "beat075" / "public_visible_index.csv").open(
                newline=""
            )
        )
    }

    scored = []
    for uid in sorted(routes):
        if uid not in base or uid not in bad:
            continue
        before = norm(base[uid])
        after = norm(bad[uid])
        if not after or after == before:
            continue
        scored.append(
            {
                "ID": uid,
                "similarity": SequenceMatcher(None, before, after).ratio(),
                "agreement_mubarak": uid in mub and after == norm(mub[uid]),
                "before": base[uid],
                "after": bad[uid],
            }
        )

    policies = {
        "sim99": lambda r: r["similarity"] >= 0.99,
        "sim98": lambda r: r["similarity"] >= 0.98,
        "sim97": lambda r: r["similarity"] >= 0.97,
        "sim95": lambda r: r["similarity"] >= 0.95,
        "consensus": lambda r: r["agreement_mubarak"],
        "full": lambda r: True,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tier_meta = {}
    for name, keep in policies.items():
        selected = [r for r in scored if keep(r)]
        selected_map = {r["ID"]: r["after"] for r in selected}
        out = args.out_dir / f"submission_phase2_badrex_sna_{name}_lug_splitjoin.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ID", "Target"])
            w.writeheader()
            for row in base_rows:
                w.writerow({"ID": row["ID"], "Target": selected_map.get(row["ID"], row["Target"])})
        changed = [r for r in selected if selected_map[r["ID"]] != base[r["ID"]]]
        tier_meta[name] = {
            "path": str(out),
            "rows": len(base_rows),
            "n_new_sna_changed": len(changed),
            "n_public_visible_changed": sum(r["ID"] in public for r in changed),
            "n_mubarak_agree": sum(r["agreement_mubarak"] for r in changed),
            "policy": name,
            "sha256": sha256(out),
        }

    meta = {
        "base": str(args.base),
        "routes": str(args.routes),
        "badrex": str(args.badrex),
        "mubarak": str(args.mubarak),
        "candidate_rows": len(base_rows),
        "new_sna_changed_by_badrex": len(scored),
        "public_visible_ids": len(public),
        "tiers": tier_meta,
        "comparison": {
            "public_best_score": 0.687889452,
            "strict_upload_target": "> 0.687889452",
            "mubarak_full_public_score": 0.684026798,
            "badrex_matched_val_delta": 0.03035765915828903,
        },
        "rows": scored,
    }
    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"meta": str(meta_path), "tiers": tier_meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
