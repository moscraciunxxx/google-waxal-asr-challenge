#!/usr/bin/env python3
"""Build a conservative Shona consensus overlay on the proven Luganda base.

Only rows where BadrEx and Mubarak agree after the competition text
normalization are replaced.  The overlay is intentionally tiny: it does not
ship either specialist's full Phase-2 decode, which was not public-validated.
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


def norm(s: str) -> str:
    return re.sub(r"[^a-z ]", "", str(s).lower())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
        default=ROOT / "outputs" / "goal_2026_08_06" / "hyps_sna_mubarak_whisper.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "submission_phase2_sna_consensus_lug_splitjoin.csv",
    )
    ap.add_argument(
        "--meta",
        type=Path,
        default=ROOT / "outputs" / "goal_2026_08_06" / "sna_consensus_meta.json",
    )
    args = ap.parse_args()

    base = {r["ID"]: r["Target"] for r in csv.DictReader(args.base.open())}
    routes = {
        r["ID"]: r
        for r in csv.DictReader(args.routes.open())
        if r.get("decode_lang") == "sna"
    }
    bad = {r["ID"]: r["Target"] for r in csv.DictReader(args.badrex.open())}
    mub = {r["ID"]: r["Target"] for r in csv.DictReader(args.mubarak.open())}

    replacements = []
    for uid in sorted(routes):
        if uid not in base or uid not in bad or uid not in mub:
            continue
        if norm(bad[uid]) != norm(mub[uid]) or norm(base[uid]) == norm(bad[uid]):
            continue
        replacements.append(
            {
                "ID": uid,
                "before": base[uid],
                "after": bad[uid],
                "base_badrex_similarity": SequenceMatcher(
                    None, norm(base[uid]), norm(bad[uid])
                ).ratio(),
            }
        )

    out_rows = []
    replacement_map = {r["ID"]: r["after"] for r in replacements}
    for uid, target in base.items():
        out_rows.append({"ID": uid, "Target": replacement_map.get(uid, target)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "Target"])
        w.writeheader()
        w.writerows(out_rows)

    meta = {
        "base": str(args.base),
        "out": str(args.out),
        "rows": len(out_rows),
        "n_changed": len(replacements),
        "rule": "new Sna only; normalized BadrEx == normalized Mubarak and differs from base",
        "strict": {
            "unique_ids": len({r["ID"] for r in out_rows}) == len(out_rows),
            "empty_targets": sum(not r["Target"].strip() for r in out_rows),
        },
        "sha256": sha256(args.out),
        "replacements": replacements,
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
