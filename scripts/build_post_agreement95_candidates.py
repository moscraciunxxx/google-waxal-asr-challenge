#!/usr/bin/env python3
"""Build BadrEx candidates isolated from the failed agreement95 submission."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM99 = ROOT / "outputs" / "goal_2026_08_07" / "badrex_tiers" / "submission_phase2_badrex_sna_sim99_lug_splitjoin.csv"
SIM98 = ROOT / "outputs" / "goal_2026_08_07" / "badrex_tiers" / "submission_phase2_badrex_sna_sim98_lug_splitjoin.csv"
FAILED = ROOT / "outputs" / "goal_2026_08_07" / "badrex_mubarak_agreement" / "submission_phase2_bad99_then_agree95_lug_splitjoin.csv"
PUBLIC = ROOT / "outputs" / "beat075" / "public_visible_index.csv"
OUT_DIR = ROOT / "outputs" / "goal_2026_08_08" / "post_agreement95"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_candidate(path: Path, ordered_ids: list[str], targets: dict[str, str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        for uid in ordered_ids:
            writer.writerow({"ID": uid, "Target": targets[uid]})


def main() -> int:
    sim99_rows = rows(SIM99)
    ordered_ids = [r["ID"] for r in sim99_rows]
    sim99 = {r["ID"]: r["Target"] for r in sim99_rows}
    sim98 = {r["ID"]: r["Target"] for r in rows(SIM98)}
    failed = {r["ID"]: r["Target"] for r in rows(FAILED)}
    public = {r["ID"] for r in rows(PUBLIC)}

    sim98_additions = {uid for uid in ordered_ids if sim98[uid] != sim99[uid]}
    failed_additions = {uid for uid in ordered_ids if failed[uid] != sim99[uid]}
    exact_failed_targets = {
        uid
        for uid in sim98_additions & failed_additions
        if sim98[uid] == failed[uid]
    }

    policies = {
        "sim98_exclude_failed_exact3": sim98_additions - exact_failed_targets,
        "sim98_zero_failed_id_overlap16": sim98_additions - failed_additions,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "sim99": str(SIM99),
        "sim98": str(SIM98),
        "failed_agreement95": str(FAILED),
        "sim98_additions": len(sim98_additions),
        "failed_additions": len(failed_additions),
        "exact_failed_target_ids": sorted(exact_failed_targets),
        "candidates": {},
    }
    for name, selected in policies.items():
        targets = dict(sim99)
        for uid in selected:
            targets[uid] = sim98[uid]
        out = OUT_DIR / f"submission_phase2_{name}_lug_splitjoin.csv"
        write_candidate(out, ordered_ids, targets)
        changed = {uid for uid in ordered_ids if targets[uid] != sim99[uid]}
        meta["candidates"][name] = {
            "path": str(out),
            "n_additions_vs_sim99": len(changed),
            "n_public_additions": len(changed & public),
            "sha256": sha256(out),
        }

    meta_path = OUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
