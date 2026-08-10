#!/usr/bin/env python3
"""Validate the two current expanded Phase-2 final candidates.

This is intentionally separate from the historical 1,500-row Phase-1/Phase-2
tests. It fails closed on missing IDs, duplicates, placeholders, and stale row
sets, then writes a small manifest suitable for code review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submission import check_phase2_submission, phase2_expected_ids


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    root = ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--safe",
        type=Path,
        default=root / "outputs/goal_2026_08_09/final_lug_domain_raw/submission_phase2_public_lin_w2vbert_sna_w2vbert_nyn_guarded.csv",
    )
    ap.add_argument(
        "--swing",
        type=Path,
        default=root / "outputs/goal_2026_08_09/final_lug_domain_raw/submission_phase2_private_lin_w2vbert_sna_w2vbert_nyn_guarded_luo2.csv",
    )
    ap.add_argument("--out", type=Path, default=root / "outputs/goal_2026_08_06/submission_manifest.json")
    args = ap.parse_args()

    manifest = {
        "phase": "phase2-expanded",
        "expected_rows": len(phase2_expected_ids()),
        "safe": {"path": str(args.safe), "public_score": None, "public_score_status": "not_uploaded_by_validator"},
        "swing": {"path": str(args.swing), "public_score": None, "public_score_status": "not_uploaded_by_validator"},
        "rules": {"no_phase1_test_gold": True, "open_source_only": True, "two_private_finals": True},
    }
    for key, path in (("safe", args.safe), ("swing", args.swing)):
        if not path.exists():
            raise SystemExit(f"Missing {key} candidate: {path}")
        report = check_phase2_submission(path, strict=True)
        manifest[key]["check"] = report
        manifest[key]["sha256"] = sha256(path)
        if not report["ok"]:
            raise SystemExit(json.dumps(report, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
