#!/usr/bin/env python3
"""Validate the curated public Phase-2 CSV without local competition data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = list(reader.fieldnames or [])

    errors: list[str] = []
    if columns != ["ID", "Target"]:
        errors.append(f"expected columns ['ID', 'Target'], got {columns}")
    ids = [str(row.get("ID", "")) for row in rows]
    if len(rows) != 2392:
        errors.append(f"expected 2392 rows, got {len(rows)}")
    if len(ids) != len(set(ids)):
        errors.append("duplicate IDs")
    if any(not row.get("ID", "").startswith("ID_") for row in rows):
        errors.append("all IDs must start with ID_")
    bad_targets = [
        row.get("ID", "")
        for row in rows
        if str(row.get("Target", "")).strip().lower() in {"", ".", "nan", "null", "none"}
    ]
    if bad_targets:
        errors.append(f"empty/placeholder Target values: {bad_targets[:5]}")
    return {
        "path": str(path),
        "rows": len(rows),
        "columns": columns,
        "sha256": sha256(path),
        "empty_or_placeholder_targets": len(bad_targets),
        "ok": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "submission" / "phase2_public_final.csv",
    )
    args = parser.parse_args()
    report = validate(args.submission)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
