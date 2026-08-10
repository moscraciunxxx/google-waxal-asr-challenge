#!/usr/bin/env python3
"""Build the deadline candidate from the proven 0.707243556 submission.

The only production edit is the locked-validation-backed post-processing
change that removes a terminal Luganda filler token (``aa``). No other
language or private-only route is changed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PUBLIC = ROOT / "outputs/goal_2026_08_10/final_mlai_lug_subset_on_current/submission_phase2_public_lin_w2vbert_sna_w2vbert_nyn_guarded.csv"
BASE_PRIVATE = ROOT / "outputs/goal_2026_08_10/final_mlai_lug_subset_on_current/submission_phase2_private_lin_w2vbert_sna_w2vbert_nyn_guarded_luo2.csv"
CANONICAL_PUBLIC = ROOT / "outputs/goal_2026_08_09/final_lug_domain_normalized/submission_phase2_public_lin_w2vbert_sna_w2vbert_nyn_guarded.csv"
ROUTE_INDEX = ROOT / "outputs/beat075/public_visible_index.csv"
OUT_DIR = ROOT / "outputs/goal_2026_08_10/final_deadline_lug_strip_aa"
OUT_PUBLIC = OUT_DIR / "submission_phase2_public_deadline_lug_strip_aa.csv"
OUT_PRIVATE = OUT_DIR / "submission_phase2_private_deadline_lug_strip_aa_luo2.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames != ["ID", "Target"]:
            raise ValueError(f"Unexpected header in {path}: {reader.fieldnames}")
    return list(reader.fieldnames), rows


def read_any_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    _, current_public = read_csv(BASE_PUBLIC)
    _, current_private = read_csv(BASE_PRIVATE)
    _, canonical = read_csv(CANONICAL_PUBLIC)
    _, route_rows = read_any_csv(ROUTE_INDEX)
    canonical_by_id = {row["ID"]: row["Target"] for row in canonical}
    lug_ids = {row["ID"] for row in route_rows if row.get("decode_lang") == "lug"}

    if len(current_public) != 2392 or len(current_private) != 2392:
        raise ValueError("Phase-2 row count changed; refusing to build candidate")
    if [row["ID"] for row in current_public] != [row["ID"] for row in current_private]:
        raise ValueError("Public/private ID order differs")

    edits: list[dict[str, str]] = []
    # Apply only to Luganda rows, and only to a terminal standalone filler.
    # Do not touch any other language or any non-terminal occurrence of 'aa'.
    for row in current_public:
        old = canonical_by_id.get(row["ID"])
        target = row["Target"]
        if old is None:
            raise ValueError(f"ID missing from canonical base: {row['ID']}")
        if row["ID"] in lug_ids and re.search(r"\saa$", target):
            new_target = re.sub(r"\saa$", "", target)
            edits.append({"ID": row["ID"], "before": target, "after": new_target})

    edit_by_id = {item["ID"]: item["after"] for item in edits}
    public_out = [
        {"ID": row["ID"], "Target": edit_by_id.get(row["ID"], row["Target"])}
        for row in current_public
    ]
    private_out = [
        {"ID": row["ID"], "Target": edit_by_id.get(row["ID"], row["Target"])}
        for row in current_private
    ]
    write_csv(OUT_PUBLIC, public_out)
    write_csv(OUT_PRIVATE, private_out)

    metadata = {
        "policy": "proven_0p707243556_base_plus_terminal_lug_aa_strip_all_lug_rows",
        "base_public": str(BASE_PUBLIC.relative_to(ROOT)),
        "base_private": str(BASE_PRIVATE.relative_to(ROOT)),
        "canonical_overlay_base": str(CANONICAL_PUBLIC.relative_to(ROOT)),
        "route_index": str(ROUTE_INDEX.relative_to(ROOT)),
        "source_public_sha256": sha256(BASE_PUBLIC),
        "source_private_sha256": sha256(BASE_PRIVATE),
        "output_public_sha256": sha256(OUT_PUBLIC),
        "output_private_sha256": sha256(OUT_PRIVATE),
        "rows": len(public_out),
        "edited_rows": len(edits),
        "edited_ids": [item["ID"] for item in edits],
        "edits": edits,
        "locked_validation_evidence": {
            "source": "outputs/goal_2026_08_10/sidecar_text_fusion.json",
            "candidate_strip_aa_zindi": 0.9213969605423674,
            "candidate_current_zindi": 0.9208050396474674,
            "all_delta": 0.0005919209148999778,
            "holdout_delta": 0.0,
            "baseline_strip_aa_zindi": 0.900016620271246,
            "baseline_current_zindi": 0.8988327784814459,
            "baseline_all_delta": 0.001183841789800089,
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
