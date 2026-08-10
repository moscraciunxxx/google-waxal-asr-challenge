#!/usr/bin/env python3
"""Build conservative two-model Shona overlays.

Priority is intentional: preserve the already public-validated BadrEx sim99
rows first, then use Mubarak only when its transcript is close to the
independent BadrEx transcript.  This avoids the previously rejected full
Mubarak swap while testing the validation-supported agreement signal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from jiwer import process_words

ROOT = Path(__file__).resolve().parents[1]


def norm(text: str) -> str:
    return re.sub(r"[^a-z ]", "", str(text).lower())


def read_map(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return {r["ID"]: r["Target"] for r in csv.DictReader(f)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=ROOT / "submission_phase2_beat075_primary_lug_splitjoin.csv")
    ap.add_argument("--routes", type=Path, default=ROOT / "outputs" / "next_iter" / "new_routes.csv")
    ap.add_argument("--badrex", type=Path, default=ROOT / "outputs" / "goal_2026_08_06" / "hyps_sna_badrex.csv")
    ap.add_argument("--mubarak", type=Path, default=ROOT / "outputs" / "goal_2026_08_06" / "hyps_sna_mubarak_whisper.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "goal_2026_08_07" / "badrex_mubarak_agreement")
    args = ap.parse_args()

    base_rows = list(csv.DictReader(args.base.open(newline="")))
    base = {r["ID"]: r["Target"] for r in base_rows}
    bad = read_map(args.badrex)
    mub = read_map(args.mubarak)
    sna_ids = {
        r["ID"]
        for r in csv.DictReader(args.routes.open(newline=""))
        if r.get("decode_lang") == "sna"
    }
    public = {
        r["ID"]
        for r in csv.DictReader((ROOT / "outputs" / "beat075" / "public_visible_index.csv").open(newline=""))
    }

    candidates = []
    for uid in sorted(sna_ids & base.keys() & bad.keys() & mub.keys()):
        b = norm(base[uid])
        x = norm(bad[uid])
        m = norm(mub[uid])
        if not x or x == b:
            continue
        word_ops = process_words(b, m)
        base_badrex_word_ops = process_words(b, x)
        mubarak_badrex_word_ops = process_words(m, x)
        candidates.append(
            {
                "ID": uid,
                "base_badrex_similarity": SequenceMatcher(None, b, x).ratio(),
                "badrex_mubarak_similarity": SequenceMatcher(None, x, m).ratio(),
                "base_mubarak_similarity": SequenceMatcher(None, b, m).ratio(),
                "base_word_count": len(b.split()),
                "mubarak_word_count": len(m.split()),
                "word_substitutions": word_ops.substitutions,
                "word_insertions": word_ops.insertions,
                "word_deletions": word_ops.deletions,
                "base_badrex_word_distance": (
                    base_badrex_word_ops.substitutions
                    + base_badrex_word_ops.insertions
                    + base_badrex_word_ops.deletions
                ),
                "mubarak_badrex_word_distance": (
                    mubarak_badrex_word_ops.substitutions
                    + mubarak_badrex_word_ops.insertions
                    + mubarak_badrex_word_ops.deletions
                ),
                "mubarak_badrex_word_substitutions": (
                    mubarak_badrex_word_ops.substitutions
                ),
                "mubarak_badrex_word_insertions": mubarak_badrex_word_ops.insertions,
                "mubarak_badrex_word_deletions": mubarak_badrex_word_ops.deletions,
                "badrex": bad[uid],
                "mubarak": mub[uid],
            }
        )

    policies = {
        # The sim99 rows are the only BadrEx rows with public score evidence.
        "bad99_then_agree99": {"agreement_threshold": 0.99, "guard": "all"},
        "bad99_then_agree98": {"agreement_threshold": 0.98, "guard": "all"},
        "bad99_then_agree95": {"agreement_threshold": 0.95, "guard": "all"},
        "bad99_then_agree90": {"agreement_threshold": 0.90, "guard": "all"},
        "bad99_then_agree95_wordeq": {
            "agreement_threshold": 0.95,
            "guard": "equal_word_count",
        },
        "bad99_then_agree95_noinsdel": {
            "agreement_threshold": 0.95,
            "guard": "zero_word_insertions_deletions",
        },
        "bad99_then_agree95_noinsdel_sub2": {
            "agreement_threshold": 0.95,
            "guard": "zero_word_insertions_deletions_and_at_most_2_substitutions",
        },
        "bad99_then_agree95_samecount_wordcloser": {
            "agreement_threshold": 0.95,
            "guard": "same_word_count_and_strictly_closer_to_badrex_by_word_distance",
        },
        "bad99_then_agree95_bm_noinsdel_dist2": {
            "agreement_threshold": 0.95,
            "guard": "badrex_mubarak_zero_insertions_deletions_and_distance_at_most_2",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"base": str(args.base), "rows": len(base_rows), "public_ids": len(public), "policies": {}}
    for name, policy in policies.items():
        agree_threshold = policy["agreement_threshold"]
        guard = policy["guard"]
        replacements = {}
        source_counts = {"badrex": 0, "mubarak": 0}
        for row in candidates:
            if row["base_badrex_similarity"] >= 0.99:
                replacements[row["ID"]] = row["badrex"]
                source_counts["badrex"] += 1
            elif row["badrex_mubarak_similarity"] >= agree_threshold and (
                guard == "all"
                or (
                    guard == "equal_word_count"
                    and row["base_word_count"] == row["mubarak_word_count"]
                )
                or (
                    guard == "zero_word_insertions_deletions"
                    and row["word_insertions"] == 0
                    and row["word_deletions"] == 0
                )
                or (
                    guard
                    == "zero_word_insertions_deletions_and_at_most_2_substitutions"
                    and row["word_insertions"] == 0
                    and row["word_deletions"] == 0
                    and row["word_substitutions"] <= 2
                )
                or (
                    guard
                    == "same_word_count_and_strictly_closer_to_badrex_by_word_distance"
                    and row["base_word_count"] == row["mubarak_word_count"]
                    and row["mubarak_badrex_word_distance"]
                    < row["base_badrex_word_distance"]
                )
                or (
                    guard
                    == "badrex_mubarak_zero_insertions_deletions_and_distance_at_most_2"
                    and row["mubarak_badrex_word_insertions"] == 0
                    and row["mubarak_badrex_word_deletions"] == 0
                    and row["mubarak_badrex_word_distance"] <= 2
                )
            ):
                replacements[row["ID"]] = row["mubarak"]
                source_counts["mubarak"] += 1

        out = args.out_dir / f"submission_phase2_{name}_lug_splitjoin.csv"
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
            writer.writeheader()
            for row in base_rows:
                writer.writerow({"ID": row["ID"], "Target": replacements.get(row["ID"], row["Target"])})

        changed = set(replacements)
        meta["policies"][name] = {
            "agreement_threshold": agree_threshold,
            "guard": guard,
            "n_changed": len(changed),
            "n_public_changed": len(changed & public),
            "source_counts": source_counts,
            "sha256": sha256(out),
            "path": str(out),
        }

    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
