#!/usr/bin/env python3
"""Cheap threshold sensitivity for the fixed long-audio length guard."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from jiwer import cer, wer

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/lug_gate/hypotheses.csv"
LONG = ROOT / "outputs/goal_2026_08_10/sidecar_long_audio_all.json"
OUT = ROOT / "outputs/goal_2026_08_10/sidecar_long_guard_thresholds.json"


def n(x: object) -> str:
    value = unicodedata.normalize("NFKC", str(x or "")).lower()
    value = re.sub(r"[^\w\s']+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def score(refs: list[str], hyps: list[str]) -> float:
    refs = [n(x) for x in refs]
    hyps = [n(x) or "." for x in hyps]
    return 1.0 - 0.5 * (float(wer(refs, hyps)) + float(cer(refs, hyps)))


def main() -> None:
    gate = pd.read_csv(GATE, dtype=str).fillna("")
    long = json.loads(LONG.read_text())
    merged = {row["ID"]: n(row["merged"]) for row in long["rows"]}
    ratios = {}
    for uid, candidate in merged.items():
        base = n(gate.loc[gate.ID.eq(uid), "baseline"].iloc[0])
        ratios[uid] = len(candidate.split()) / max(1, len(base.split()))
    current = [n(x) for x in gate["candidate"]]
    incumbent = [n(x) for x in gate["baseline"]]
    refs = gate["reference"].tolist()
    rows = []
    for threshold in (1.10, 1.15, 1.20, 1.30, 1.50, 2.00, 2.50, 2.75, 3.00):
        out = []
        switches = 0
        for uid, c, b in zip(gate.ID, current, incumbent):
            if uid in merged and ratios[uid] <= threshold:
                out.append(merged[uid])
                switches += 1
            else:
                out.append(b if uid in merged else c)
        rows.append(
            {
                "threshold": threshold,
                "long_chunk_switches": switches,
                "zindi": score(refs, out),
                "delta_vs_current_1b_duration_guard": score(refs, out) - score(refs, current),
            }
        )
    report = {
        "protocol": {
            "source_gate": str(GATE),
            "source_long_audio": str(LONG),
            "thresholds": [1.10, 1.15, 1.20, 1.30, 1.50, 2.00, 2.50, 2.75, 3.00],
            "production_candidates_edited": False,
        },
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
