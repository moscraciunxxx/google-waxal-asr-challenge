#!/usr/bin/env python3
"""Bounded text-normalization and fusion diagnostics for the 1B Luganda gate.

This script reads only the immutable locked-gate hypotheses and writes only
sidecar diagnostics.  It does not edit or rebuild any production candidate.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

import pandas as pd
from jiwer import cer, wer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text_norm import normalize_text


GATE = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/lug_gate/hypotheses.csv"
OUT = ROOT / "outputs/goal_2026_08_10/sidecar_text_fusion.json"


def metric(refs: list[str], hyps: list[str]) -> dict[str, float | int]:
    refs = [normalize_text(x) for x in refs]
    hyps = [normalize_text(x) or "." for x in hyps]
    w, c = float(wer(refs, hyps)), float(cer(refs, hyps))
    return {"n": len(refs), "wer": w, "cer": c, "zindi": 1.0 - 0.5 * (w + c)}


def split_metrics(frame: pd.DataFrame, col: str) -> dict[str, dict[str, float | int]]:
    return {
        "all": metric(frame.reference.tolist(), frame[col].tolist()),
        "tune": metric(frame.iloc[:20].reference.tolist(), frame.iloc[:20][col].tolist()),
        "holdout": metric(frame.iloc[20:].reference.tolist(), frame.iloc[20:][col].tolist()),
    }


def strip_trailing_fillers(text: str, fillers: set[str]) -> str:
    words = normalize_text(text).split()
    while words and words[-1] in fillers:
        words.pop()
    return " ".join(words)


def collapse_adjacent_duplicates(text: str) -> str:
    words = normalize_text(text).split()
    out: list[str] = []
    for word in words:
        if not out or out[-1] != word:
            out.append(word)
    return " ".join(out)


def adjacent_duplicate(text: str) -> bool:
    words = normalize_text(text).split()
    return any(a == b for a, b in zip(words, words[1:]))


def word_ratio(candidate: str, incumbent: str) -> float:
    c, b = len(normalize_text(candidate).split()), len(normalize_text(incumbent).split())
    return c / max(1, b)


def corpus_delta(frame: pd.DataFrame, candidate_col: str, baseline_col: str) -> dict[str, float]:
    c = split_metrics(frame, candidate_col)
    b = split_metrics(frame, baseline_col)
    return {part: c[part]["zindi"] - b[part]["zindi"] for part in ("all", "tune", "holdout")}


def main() -> None:
    frame = pd.read_csv(GATE, dtype=str).fillna("")
    required = {"ID", "speaker_id", "reference", "baseline", "raw_candidate", "candidate"}
    missing = required - set(frame.columns)
    if missing or len(frame) != 40:
        raise RuntimeError(f"locked gate shape/columns invalid: rows={len(frame)}, missing={sorted(missing)}")

    # Text-only alternatives.  The filler rules are diagnostics, not blind
    # edits: they are reported separately and must pass both locked halves.
    variants: dict[str, list[str]] = {
        "candidate_current": [normalize_text(x) for x in frame.raw_candidate],
        "candidate_no_apostrophe": [normalize_text(x).replace("'", " ") for x in frame.raw_candidate],
        "candidate_strip_aa": [strip_trailing_fillers(x, {"aa"}) for x in frame.raw_candidate],
        "candidate_strip_aa_ah": [strip_trailing_fillers(x, {"aa", "ah"}) for x in frame.raw_candidate],
        "candidate_collapse_adjacent_dup": [collapse_adjacent_duplicates(x) for x in frame.raw_candidate],
    }
    for name, values in variants.items():
        frame[name] = values

    text_metrics = {name: split_metrics(frame, name) for name in variants}

    # Validation-tuned, low-complexity fusion policies.  Each policy uses only
    # observable candidate/incumbent text features and is scored on the locked
    # tune and holdout halves independently.
    candidate = frame.candidate_current.tolist()
    incumbent = [normalize_text(x) for x in frame.baseline]
    ratios = [word_ratio(c, b) for c, b in zip(candidate, incumbent)]
    trailing_aa = [bool(re.search(r"(?:^| )aa$", normalize_text(x))) for x in candidate]
    trailing_filler = [bool(re.search(r"(?:^| )(?:aa|ah|ee|eh)$", normalize_text(x))) for x in candidate]
    duplicate = [adjacent_duplicate(x) for x in candidate]
    similarity = [
        difflib.SequenceMatcher(None, normalize_text(c), normalize_text(b)).ratio()
        for c, b in zip(candidate, incumbent)
    ]

    policies: dict[str, list[str]] = {
        "candidate_all": candidate,
        "incumbent_all": incumbent,
        "incumbent_if_trailing_aa": [b if bad else c for c, b, bad in zip(candidate, incumbent, trailing_aa)],
        "incumbent_if_trailing_filler": [b if bad else c for c, b, bad in zip(candidate, incumbent, trailing_filler)],
    }
    for threshold in (1.20, 1.30, 1.40, 1.60, 1.80):
        policies[f"incumbent_if_word_ratio_gt_{threshold:.2f}"] = [
            b if r > threshold else c for c, b, r in zip(candidate, incumbent, ratios)
        ]
    for threshold in (0.55, 0.65, 0.75, 0.85):
        policies[f"incumbent_if_word_ratio_lt_{threshold:.2f}"] = [
            b if r < threshold else c for c, b, r in zip(candidate, incumbent, ratios)
        ]
    for threshold in (0.40, 0.55, 0.70, 0.85):
        policies[f"incumbent_if_similarity_lt_{threshold:.2f}"] = [
            b if s < threshold else c for c, b, s in zip(candidate, incumbent, similarity)
        ]
    policies["incumbent_if_duplicate_and_ratio_gt_1.20"] = [
        b if d and r > 1.20 else c for c, b, d, r in zip(candidate, incumbent, duplicate, ratios)
    ]

    fusion_rows = []
    for name, values in policies.items():
        frame[name] = values
        scores = split_metrics(frame, name)
        fusion_rows.append(
            {
                "policy": name,
                "switches": sum(a != b for a, b in zip(values, candidate)),
                **{f"{part}_zindi": scores[part]["zindi"] for part in ("all", "tune", "holdout")},
                **{f"{part}_delta_vs_candidate": scores[part]["zindi"] - text_metrics["candidate_current"][part]["zindi"] for part in ("all", "tune", "holdout")},
            }
        )

    # Oracle is a diagnostic upper bound only; it is never a deployable route.
    oracle = []
    for ref, c, b in zip(frame.reference, candidate, incumbent):
        mc = metric([ref], [c])["zindi"]
        mb = metric([ref], [b])["zindi"]
        oracle.append(c if mc >= mb else b)

    feature_rows = [
        {
            "ID": uid,
            "word_ratio_candidate_over_incumbent": ratio,
            "similarity_candidate_incumbent": sim,
            "trailing_aa": aa,
            "trailing_filler": fill,
            "adjacent_duplicate": dup,
        }
        for uid, ratio, sim, aa, fill, dup in zip(
            frame.ID, ratios, similarity, trailing_aa, trailing_filler, duplicate
        )
    ]
    report = {
        "protocol": {
            "source": str(GATE),
            "rows": len(frame),
            "locked_split": "first 20 tune, last 20 holdout",
            "labels_used": True,
            "phase2_labels_or_audio_used": False,
            "production_candidates_edited": False,
        },
        "normalization": text_metrics,
        "fusion": sorted(fusion_rows, key=lambda x: (x["tune_zindi"], x["holdout_zindi"]), reverse=True),
        "candidate_vs_incumbent": corpus_delta(frame, "candidate_current", "baseline"),
        "oracle_best_of_candidate_incumbent": split_metrics(pd.DataFrame({"reference": frame.reference, "oracle": oracle}), "oracle"),
        "features": feature_rows,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "artifact": str(OUT),
        "candidate_vs_incumbent": report["candidate_vs_incumbent"],
        "top_fusion": report["fusion"][:8],
        "normalization": {k: v["all"] for k, v in text_metrics.items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
