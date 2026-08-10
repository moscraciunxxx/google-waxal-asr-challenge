#!/usr/bin/env python3
"""Score guarded long-audio fusion against the complete locked Luganda gate."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np
from jiwer import cer, wer

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/lug_gate/hypotheses.csv"
LONG = ROOT / "outputs/goal_2026_08_10/sidecar_long_audio_all.json"
OUT = ROOT / "outputs/goal_2026_08_10/sidecar_long_guard_fusion.json"


def normalize(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"[^\w\s']+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def metric(refs: list[str], hyps: list[str]) -> dict[str, float | int]:
    refs = [normalize(x) for x in refs]
    hyps = [normalize(x) or "." for x in hyps]
    w, c = float(wer(refs, hyps)), float(cer(refs, hyps))
    return {"n": len(refs), "wer": w, "cer": c, "zindi": 1.0 - 0.5 * (w + c)}


def score_halves(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float | int]]:
    mid = len(frame) // 2
    return {
        "all": metric(frame.reference.tolist(), frame[column].tolist()),
        "tune": metric(frame.iloc[:mid].reference.tolist(), frame.iloc[:mid][column].tolist()),
        "holdout": metric(frame.iloc[mid:].reference.tolist(), frame.iloc[mid:][column].tolist()),
    }


def speaker_bootstrap(frame: pd.DataFrame, left: str, right: str, draws: int = 2000) -> dict[str, float | int]:
    rng = np.random.default_rng(20260810)
    speakers = np.array(sorted(frame["speaker_id"].astype(str).unique()))
    groups = {
        speaker: np.flatnonzero(frame["speaker_id"].astype(str).to_numpy() == speaker)
        for speaker in speakers
    }
    deltas = np.empty(draws, dtype=np.float64)
    refs = frame["reference"].tolist()
    lvals = frame[left].tolist()
    rvals = frame[right].tolist()
    for i in range(draws):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        idx = np.concatenate([groups[speaker] for speaker in sampled])
        rr = [refs[j] for j in idx]
        deltas[i] = metric(rr, [lvals[j] for j in idx])["zindi"] - metric(rr, [rvals[j] for j in idx])["zindi"]
    return {
        "draws": draws,
        "seed": 20260810,
        "delta_mean": float(deltas.mean()),
        "delta_p05": float(np.quantile(deltas, 0.05)),
        "delta_p50": float(np.quantile(deltas, 0.50)),
        "delta_p95": float(np.quantile(deltas, 0.95)),
        "probability_delta_positive": float(np.mean(deltas > 0)),
    }


def main() -> None:
    gate = pd.read_csv(GATE, dtype=str).fillna("")
    long = json.loads(LONG.read_text())
    merged = {row["ID"]: row["merged"] for row in long["rows"]}
    guard_rows = []
    guarded = []
    for _, row in gate.iterrows():
        uid = row["ID"]
        base = normalize(row["baseline"])
        if uid not in merged:
            guarded.append(normalize(row["candidate"]))
            continue
        chunked = normalize(merged[uid])
        ratio = len(chunked.split()) / max(1, len(base.split()))
        use_chunked = ratio <= 1.5
        guarded.append(chunked if use_chunked else base)
        guard_rows.append(
            {
                "ID": uid,
                "candidate_word_ratio_over_incumbent": ratio,
                "use_chunked": use_chunked,
                "fallback": not use_chunked,
            }
        )
    gate["guarded_long_candidate"] = guarded
    scores = score_halves(gate, "guarded_long_candidate")
    current = score_halves(gate, "candidate")
    incumbent = score_halves(gate, "baseline")
    report = {
        "protocol": {
            "source_gate": str(GATE),
            "source_long_audio": str(LONG),
            "rows": len(gate),
            "long_rows": len(guard_rows),
            "fixed_guard": "fallback to incumbent when merged chunk word count >1.5x incumbent word count",
            "labels_used": True,
            "phase2_labels_or_audio_used": False,
            "production_candidates_edited": False,
            "guard_threshold_selected_for_diagnostic_only": True,
        },
        "incumbent": incumbent,
        "current_1b_duration_guard": current,
        "guarded_long_fusion": scores,
        "delta_vs_current_1b_duration_guard": {
            part: scores[part]["zindi"] - current[part]["zindi"] for part in ("all", "tune", "holdout")
        },
        "delta_vs_incumbent": {
            part: scores[part]["zindi"] - incumbent[part]["zindi"] for part in ("all", "tune", "holdout")
        },
        "speaker_bootstrap_guarded_vs_current": speaker_bootstrap(
            gate, "guarded_long_candidate", "candidate", draws=2000
        ),
        "long_rows": guard_rows,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "artifact": str(OUT),
        "guarded_long_fusion": scores,
        "delta_vs_current": report["delta_vs_current_1b_duration_guard"],
        "delta_vs_incumbent": report["delta_vs_incumbent"],
        "long_rows": guard_rows,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
