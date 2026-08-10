#!/usr/bin/env python3
"""Evaluate one fixed long-audio policy on all locked >40-second Luganda rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from sidecar_lug_decode_sweep import (
    GATE,
    load_pipeline,
    merge_chunks,
    metric,
    wave_duration,
    write_wav_chunks,
)

ROOT = Path(__file__).resolve().parents[1]
HYPOTHESES = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/lug_gate/hypotheses.csv"
OUT = ROOT / "outputs/goal_2026_08_10/sidecar_long_audio_all.json"


def main() -> None:
    manifest = pd.read_csv(GATE, dtype=str).fillna("")
    hypotheses = pd.read_csv(HYPOTHESES, dtype=str).fillna("").set_index("ID")
    manifest["baseline"] = [hypotheses.loc[uid, "baseline"] for uid in manifest["ID"]]
    manifest["duration_sec"] = [wave_duration(path) for path in manifest["audio"]]
    long_rows = manifest[manifest["duration_sec"] > 40.0].reset_index(drop=True)
    if len(long_rows) != 5:
        raise RuntimeError(f"expected five locked >40-second rows, got {len(long_rows)}")

    all_chunks: list[str] = []
    chunk_map: list[tuple[int, int, int]] = []
    for row_idx, row in long_rows.iterrows():
        chunks = write_wav_chunks(
            row["audio"],
            chunk_seconds=30.0,
            overlap_seconds=1.0,
            tag=f"all_chunk30_overlap1_{row['ID']}",
        )
        start = len(all_chunks)
        all_chunks.extend(chunks)
        chunk_map.append((row_idx, start, len(all_chunks)))

    pipe = load_pipeline()
    chunk_hyps = pipe.transcribe(all_chunks, lang=["lug_Latn"] * len(all_chunks), batch_size=1)
    merged: list[str] = []
    rows: list[dict] = []
    for row_idx, start, stop in chunk_map:
        row_hyps = list(chunk_hyps[start:stop])
        merged_text = merge_chunks(row_hyps)
        merged.append(merged_text)
        rows.append(
            {
                "ID": long_rows.iloc[row_idx]["ID"],
                "duration_sec": float(long_rows.iloc[row_idx]["duration_sec"]),
                "chunk_paths": all_chunks[start:stop],
                "chunk_hypotheses": row_hyps,
                "merged": merged_text,
            }
        )
    candidate = metric(long_rows["reference"].tolist(), merged)
    incumbent = metric(long_rows["reference"].tolist(), long_rows["baseline"].tolist())
    report = {
        "protocol": {
            "source": str(GATE),
            "locked_long_rows": len(long_rows),
            "threshold_seconds": 40.0,
            "chunk_seconds": 30.0,
            "overlap_seconds": 1.0,
            "merge": "conservative exact word suffix/prefix overlap, max 12 words",
            "labels_used": True,
            "phase2_labels_or_audio_used": False,
            "production_candidates_edited": False,
        },
        "candidate": candidate,
        "incumbent": incumbent,
        "delta_vs_incumbent": candidate["zindi"] - incumbent["zindi"],
        "rows": rows,
        "model": {
            "repo": "mlai-dante/waxal-omniASR-LLM-1B-v2",
            "checkpoint": "step_1000/model",
            "device": "cpu",
            "lang": "lug_Latn",
        },
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "artifact": str(OUT),
        "candidate": candidate,
        "incumbent": incumbent,
        "delta_vs_incumbent": report["delta_vs_incumbent"],
        "row_ids": [row["ID"] for row in rows],
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
