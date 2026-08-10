#!/usr/bin/env python3
"""Finish the bounded sidecar decode sweep from its partial artifact."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

from sidecar_lug_decode_sweep import (
    CHUNK_OUT,
    GATE,
    OUT,
    PARTIAL,
    load_pipeline,
    merge_chunks,
    metric,
    split_metrics,
    wave_duration,
    write_wav_chunks,
)

ROOT = Path(__file__).resolve().parents[1]
HYPOTHESES = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/lug_gate/hypotheses.csv"


def main() -> None:
    result = json.loads(PARTIAL.read_text())
    manifest = pd.read_csv(GATE, dtype=str).fillna("")
    hypotheses = pd.read_csv(HYPOTHESES, dtype=str).fillna("").set_index("ID")
    manifest["baseline"] = [hypotheses.loc[uid, "baseline"] for uid in manifest["ID"]]
    manifest["duration_sec"] = [wave_duration(path) for path in manifest["audio"].tolist()]
    screen = manifest.iloc[result["protocol"]["decode_screen_indices"]].reset_index(drop=True)
    long_rows = manifest[manifest["duration_sec"] > 40.0].head(1).reset_index(drop=True)
    pipe = load_pipeline()
    default_cfg = pipe.beam_search_generator.config

    paths = screen["audio"].tolist()
    no_lang = pipe.transcribe(paths, lang=[None] * len(paths), batch_size=1)
    result["variants"]["no_language_tag_control"] = {
        "config": {"lang": None, "beam": "default"},
        "metrics": split_metrics(screen, list(no_lang)),
        "raw_hypotheses": list(no_lang),
    }
    PARTIAL.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"variant": "no_language_tag_control", "metrics": result["variants"]["no_language_tag_control"]["metrics"]}, ensure_ascii=False), flush=True)

    for scheme, chunk_seconds, overlap_seconds in (
        ("chunk30_overlap1", 30.0, 1.0),
        ("chunk20_overlap1", 20.0, 1.0),
    ):
        all_chunks: list[str] = []
        chunk_map: list[tuple[int, int, int]] = []
        for row_idx, row in long_rows.iterrows():
            chunks = write_wav_chunks(
                row["audio"],
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                tag=f"{scheme}_{row['ID']}",
            )
            start = len(all_chunks)
            all_chunks.extend(chunks)
            chunk_map.append((row_idx, start, len(all_chunks)))
        pipe.beam_search_generator.config = default_cfg
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
        result["long_audio"][scheme] = {
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "n_chunks": len(all_chunks),
            "candidate_metrics": candidate,
            "incumbent_metrics": incumbent,
            "delta_vs_incumbent": candidate["zindi"] - incumbent["zindi"],
            "rows": rows,
        }
        PARTIAL.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"long_scheme": scheme, "candidate": candidate, "incumbent": incumbent}, ensure_ascii=False), flush=True)

    result["model"] = {
        "repo": "mlai-dante/waxal-omniASR-LLM-1B-v2",
        "checkpoint": "step_1000/model",
        "device": "cpu",
        "valid_luganda_tag": "lug_Latn",
        "production_candidates_edited": False,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    PARTIAL.unlink(missing_ok=True)
    print(json.dumps({"artifact": str(OUT), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
