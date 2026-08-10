#!/usr/bin/env python3
"""Decode the public-visible Luganda route with the validated WAXAL 1B model."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import wave
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/phase2"
ROUTE_INDEX = ROOT / "outputs/beat075/public_visible_index.csv"
BASE = ROOT / "outputs/goal_2026_08_08/nyn_ensemble/submission_phase2_nyn_cv_ensemble_guarded.csv"


def load_pipeline():
    loader_path = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/omni_cpu_screen.py"
    spec = importlib.util.spec_from_file_location("mlai_omni_cpu_screen", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_pipeline()


def duration(path: str) -> float:
    with wave.open(path) as handle:
        return handle.getnframes() / handle.getframerate()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    route = pd.read_csv(ROUTE_INDEX, dtype=str).fillna("")
    route = route[route["decode_lang"].eq("lug")].copy()
    if len(route) != 433 or route["ID"].duplicated().any():
        raise RuntimeError(f"expected 433 unique Luganda route rows, got {len(route)}")
    base = pd.read_csv(BASE, dtype=str).fillna("").set_index("ID")
    if not set(route.ID).issubset(base.index):
        raise RuntimeError("route IDs are not all present in the base submission")
    paths = route["audio"].tolist()
    durations = [duration(path) for path in paths]
    short_idx = [i for i, seconds in enumerate(durations) if seconds <= 40.0]
    pipe = load_pipeline()
    short_paths = [paths[i] for i in short_idx]
    # Batch four is still CPU-safe for this checkpoint and materially reduces
    # overhead relative to the batch-one locked gate.
    targets = [str(base.loc[uid, "Target"]) for uid in route.ID]
    partial_path = OUT / "route_cache_mlai_1b_lug_duration_guard_raw.partial.csv"
    chunk_size = 20
    for start in range(0, len(short_paths), chunk_size):
        stop = min(start + chunk_size, len(short_paths))
        chunk_raw = pipe.transcribe(
            short_paths[start:stop],
            lang=["lug_Latn"] * (stop - start),
            batch_size=4,
        )
        for i, text in zip(short_idx[start:stop], chunk_raw):
            targets[i] = str(text).strip() or "."
        pd.DataFrame({"ID": route.ID.tolist(), "Target": targets}).to_csv(partial_path, index=False)
        print(json.dumps({"decoded_short_rows": stop, "total_short_rows": len(short_idx), "partial": str(partial_path)}), flush=True)
    cache = pd.DataFrame({"ID": route.ID.tolist(), "Target": targets})
    cache_path = OUT / "route_cache_mlai_1b_lug_duration_guard_raw.csv"
    cache.to_csv(cache_path, index=False)
    manifest = {
        "model": "mlai-dante/waxal-omniASR-LLM-1B-v2",
        "checkpoint": "step_1000/model",
        "device": "cpu",
        "route": "lug",
        "rows": len(cache),
        "decoded_with_mlai_1b": len(short_idx),
        "incumbent_duration_guard": len(cache) - len(short_idx),
        "max_single_audio_seconds": 40.0,
        "id_sha256": hashlib.sha256("\n".join(cache.ID).encode()).hexdigest(),
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "cache": str(cache_path),
        "base": str(BASE),
        "phase2_labels_used": False,
        "upload_performed": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
