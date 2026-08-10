#!/usr/bin/env python3
"""Decode a bounded public Luganda subset and materialize a full fallback cache."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/subset"
INDEX = ROOT / "outputs/beat075/public_visible_index.csv"
BASE = ROOT / "outputs/goal_2026_08_08/nyn_ensemble/submission_phase2_nyn_cv_ensemble_guarded.csv"


def load_pipeline():
    loader = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/omni_cpu_screen.py"
    spec = importlib.util.spec_from_file_location("omni_screen", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load 1B loader")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_pipeline()


def main() -> None:
    import os

    limit = int(os.environ.get("MLAI_SUBSET_ROWS", "40"))
    route = pd.read_csv(INDEX, dtype=str).fillna("")
    route = route[route.decode_lang.eq("lug")].copy().reset_index(drop=True)
    base = pd.read_csv(BASE, dtype=str).fillna("")
    targets = dict(zip(base.ID, base.Target))
    selected = route.head(limit).copy()
    progress = OUT / "subset_progress.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    done = {}
    if progress.exists():
        old = pd.read_csv(progress, dtype=str).fillna("")
        done = dict(zip(old.ID, old.Target))
    todo = selected[~selected.ID.isin(done)].copy()
    started = time.time()
    if len(todo):
        pipe = load_pipeline()
        for n, row in enumerate(todo.itertuples(index=False), start=1):
            hyp = pipe.transcribe([row.audio], lang=["lug_Latn"], batch_size=1)[0]
            done[row.ID] = str(hyp).strip() or "."
            pd.DataFrame({"ID": list(done), "Target": list(done.values())}).to_csv(progress, index=False)
            print(json.dumps({"decoded": n, "todo": len(todo), "ID": row.ID, "elapsed_sec": time.time() - started}), flush=True)
    cache = pd.DataFrame({"ID": route.ID, "Target": [done.get(uid, targets[uid]) for uid in route.ID]})
    cache_path = OUT / "route_cache_mlai_1b_lug_subset_fallback.csv"
    cache.to_csv(cache_path, index=False)
    manifest = {
        "model": "mlai-dante/waxal-omniASR-LLM-1B-v2",
        "decoded_rows": len(done),
        "fallback_rows": len(route) - len(done),
        "selection": "first public-visible Luganda route rows",
        "max_generation_length": 4096,
        "phase2_labels_used": False,
        "upload_performed": False,
        "cache": str(cache_path),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
