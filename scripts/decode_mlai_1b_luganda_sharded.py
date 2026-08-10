#!/usr/bin/env python3
"""Decode the public Luganda route with resumable CPU model shards."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import wave
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/phase2_sharded"
INDEX = ROOT / "outputs/beat075/public_visible_index.csv"
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


def worker(rank: int, workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    route = pd.read_csv(INDEX, dtype=str).fillna("")
    route = route[route.decode_lang.eq("lug")].copy().reset_index(drop=True)
    base = pd.read_csv(BASE, dtype=str).fillna("").set_index("ID")
    shard = route.iloc[rank::workers].copy().reset_index(drop=True)
    shard_path = OUT / f"shard_{rank:02d}.csv"
    if shard_path.exists():
        old = pd.read_csv(shard_path, dtype=str).fillna("")
        if len(old) == len(shard) and set(old.ID) == set(shard.ID) and old.Target.str.strip().ne("").all():
            print(json.dumps({"rank": rank, "status": "reused", "rows": len(old)}), flush=True)
            return

    durations = [duration(p) for p in shard.audio]
    targets = [str(base.loc[uid, "Target"]) for uid in shard.ID]
    short = [i for i, seconds in enumerate(durations) if seconds <= 40.0]
    pipe = load_pipeline()
    for pos, i in enumerate(short):
        hyp = pipe.transcribe(
            [str(shard.iloc[i].audio)],
            lang=["lug_Latn"],
            batch_size=1,
        )[0]
        targets[i] = str(hyp).strip() or "."
        if (pos + 1) % 4 == 0 or pos + 1 == len(short):
            pd.DataFrame({"ID": shard.ID.tolist(), "Target": targets}).to_csv(shard_path, index=False)
            print(json.dumps({"rank": rank, "decoded": pos + 1, "total": len(short), "path": str(shard_path)}), flush=True)
    pd.DataFrame({"ID": shard.ID.tolist(), "Target": targets}).to_csv(shard_path, index=False)
    manifest = {
        "rank": rank,
        "workers": workers,
        "rows": len(shard),
        "decoded_with_mlai_1b": len(short),
        "duration_guard_rows": len(shard) - len(short),
        "model": "mlai-dante/waxal-omniASR-LLM-1B-v2",
        "max_generation_length": 4096,
        "phase2_labels_used": False,
        "cache": str(shard_path),
        "cache_sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
    }
    (OUT / f"shard_{rank:02d}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def main() -> None:
    workers = int(os.environ.get("MLAI_WORKERS", "4"))
    if workers < 1 or workers > 8:
        raise ValueError("MLAI_WORKERS must be in [1, 8]")
    worker_id = os.environ.get("MLAI_WORKER")
    if worker_id is not None:
        worker(int(worker_id), workers)
        return
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(workers):
        env = os.environ.copy()
        env["MLAI_WORKER"] = str(rank)
        # This launcher is intentionally replaced below by a simple subprocess
        # fan-out in the shell; retaining the worker entry point makes each
        # shard independently resumable.
        del env
        p = ctx.Process(target=worker, args=(rank, workers))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
