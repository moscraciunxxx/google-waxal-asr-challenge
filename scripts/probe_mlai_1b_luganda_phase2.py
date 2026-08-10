#!/usr/bin/env python3
"""Bounded CPU smoke for the WAXAL 1B model on public Luganda audio."""

from __future__ import annotations

import importlib.util
import json
import time
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    route = pd.read_csv(ROOT / "outputs/beat075/public_visible_index.csv", dtype=str)
    limit = int(os.environ.get("PROBE_LIMIT", "3"))
    route = route[route.decode_lang.eq("lug")].head(limit)
    loader = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/omni_cpu_screen.py"
    spec = importlib.util.spec_from_file_location("omni_screen", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load 1B screen loader")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    started = time.time()
    pipe = mod.load_pipeline()
    loaded = time.time()
    hyps = pipe.transcribe(route.audio.tolist(), lang=["lug_Latn"] * len(route), batch_size=1)
    result = {
        "rows": [{"ID": uid, "audio": audio, "hypothesis": hyp} for uid, audio, hyp in zip(route.ID, route.audio, hyps)],
        "load_seconds": loaded - started,
        "decode_seconds": time.time() - loaded,
        "max_generation_length": int(pipe.model.max_generation_length),
        "test_labels_used": False,
    }
    out = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/phase2_probe_3.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(out), **{k: result[k] for k in ("load_seconds", "decode_seconds", "max_generation_length")}}, indent=2))


if __name__ == "__main__":
    main()
