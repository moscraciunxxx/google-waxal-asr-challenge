#!/usr/bin/env python3
"""Probe alternate MLAI 1B beam settings on locked Luganda validation only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
from jiwer import cer, wer

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/omni_cpu_screen.py"
MANIFEST = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/locked_screen_manifest.csv"
OUT = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/beam_probe"
AUDIO_ROOT = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval"


def load_pipeline():
    spec = importlib.util.spec_from_file_location("omni_screen", LOADER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load MLAI loader")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_pipeline(), mod.normalize_text


def main() -> None:
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from omnilingual_asr.models.wav2vec2_llama.beamsearch import Wav2Vec2LlamaBeamSearchSeq2SeqGenerator
    from omnilingual_asr.models.wav2vec2_llama.config import Wav2Vec2LlamaBeamSearchConfig

    del ASRInferencePipeline  # import checks the installed public runtime
    manifest = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    rows = manifest[manifest.lang.eq("lug")].head(10).copy()
    rows["audio"] = rows["audio"].map(lambda p: str(AUDIO_ROOT / p))
    pipe, normalize = load_pipeline()
    # Keep the exploratory beam probe bounded.  The 1B decoder can otherwise
    # spend several minutes on a single malformed long hypothesis.
    # The longest locked audio context is ~1.5k decoder frames, so 2,048 is
    # the smallest safe bound for this probe.
    pipe.model.max_generation_length = 2048
    results = []
    for length_norm in (False, True):
        config = Wav2Vec2LlamaBeamSearchConfig(
            nbest=3,
            length_norm=length_norm,
            compression_window=50,
            compression_threshold=2.5,
        )
        pipe.beam_search_generator = Wav2Vec2LlamaBeamSearchSeq2SeqGenerator(
            model=pipe.model,
            config=config,
            streaming_config=pipe.streaming_config,
        )
        hyps = pipe.transcribe(rows.audio.tolist(), lang=["lug_Latn"] * len(rows), batch_size=1)
        refs = [normalize(x) for x in rows.reference.tolist()]
        pred = [normalize(x) for x in hyps]
        results.append({
            "nbest": 3,
            "length_norm": length_norm,
            "wer": float(wer(refs, pred)),
            "cer": float(cer(refs, pred)),
            "zindi": float(1.0 - 0.5 * (wer(refs, pred) + cer(refs, pred))),
            "predictions": [{"ID": uid, "Target": hyp} for uid, hyp in zip(rows.ID, pred)],
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps({"rows": len(rows), "results": results}, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "results": results}, indent=2))


if __name__ == "__main__":
    main()
