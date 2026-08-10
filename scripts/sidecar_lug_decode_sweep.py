#!/usr/bin/env python3
"""Bounded 1B Luganda inference diagnostics.

The script exercises only the locked 40-row validation gate.  It writes test
audio chunks and JSON diagnostics under ``outputs/goal_2026_08_10/sidecar_*``
and never touches a submission candidate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import wave
from pathlib import Path

import pandas as pd
from jiwer import cer, wer

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/lug_gate/manifest.csv"
OUT = ROOT / "outputs/goal_2026_08_10/sidecar_decode_sweep.json"
PARTIAL = ROOT / "outputs/goal_2026_08_10/sidecar_decode_sweep.partial.json"
CHUNK_OUT = ROOT / "outputs/goal_2026_08_10/sidecar_long_audio"


def normalize(text: object) -> str:
    import unicodedata

    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"[^\w\s']+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def metric(refs: list[str], hyps: list[str]) -> dict[str, float | int]:
    refs = [normalize(x) for x in refs]
    hyps = [normalize(x) or "." for x in hyps]
    w, c = float(wer(refs, hyps)), float(cer(refs, hyps))
    return {"n": len(refs), "wer": w, "cer": c, "zindi": 1.0 - 0.5 * (w + c)}


def load_pipeline():
    loader_path = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/omni_cpu_screen.py"
    spec = importlib.util.spec_from_file_location("mlai_omni_cpu_screen", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_pipeline()


def write_wav_chunks(path: str, *, chunk_seconds: float, overlap_seconds: float, tag: str) -> list[str]:
    CHUNK_OUT.mkdir(parents=True, exist_ok=True)
    with wave.open(path) as src:
        nframes = src.getnframes()
        framerate = src.getframerate()
        nchannels = src.getnchannels()
        sampwidth = src.getsampwidth()
        payload = src.readframes(nframes)
    chunk_frames = max(1, int(round(chunk_seconds * framerate)))
    overlap_frames = max(0, int(round(overlap_seconds * framerate)))
    step = max(1, chunk_frames - overlap_frames)
    out: list[str] = []
    start = 0
    index = 0
    while start < nframes:
        stop = min(nframes, start + chunk_frames)
        data = payload[start * nchannels * sampwidth : stop * nchannels * sampwidth]
        target = CHUNK_OUT / f"{tag}_{index:02d}.wav"
        with wave.open(str(target), "wb") as dst:
            dst.setnchannels(nchannels)
            dst.setsampwidth(sampwidth)
            dst.setframerate(framerate)
            dst.writeframes(data)
        out.append(str(target))
        index += 1
        if stop == nframes:
            break
        start += step
    return out


def merge_chunks(texts: list[str]) -> str:
    """Merge chunk hypotheses with a conservative exact word overlap."""
    merged: list[str] = []
    for text in texts:
        words = normalize(text).split()
        if not words:
            continue
        overlap = 0
        max_overlap = min(12, len(merged), len(words))
        for size in range(max_overlap, 0, -1):
            if merged[-size:] == words[:size]:
                overlap = size
                break
        merged.extend(words[overlap:])
    return " ".join(merged)


def split_metrics(frame: pd.DataFrame, hyps: list[str]) -> dict[str, dict[str, float | int]]:
    refs = frame["reference"].tolist()
    return {
        "all": metric(refs, hyps),
        "tune": metric(refs[: len(refs) // 2], hyps[: len(refs) // 2]),
        "holdout": metric(refs[len(refs) // 2 :], hyps[len(refs) // 2 :]),
    }


def main() -> None:
    manifest = pd.read_csv(GATE, dtype=str).fillna("")
    if len(manifest) != 40 or manifest["ID"].duplicated().any():
        raise RuntimeError(f"locked gate invalid: {len(manifest)} rows")
    manifest["duration_sec"] = [
        wave_duration(path) for path in manifest["audio"].tolist()
    ]
    # Two rows with equal tune/holdout coverage keep this CPU-only sweep
    # bounded while retaining a genuine out-of-sample control.
    screen_idx = [0, 20]
    screen = manifest.iloc[screen_idx].reset_index(drop=True)
    long_rows = manifest[manifest["duration_sec"] > 40.0].head(1).reset_index(drop=True)
    result: dict = {
        "protocol": {
            "source": str(GATE),
            "gate_rows": len(manifest),
            "decode_screen_rows": len(screen),
            "decode_screen_indices": screen_idx,
            "decode_screen_tune_rows": 1,
            "decode_screen_holdout_rows": 1,
            "long_rows": len(long_rows),
            "labels_used": True,
            "phase2_labels_or_audio_used": False,
            "production_candidates_edited": False,
        },
        "variants": {},
        "long_audio": {},
    }
    pipe = load_pipeline()
    from omnilingual_asr.models.wav2vec2_llama import Wav2Vec2LlamaBeamSearchConfig

    default_cfg = pipe.beam_search_generator.config
    configs = {
        "beam_nbest1": Wav2Vec2LlamaBeamSearchConfig(nbest=1, length_norm=False, compression_window=100, compression_threshold=4.0),
        "beam_nbest2": Wav2Vec2LlamaBeamSearchConfig(nbest=2, length_norm=False, compression_window=100, compression_threshold=4.0),
        "beam_nbest2_length_norm": Wav2Vec2LlamaBeamSearchConfig(nbest=2, length_norm=True, compression_window=100, compression_threshold=4.0),
        "beam_nbest4_length_norm": Wav2Vec2LlamaBeamSearchConfig(nbest=4, length_norm=True, compression_window=100, compression_threshold=4.0),
    }
    paths = screen["audio"].tolist()
    refs = screen["reference"].tolist()
    for name, cfg in configs.items():
        pipe.beam_search_generator.config = cfg
        hyps = pipe.transcribe(paths, lang=["lug_Latn"] * len(paths), batch_size=1)
        scores = split_metrics(screen, list(hyps))
        result["variants"][name] = {
            "config": {
                "nbest": cfg.nbest,
                "length_norm": cfg.length_norm,
                "compression_window": cfg.compression_window,
                "compression_threshold": cfg.compression_threshold,
            },
            "metrics": scores,
            "raw_hypotheses": list(hyps),
        }
        PARTIAL.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"variant": name, "metrics": scores}, ensure_ascii=False), flush=True)

    # Language conditioning control: the only supported production tag is
    # lug_Latn.  None is tested as a negative control, not as a candidate.
    pipe.beam_search_generator.config = default_cfg
    no_lang = pipe.transcribe(paths, lang=[None] * len(paths), batch_size=1)
    result["variants"]["no_language_tag_control"] = {
        "config": {"lang": None, "beam": "default"},
        "metrics": split_metrics(screen, list(no_lang)),
        "raw_hypotheses": list(no_lang),
    }
    PARTIAL.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"variant": "no_language_tag_control", "metrics": result["variants"]["no_language_tag_control"]["metrics"]}, ensure_ascii=False), flush=True)

    # Long-audio sidecar: split only the five >40-second locked-gate rows.
    # Compare two fixed chunking geometries; do not tune per utterance.
    for scheme, chunk_seconds, overlap_seconds in (
        ("chunk30_overlap1", 30.0, 1.0),
        ("chunk20_overlap1", 20.0, 1.0),
    ):
        per_row: list[dict] = []
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
        for row_idx, start, stop in chunk_map:
            row_hypotheses = list(chunk_hyps[start:stop])
            merged.append(merge_chunks(row_hypotheses))
            per_row.append(
                {
                    "ID": long_rows.iloc[row_idx]["ID"],
                    "duration_sec": float(long_rows.iloc[row_idx]["duration_sec"]),
                    "chunk_paths": all_chunks[start:stop],
                    "chunk_hypotheses": row_hypotheses,
                    "merged": merged[-1],
                }
            )
        baseline = long_rows["baseline"].tolist()
        score_candidate = metric(long_rows["reference"].tolist(), merged)
        score_baseline = metric(long_rows["reference"].tolist(), baseline)
        result["long_audio"][scheme] = {
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "n_chunks": len(all_chunks),
            "candidate_metrics": score_candidate,
            "incumbent_metrics": score_baseline,
            "delta_vs_incumbent": score_candidate["zindi"] - score_baseline["zindi"],
            "rows": per_row,
        }
        PARTIAL.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"long_scheme": scheme, "candidate": score_candidate, "incumbent": score_baseline}, ensure_ascii=False), flush=True)

    result["model"] = {
        "repo": "mlai-dante/waxal-omniASR-LLM-1B-v2",
        "checkpoint": "step_1000/model",
        "device": "cpu",
        "valid_luganda_tag": "lug_Latn",
        "checkpoint_sha256_not_recomputed": True,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    PARTIAL.unlink(missing_ok=True)
    print(json.dumps({"artifact": str(OUT), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, indent=2), flush=True)


def wave_duration(path: str) -> float:
    with wave.open(path) as handle:
        return handle.getnframes() / handle.getframerate()


if __name__ == "__main__":
    main()
