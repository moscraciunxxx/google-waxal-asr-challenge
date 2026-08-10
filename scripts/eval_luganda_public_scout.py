#!/usr/bin/env python3
"""Evaluate one public Luganda/WAXAL checkpoint against the production model.

The protocol is deliberately small and immutable: 20 Luganda validation rows
selected without replacement by NumPy RNG seed 42, with the first 10 sampled
rows designated as screening and the final 10 as holdout before inference.
No challenge test labels or test audio are loaded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mms_adapter_ft import fix_mms_tokenizer, pick_device
from scripts.phase3_text_norm_ablations import feat_D_join_lug_splits
from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.text_norm import normalize_text

OUT = ROOT / "outputs" / "goal_2026_08_08" / "luganda_public_scout"
PUBLIC_MODEL = ROOT / "checkpoints" / "luganda_public_scout" / "w2vbert-waxal-r5"
WHISPER_MODEL = (
    ROOT / "checkpoints" / "luganda_public_scout" / "whisper-small-luganda-waxal"
)
INCUMBENT_MODEL = ROOT / "checkpoints" / "mms-lug-ft-v3"
SEED = 42
N = 20
BOOTSTRAP_DRAWS = 2000


def metric(refs: list[str], hyps: list[str]) -> dict[str, float | int]:
    result = score_pairs(refs, hyps)
    return {
        "n": len(refs),
        "wer": float(result["wer"]),
        "cer": float(result["cer"]),
        "error": float(result["score"]),
        "zindi": float(1.0 - result["score"]),
    }


def prepare_audio(example: dict) -> np.ndarray:
    audio = example["audio"]
    array = np.asarray(audio["array"], dtype=np.float32)
    sampling_rate = int(audio.get("sampling_rate") or TARGET_SR)
    if sampling_rate != TARGET_SR:
        import librosa

        array = librosa.resample(
            array, orig_sr=sampling_rate, target_sr=TARGET_SR
        ).astype(np.float32)
    peak = float(np.max(np.abs(array)) + 1e-9)
    return array / peak


def load_sample() -> list[dict]:
    dataset = load_hf_asr_split("lug", "validation")
    if len(dataset) < N:
        raise RuntimeError(f"Luganda validation has only {len(dataset)} rows")
    rng = np.random.default_rng(SEED)
    positions = rng.choice(len(dataset), size=N, replace=False).tolist()
    rows: list[dict] = []
    for sample_order, position in enumerate(positions):
        example = dataset[int(position)]
        uid = str(example.get("id") or example.get("ID") or "")
        reference = normalize_text(str(example.get("transcription") or ""))
        speaker = str(example.get("speaker_id") or "")
        if not uid or not reference or not speaker:
            raise RuntimeError(f"invalid validation row at position {position}")
        rows.append(
            {
                "sample_order": sample_order,
                "dataset_position": int(position),
                "split": "screen" if sample_order < N // 2 else "holdout",
                "ID": uid,
                "speaker_id": speaker,
                "reference": reference,
                "audio": prepare_audio(example),
            }
        )
    if len({row["ID"] for row in rows}) != N:
        raise RuntimeError("seeded sample has duplicate IDs")
    return rows


def load_splitjoin_counts() -> tuple[dict[str, int], dict[str, int]]:
    payload = json.loads((ROOT / "data" / "lms" / "lug_counts.json").read_text())
    unigrams = {
        str(word): int(count)
        for word, count in payload["uni"].items()
        if not str(word).startswith("<")
    }
    bigrams = {str(pair): int(count) for pair, count in payload["bi"].items()}
    return unigrams, bigrams


def apply_splitjoin(
    text: str, counts: tuple[dict[str, int], dict[str, int]]
) -> str:
    return normalize_text(feat_D_join_lug_splits(text, *counts)) or "."


@torch.inference_mode()
def decode_model(
    path: Path,
    rows: list[dict],
    device: torch.device,
    *,
    incumbent: bool,
) -> list[str]:
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    if incumbent:
        fix_mms_tokenizer(processor, "lug")
        model = Wav2Vec2ForCTC.from_pretrained(
            str(path), local_files_only=True, low_cpu_mem_usage=True
        )
    else:
        tokenizer = processor.tokenizer
        delimiter = int(tokenizer.word_delimiter_token_id)
        pad = int(tokenizer.pad_token_id)
        if str(tokenizer.convert_ids_to_tokens(delimiter)) != "|":
            raise RuntimeError(f"candidate delimiter ID {delimiter} is not '|'")
        if pad != int(json.loads((path / "config.json").read_text())["pad_token_id"]):
            raise RuntimeError("candidate tokenizer/model blank mismatch")
        model = AutoModelForCTC.from_pretrained(
            str(path), local_files_only=True, low_cpu_mem_usage=True
        )
    model.to(device).eval()
    hypotheses: list[str] = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        inputs = processor(
            row["audio"],
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        kwargs = {
            key: value.to(device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        logits = model(**kwargs).logits[0]
        token_ids = torch.argmax(logits, dim=-1)
        # CTC alphabets can contain ordinary graphemes marked as added/special.
        # Retaining them is required for valid CER.
        hypothesis = normalize_text(processor.decode(token_ids)) or "."
        hypotheses.append(hypothesis)
        if index % 5 == 0:
            print(
                f"{path.name}: {index}/{len(rows)} "
                f"({(time.time() - started) / index:.2f}s/utt)",
                flush=True,
            )
    model.to("cpu")
    del model, processor
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return hypotheses


@torch.inference_mode()
def decode_whisper(
    path: Path, rows: list[dict], device: torch.device
) -> list[str]:
    processor = WhisperProcessor.from_pretrained(str(path), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(path), local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    hypotheses: list[str] = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        features = processor(
            row["audio"], sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(device)
        generated = model.generate(
            features,
            do_sample=False,
            num_beams=1,
            max_new_tokens=256,
        )
        hypothesis = normalize_text(
            processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        ) or "."
        hypotheses.append(hypothesis)
        if index % 5 == 0:
            print(
                f"{path.name}: {index}/{len(rows)} "
                f"({(time.time() - started) / index:.2f}s/utt)",
                flush=True,
            )
    model.to("cpu")
    del model, processor
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return hypotheses


def metric_for_indices(
    frame: pd.DataFrame, candidate: str, indices: np.ndarray
) -> dict[str, float | int]:
    part = frame.iloc[indices]
    return metric(part.reference.tolist(), part[candidate].tolist())


def paired_bootstrap(
    frame: pd.DataFrame, candidate: str, baseline: str, *, by_speaker: bool
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(SEED + (1000 if by_speaker else 0))
    deltas = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    if by_speaker:
        units = np.array(sorted(frame.speaker_id.unique()))
        groups = {
            unit: np.flatnonzero(frame.speaker_id.to_numpy() == unit) for unit in units
        }
        for draw in range(BOOTSTRAP_DRAWS):
            sampled = rng.choice(units, size=len(units), replace=True)
            indices = np.concatenate([groups[unit] for unit in sampled])
            deltas[draw] = (
                metric_for_indices(frame, candidate, indices)["zindi"]
                - metric_for_indices(frame, baseline, indices)["zindi"]
            )
        method = "paired speaker-block bootstrap"
    else:
        for draw in range(BOOTSTRAP_DRAWS):
            indices = rng.integers(0, len(frame), size=len(frame))
            deltas[draw] = (
                metric_for_indices(frame, candidate, indices)["zindi"]
                - metric_for_indices(frame, baseline, indices)["zindi"]
            )
        method = "paired row bootstrap"
    return {
        "method": method,
        "draws": BOOTSTRAP_DRAWS,
        "delta_mean": float(deltas.mean()),
        "delta_p025": float(np.quantile(deltas, 0.025)),
        "delta_p05": float(np.quantile(deltas, 0.05)),
        "delta_p50": float(np.quantile(deltas, 0.50)),
        "delta_p95": float(np.quantile(deltas, 0.95)),
        "delta_p975": float(np.quantile(deltas, 0.975)),
        "p_delta_gt_0": float(np.mean(deltas > 0)),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (PUBLIC_MODEL, WHISPER_MODEL, INCUMBENT_MODEL):
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError(path / "model.safetensors")

    rows = load_sample()
    device = pick_device(args.device)
    print(f"device={device} seed={SEED} n={N}", flush=True)
    incumbent_raw = decode_model(INCUMBENT_MODEL, rows, device, incumbent=True)
    candidate_raw = decode_model(PUBLIC_MODEL, rows, device, incumbent=False)
    whisper_raw = decode_whisper(WHISPER_MODEL, rows, device)
    counts = load_splitjoin_counts()

    frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "audio"} for row in rows]
    )
    frame["incumbent_raw"] = incumbent_raw
    frame["incumbent_splitjoin"] = [apply_splitjoin(x, counts) for x in incumbent_raw]
    frame["candidate_raw"] = candidate_raw
    frame["candidate_splitjoin"] = [apply_splitjoin(x, counts) for x in candidate_raw]
    frame["whisper_raw"] = whisper_raw
    frame["whisper_splitjoin"] = [apply_splitjoin(x, counts) for x in whisper_raw]
    frame.to_csv(OUT / "matched_seed42_n20.csv", index=False)

    systems = [
        "incumbent_raw",
        "incumbent_splitjoin",
        "candidate_raw",
        "candidate_splitjoin",
        "whisper_raw",
        "whisper_splitjoin",
    ]
    metrics: dict[str, dict] = {}
    for system in systems:
        metrics[system] = {
            "all": metric(frame.reference.tolist(), frame[system].tolist()),
            "screen": metric(
                frame.loc[frame.split == "screen", "reference"].tolist(),
                frame.loc[frame.split == "screen", system].tolist(),
            ),
            "holdout": metric(
                frame.loc[frame.split == "holdout", "reference"].tolist(),
                frame.loc[frame.split == "holdout", system].tolist(),
            ),
        }

    candidate = max(
        (
            "candidate_raw",
            "candidate_splitjoin",
            "whisper_raw",
            "whisper_splitjoin",
        ),
        key=lambda name: metrics[name]["screen"]["zindi"],
    )
    baseline = "incumbent_splitjoin"
    report = {
        "protocol": {
            "dataset": "google/WaxalNLP lug validation",
            "seed": SEED,
            "sample_size": N,
            "selection": "numpy.default_rng(seed).choice(dataset length, n, replace=False)",
            "split": "first 10 sampled rows screen; final 10 holdout, fixed before inference",
            "exact_ids": frame.ID.tolist(),
            "exact_id_sha256": hashlib.sha256(
                "\n".join(frame.ID.tolist()).encode("utf-8")
            ).hexdigest(),
            "unique_speakers": int(frame.speaker_id.nunique()),
            "screen_holdout_speaker_overlap": sorted(
                set(frame.loc[frame.split == "screen", "speaker_id"])
                & set(frame.loc[frame.split == "holdout", "speaker_id"])
            ),
            "test_audio_loaded": False,
            "test_labels_used": False,
        },
        "candidate": {
            "w2vbert_r5": {
                "repo": "sulaimank/w2vbert-waxal-r5",
                "revision": "68c82a930a6183d3235d32c0c2ba0ac1e2bdd36f",
                "architecture": "Wav2Vec2BertForCTC",
                "model_bytes": (PUBLIC_MODEL / "model.safetensors").stat().st_size,
                "model_sha256": file_sha256(PUBLIC_MODEL / "model.safetensors"),
                "license": None,
                "provenance": "undocumented model card; latest public commit says training step 9000",
                "caveat": "No declared language, dataset, evaluation split, metrics, or license.",
            },
            "whisper_small": {
                "repo": "ElizabethMwangi/whisper-small-luganda-waxal",
                "revision": "7b85793afac57826122db4cb0db3781ac9b72ccb",
                "architecture": "WhisperForConditionalGeneration (small)",
                "model_bytes": (WHISPER_MODEL / "model.safetensors").stat().st_size,
                "model_sha256": file_sha256(WHISPER_MODEL / "model.safetensors"),
                "license": None,
                "provenance": "repository name identifies Luganda/WAXAL; no model card",
                "caveat": "No declared dataset split, metrics, training recipe, or license.",
            },
        },
        "requested_stage2": {
            "repo": "sulaimank/w2vbert-luganda-waxal-stage2",
            "revision": "207754aeabcf400657e9aa7eb5776f8aee6e6fb6",
            "access": "manual gated; existing authenticated CLI account denied",
            "license_metadata": "mit",
            "base_model": "sulaimank/w2vbert-luganda-waxal",
            "evaluated": False,
            "new_access_requested": False,
        },
        "metrics": metrics,
        "screen_selected_candidate_variant": candidate,
        "delta_vs_incumbent": {
            split: float(metrics[candidate][split]["zindi"] - metrics[baseline][split]["zindi"])
            for split in ("all", "screen", "holdout")
        },
        "bootstrap_all": {
            "rows": paired_bootstrap(frame, candidate, baseline, by_speaker=False),
            "speakers": paired_bootstrap(frame, candidate, baseline, by_speaker=True),
        },
        "pass_rule": {
            "required": "candidate all-sample Zindi >0.91, holdout delta >0, WER and CER both no worse on all rows, speaker-bootstrap p05 delta >0",
            "passed": bool(
                metrics[candidate]["all"]["zindi"] > 0.91
                and metrics[candidate]["holdout"]["zindi"] > metrics[baseline]["holdout"]["zindi"]
                and metrics[candidate]["all"]["wer"] <= metrics[baseline]["all"]["wer"]
                and metrics[candidate]["all"]["cer"] <= metrics[baseline]["all"]["cer"]
            ),
        },
    }
    report["pass_rule"]["passed"] = bool(
        report["pass_rule"]["passed"]
        and report["bootstrap_all"]["speakers"]["delta_p05"] > 0
    )
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
