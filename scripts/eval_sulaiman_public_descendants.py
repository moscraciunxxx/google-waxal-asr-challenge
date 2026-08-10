#!/usr/bin/env python3
"""Matched evaluation of three public Sulaiman WAXAL descendants.

Safety properties:
* validation IDs are the immutable seed=42/n=80 IDs used by the existing
  route A/B scripts;
* every candidate is compared on the same IDs with the actual route incumbent;
* Phase-2 CSVs are read with an explicit four-column projection and never use
  a target/transcription column;
* Phase-2 audio is decoded only after a candidate clears the strict pass gate;
* models are loaded and released one at a time for bounded memory use.

This script creates prediction caches only.  It never edits or builds a
competition submission.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Keep datasets' derived Arrow files and locks inside this task's write scope.
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "goal_2026_08_08"
        / "sulaiman_public_descendants"
        / "hf_datasets_cache"
    ),
)

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import (
    AutoProcessor,
    Wav2Vec2BertForCTC,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mms_adapter_ft import fix_mms_tokenizer
from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.text_norm import normalize_text

OUT = ROOT / "outputs" / "goal_2026_08_08" / "sulaiman_public_descendants"
ROUTES = ROOT / "outputs" / "beat075" / "public_visible_index.csv"
SAMPLE_SEED = 42
SAMPLE_N = 80
PASS_MARGIN = 0.01
BOOTSTRAP_DRAWS = 2000

# Heads pin the historical seed-42 protocol and catch dataset-order drift.
EXPECTED_HEAD = {
    "lin": [
        "lin_14239", "lin_94913", "lin_31328", "lin_17162", "lin_11374",
        "lin_57062", "lin_64649", "lin_8297", "lin_39747", "lin_80492",
    ],
    "sna": [
        "sna_77112", "sna_56117", "sna_79706", "sna_28169", "sna_13231",
        "sna_91118", "sna_38268", "sna_66743", "sna_88683", "sna_33808",
    ],
}


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    model_id: str
    checkpoint: Path
    lang: str
    kind: str


SPECS = {
    "whisper-small-lingala-cased-2": ModelSpec(
        tag="whisper-small-lingala-cased-2",
        model_id="sulaimank/whisper-small-lingala-cased-2",
        checkpoint=ROOT / "checkpoints" / "sulaimank-whisper-small-lingala-cased-2",
        lang="lin",
        kind="whisper",
    ),
    "w2vbert-lingala-sd3": ModelSpec(
        tag="w2vbert-lingala-sd3",
        model_id="sulaimank/w2vbert-lingala-sd3",
        checkpoint=ROOT / "checkpoints" / "sulaimank-w2vbert-lingala-sd3",
        lang="lin",
        kind="w2vbert",
    ),
    "w2vbert-shona-sd2": ModelSpec(
        tag="w2vbert-shona-sd2",
        model_id="sulaimank/w2vbert-shona-sd2",
        checkpoint=ROOT / "checkpoints" / "sulaimank-w2vbert-shona-sd2",
        lang="sna",
        kind="w2vbert",
    ),
}


def pick_device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sha_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def normalize_audio(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=-1)
    peak = float(np.max(np.abs(array)) + 1e-9)
    return array / peak


def validation_sample(lang: str) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    dataset = load_hf_asr_split(lang, "validation")
    indices = list(range(len(dataset)))
    random.Random(SAMPLE_SEED).shuffle(indices)
    indices = indices[:SAMPLE_N]
    examples: list[dict[str, Any]] = []
    rows = []
    for pos, index in enumerate(indices):
        ex = dict(dataset[index])
        uid = str(ex.get("id") or ex.get("ID"))
        ref = normalize_text(ex.get("transcription") or "") or "."
        audio = ex["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate") or TARGET_SR)
        if sr != TARGET_SR:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        ex["_array"] = normalize_audio(arr)
        ex["_id"] = uid
        ex["_reference"] = ref
        examples.append(ex)
        rows.append(
            {
                "ID": uid,
                "language": lang,
                "split": "validation",
                "sample_position": pos,
                "fold": "tune" if pos % 2 == 0 else "holdout",
                "reference": ref,
            }
        )
    ids = [row["ID"] for row in rows]
    if ids[:10] != EXPECTED_HEAD[lang]:
        raise RuntimeError(
            f"{lang} validation order drift: {ids[:10]} != {EXPECTED_HEAD[lang]}"
        )
    if len(ids) != SAMPLE_N or len(set(ids)) != SAMPLE_N:
        raise RuntimeError(f"{lang}: invalid immutable sample cardinality")
    return examples, pd.DataFrame(rows)


def phase2_route(lang: str) -> pd.DataFrame:
    # Deliberately exclude prediction/floor/Target/transcription columns.
    frame = pd.read_csv(
        ROUTES,
        usecols=["ID", "decode_lang", "split", "audio"],
        dtype={"ID": str, "decode_lang": str, "split": str, "audio": str},
    )
    frame = frame[(frame["split"] == "new") & (frame["decode_lang"] == lang)].copy()
    frame = frame.sort_values("ID", kind="stable").reset_index(drop=True)
    expected = 444 if lang == "lin" else 445
    if len(frame) != expected or frame.ID.nunique() != expected:
        raise RuntimeError(f"{lang}: expected {expected} public-visible Phase2 IDs, got {len(frame)}")
    missing = [p for p in frame.audio.map(Path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{lang}: missing Phase2 audio, e.g. {missing[:3]}")
    return frame


def checkpoint_revision(path: Path) -> str | None:
    tree_dir = path / ".cache" / "huggingface" / "trees"
    trees = sorted(tree_dir.glob("*.json")) if tree_dir.is_dir() else []
    return trees[-1].stem if trees else None


def checkpoint_audit(spec: ModelSpec) -> dict[str, Any]:
    if not (spec.checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint incomplete: {spec.checkpoint}")
    cfg = json.loads((spec.checkpoint / "config.json").read_text())
    readme = (spec.checkpoint / "README.md").read_text(errors="replace")
    weight = spec.checkpoint / "model.safetensors"
    metadata = spec.checkpoint / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
    lfs_sha = None
    if metadata.is_file():
        lines = metadata.read_text().splitlines()
        lfs_sha = lines[1] if len(lines) > 1 else None
    return {
        "model_id": spec.model_id,
        "checkpoint": str(spec.checkpoint),
        "revision": checkpoint_revision(spec.checkpoint),
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "vocab_size": cfg.get("vocab_size"),
        "pad_token_id": cfg.get("pad_token_id"),
        "weight_bytes": weight.stat().st_size if weight.is_file() else None,
        "weight_lfs_sha256": lfs_sha,
        "training_data_disclosed": "unknown dataset" not in readme.lower()
        and "more information needed" not in readme.lower(),
        "card_provenance_warning": (
            "The public card does not identify training/evaluation data; matched metrics "
            "are measured here, but validation-set exposure cannot be independently excluded."
            if "unknown dataset" in readme.lower() or "more information needed" in readme.lower()
            else None
        ),
    }


def load_candidate(spec: ModelSpec, device: torch.device):
    if spec.kind == "whisper":
        processor = WhisperProcessor.from_pretrained(str(spec.checkpoint), local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            str(spec.checkpoint), local_files_only=True, low_cpu_mem_usage=True
        ).to(device).eval()
        # Pin the checkpoint's documented Lingala generation protocol.
        model.generation_config.language = "ln"
        model.generation_config.task = "transcribe"
        model.generation_config.return_timestamps = False
        return model, processor
    processor = AutoProcessor.from_pretrained(str(spec.checkpoint), local_files_only=True)
    model = Wav2Vec2BertForCTC.from_pretrained(
        str(spec.checkpoint), local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()
    return model, processor


def load_incumbent(lang: str, device: torch.device):
    if lang == "lin":
        model_id = "facebook/mms-1b-all"
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(
            model_id, local_files_only=True, low_cpu_mem_usage=True
        )
        fix_mms_tokenizer(processor, "lin")
        model.load_adapter("lin", local_files_only=True)
        return "mms1b_lin_production", model.to(device).eval(), processor
    model_id = "badrex/w2v-bert-2.0-shona-asr"
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = Wav2Vec2BertForCTC.from_pretrained(
        model_id, local_files_only=True, low_cpu_mem_usage=True
    )
    return "badrex_sna_production", model.to(device).eval(), processor


@torch.inference_mode()
def decode_one(model, processor, kind: str, array: np.ndarray, device: torch.device) -> str:
    if kind == "whisper":
        features = processor(
            array, sampling_rate=TARGET_SR, do_normalize=True, return_tensors="pt"
        ).input_features.to(device)
        ids = model.generate(
            features,
            language="ln",
            task="transcribe",
            return_timestamps=False,
            do_sample=False,
            num_beams=1,
        )
        text = processor.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return normalize_text(text) or "."
    inputs = processor(array, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    kwargs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
    logits = model(**kwargs).logits
    ids = torch.argmax(logits, dim=-1)[0].detach().cpu()
    return normalize_text(processor.decode(ids)) or "."


@torch.inference_mode()
def decode_many(
    model, processor, kind: str, arrays: list[np.ndarray], device: torch.device
) -> list[str]:
    """Deterministic batched equivalent of ``decode_one``."""
    if kind == "whisper":
        features = processor(
            arrays,
            sampling_rate=TARGET_SR,
            do_normalize=True,
            padding=True,
            return_tensors="pt",
        ).input_features.to(device)
        ids = model.generate(
            features,
            language="ln",
            task="transcribe",
            return_timestamps=False,
            do_sample=False,
            num_beams=1,
        )
        texts = processor.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [normalize_text(text) or "." for text in texts]
    inputs = processor(
        arrays, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    )
    kwargs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
    logits = model(**kwargs).logits
    ids = torch.argmax(logits, dim=-1).detach().cpu()
    return [normalize_text(text) or "." for text in processor.batch_decode(ids)]


def release(*objects: Any, device: torch.device) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def decode_validation(
    spec: ModelSpec,
    examples: list[dict[str, Any]],
    manifest: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    detail_path = OUT / f"validation_{spec.tag}.csv"
    if detail_path.is_file():
        detail = pd.read_csv(detail_path, dtype={"ID": str})
        if detail.ID.tolist() != manifest.ID.tolist():
            raise RuntimeError(f"stale candidate cache IDs: {detail_path}")
        return detail
    incumbent_path = OUT / f"validation_incumbent_{spec.lang}.csv"
    if incumbent_path.is_file():
        incumbent = pd.read_csv(incumbent_path, dtype={"ID": str})
        if incumbent.ID.tolist() != manifest.ID.tolist():
            raise RuntimeError(f"stale incumbent cache IDs: {incumbent_path}")
    else:
        tag, model, processor = load_incumbent(spec.lang, device)
        started = time.time()
        rows = []
        kind = "w2vbert" if spec.lang == "sna" else "wav2vec2"
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            hyps = decode_many(model, processor, kind, [ex["_array"] for ex in chunk], device)
            rows.extend({"ID": ex["_id"], "incumbent": hyp} for ex, hyp in zip(chunk, hyps))
            pos = start + len(chunk)
            print(f"{tag}: {pos}/{len(examples)} ({time.time()-started:.1f}s)", flush=True)
        incumbent = pd.DataFrame(rows)
        incumbent.to_csv(incumbent_path, index=False)
        release(model, processor, device=device)

    model, processor = load_candidate(spec, device)
    started = time.time()
    rows = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        hyps = decode_many(model, processor, spec.kind, [ex["_array"] for ex in chunk], device)
        rows.extend({"ID": ex["_id"], "candidate": hyp} for ex, hyp in zip(chunk, hyps))
        pos = start + len(chunk)
        print(f"{spec.tag}: {pos}/{len(examples)} ({time.time()-started:.1f}s)", flush=True)
    release(model, processor, device=device)
    candidate = pd.DataFrame(rows)
    detail = manifest.merge(incumbent, on="ID", validate="one_to_one").merge(
        candidate, on="ID", validate="one_to_one"
    )
    detail.to_csv(detail_path, index=False)
    return detail


def metric(hyp: pd.Series, ref: pd.Series) -> dict[str, float]:
    score = score_pairs(ref.astype(str).tolist(), hyp.astype(str).tolist())
    return {
        "n": int(len(ref)),
        "wer": float(score["wer"]),
        "cer": float(score["cer"]),
        "zindi": float(1.0 - score["score"]),
    }


def paired_bootstrap(detail: pd.DataFrame) -> dict[str, float]:
    rng = np.random.default_rng(20260808)
    n = len(detail)
    deltas = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        take = rng.integers(0, n, size=n)
        part = detail.iloc[take]
        old = metric(part.incumbent, part.reference)["zindi"]
        new = metric(part.candidate, part.reference)["zindi"]
        deltas[draw] = new - old
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": 20260808,
        "delta_mean": float(deltas.mean()),
        "delta_p05": float(np.quantile(deltas, 0.05)),
        "delta_p50": float(np.quantile(deltas, 0.50)),
        "delta_p95": float(np.quantile(deltas, 0.95)),
        "probability_delta_positive": float(np.mean(deltas > 0)),
    }


def evaluate(detail: pd.DataFrame) -> dict[str, Any]:
    incumbent = metric(detail.incumbent, detail.reference)
    candidate = metric(detail.candidate, detail.reference)
    folds = {}
    for fold in ("tune", "holdout"):
        part = detail[detail.fold == fold]
        old = metric(part.incumbent, part.reference)
        new = metric(part.candidate, part.reference)
        folds[fold] = {
            "incumbent": old,
            "candidate": new,
            "delta_zindi": new["zindi"] - old["zindi"],
        }
    bootstrap = paired_bootstrap(detail)
    delta = candidate["zindi"] - incumbent["zindi"]
    pass_reasons = {
        "overall_delta_at_least_0p01": delta >= PASS_MARGIN,
        "wer_strictly_better": candidate["wer"] < incumbent["wer"],
        "tune_delta_positive": folds["tune"]["delta_zindi"] > 0,
        "holdout_delta_positive": folds["holdout"]["delta_zindi"] > 0,
        "bootstrap_p05_positive": bootstrap["delta_p05"] > 0,
    }
    return {
        "incumbent": incumbent,
        "candidate": candidate,
        "delta_zindi": delta,
        "folds": folds,
        "paired_bootstrap": bootstrap,
        "pass_rule": (
            "delta_zindi>=0.01, candidate WER<incumbent WER, positive tune and "
            "holdout deltas, and paired-bootstrap 5th-percentile delta>0"
        ),
        "pass_checks": pass_reasons,
        "strong_pass": all(pass_reasons.values()),
    }


def update_report(spec: ModelSpec, audit: dict[str, Any], result: dict[str, Any], manifest: pd.DataFrame) -> None:
    path = OUT / "report.json"
    report = json.loads(path.read_text()) if path.is_file() else {
        "protocol": {
            "dataset": "google/WaxalNLP validation",
            "sample_seed": SAMPLE_SEED,
            "n_per_language": SAMPLE_N,
            "normalization": "src.text_norm.normalize_text",
            "test_labels_read": False,
            "submission_built": False,
        },
        "models": {},
    }
    report["models"][spec.tag] = {
        "audit": audit,
        "validation_ids_sha256": sha_lines(manifest.ID.tolist()),
        "metrics": result,
        "validation_detail": str(OUT / f"validation_{spec.tag}.csv"),
    }
    path.write_text(json.dumps(report, indent=2) + "\n")


def decode_phase2_cache(
    spec: ModelSpec,
    device: torch.device,
    validation_ids: set[str],
    batch_size: int,
) -> Path:
    route = phase2_route(spec.lang)
    overlap = sorted(validation_ids & set(route.ID))
    if overlap:
        raise RuntimeError(f"validation/Phase2 ID leakage: {overlap[:5]}")
    cache = OUT / f"phase2_cache_{spec.tag}.csv"
    done: dict[str, dict[str, str]] = {}
    if cache.is_file():
        previous = pd.read_csv(cache, dtype={"ID": str})
        done = {str(row.ID): {"ID": str(row.ID), "Target": str(row.Target)} for row in previous.itertuples()}
    model, processor = load_candidate(spec, device)
    started = time.time()
    try:
        todo = [row for row in route.itertuples(index=False) if str(row.ID) not in done]
        for start in range(0, len(todo), batch_size):
            chunk = todo[start : start + batch_size]
            arrays = []
            for row in chunk:
                array, sr = sf.read(str(row.audio), dtype="float32", always_2d=False)
                if int(sr) != TARGET_SR:
                    import librosa

                    array = librosa.resample(np.asarray(array), orig_sr=int(sr), target_sr=TARGET_SR)
                arrays.append(normalize_audio(array))
            hyps = decode_many(model, processor, spec.kind, arrays, device)
            for row, hyp in zip(chunk, hyps):
                uid = str(row.ID)
                done[uid] = {"ID": uid, "Target": hyp}
            ordered = [done[uid] for uid in route.ID if uid in done]
            pd.DataFrame(ordered, columns=["ID", "Target"]).to_csv(cache, index=False)
            print(f"{spec.tag} Phase2: {len(done)}/{len(route)} ({time.time()-started:.1f}s)", flush=True)
    finally:
        release(model, processor, device=device)
    final = pd.read_csv(cache, dtype={"ID": str})
    if final.ID.tolist() != route.ID.tolist() or final.Target.isna().any():
        raise RuntimeError(f"incomplete or misordered cache: {cache}")
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(SPECS), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-if-pass", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = SPECS[args.model]
    device = pick_device(args.device)
    print(f"model={spec.model_id} lang={spec.lang} device={device}", flush=True)

    audit = checkpoint_audit(spec)
    examples, manifest = validation_sample(spec.lang)
    manifest_path = OUT / f"validation_manifest_{spec.lang}.csv"
    if manifest_path.is_file():
        old = pd.read_csv(manifest_path, dtype={"ID": str})
        if old.ID.tolist() != manifest.ID.tolist() or old.reference.tolist() != manifest.reference.tolist():
            raise RuntimeError(f"immutable manifest changed: {manifest_path}")
    else:
        manifest.to_csv(manifest_path, index=False)

    # ID-only leakage audit is always performed before any candidate decode.
    phase2_ids = set(phase2_route(spec.lang).ID)
    overlap = sorted(set(manifest.ID) & phase2_ids)
    if overlap:
        raise RuntimeError(f"validation/Phase2 ID overlap: {overlap[:5]}")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    detail = decode_validation(spec, examples, manifest, device, args.batch_size)
    result = evaluate(detail)
    update_report(spec, audit, result, manifest)
    print(json.dumps({"model": spec.tag, "metrics": result, "audit": audit}, indent=2), flush=True)

    if args.cache_if_pass and result["strong_pass"]:
        cache = decode_phase2_cache(spec, device, set(manifest.ID), args.batch_size)
        report_path = OUT / "report.json"
        report = json.loads(report_path.read_text())
        report["models"][spec.tag]["phase2_cache"] = str(cache)
        report["models"][spec.tag]["phase2_cache_rows"] = len(pd.read_csv(cache))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"strong pass; wrote test cache {cache}", flush=True)
    elif args.cache_if_pass:
        print("no strong pass; Phase2 audio was not decoded", flush=True)


if __name__ == "__main__":
    main()
