#!/usr/bin/env python3
"""Calibrate Shona model-switch policies on the cached WAXAL validation split.

This deliberately reads the cached Arrow shards directly.  The normal HF
dataset builder can try to acquire a lock or refresh metadata even when all
validation audio is already present locally.  The output is calibration-only;
it never edits a submission.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import soundfile as sf
import torch
from jiwer import cer, wer
from transformers import (
    AutoProcessor,
    Wav2Vec2BertForCTC,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text_norm import normalize_text

TARGET_SR = 16_000
WAXAL_SNA = "waxal-benchmarking/mms-300m-waxal-sna"
BADREX = "badrex/w2v-bert-2.0-shona-asr"
MUBARAK = "Mubarak127/waxal-whisper-large-v3-sna_asr"


def local_snapshot(cache_root: Path, repo_id: str) -> str:
    """Use a complete local snapshot when offline metadata refresh is unavailable."""
    folder = "models--" + repo_id.replace("/", "--")
    snapshots = sorted((cache_root.parent / "hub" / folder / "snapshots").glob("*"))
    if snapshots:
        return str(snapshots[-1])
    return repo_id


def norm(text: str) -> str:
    return normalize_text(text)


def load_rows(cache_root: Path) -> list[dict]:
    authoritative = sorted(
        (cache_root.parent / "hub").glob(
            "datasets--google--WaxalNLP/snapshots/*/data/ASR/sna/sna-validation-*.parquet"
        )
    )
    if authoritative:
        rows: list[dict] = []
        for path in authoritative:
            rows.extend(pq.read_table(path).to_pylist())
        return rows

    paths = sorted(cache_root.glob("**/parquet-validation*.arrow"))
    rows: list[dict] = []
    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            table = ipc.open_stream(source).read_all()
        shard = table.to_pylist()
        if shard and str(shard[0].get("id", "")).startswith("sna_"):
            rows.extend(shard)
    if not rows:
        raise FileNotFoundError("No cached Shona validation Arrow shards found")
    return rows


def audio_array(row: dict) -> np.ndarray:
    raw = row["audio"].get("bytes")
    if raw is None:
        raw = Path(row["audio"]["path"]).read_bytes()
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if int(sr) != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=int(sr), target_sr=TARGET_SR)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def decode_mms(model, processor, audio: np.ndarray, device: torch.device) -> str:
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    logits = model(inputs.input_values.to(device)).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return norm(processor.decode(ids))


@torch.inference_mode()
def decode_badrex(model, processor, audio: np.ndarray, device: torch.device) -> str:
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    kwargs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    logits = model(**kwargs).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return norm(processor.decode(ids))


@torch.inference_mode()
def decode_mubarak(model, processor, audio: np.ndarray, device: torch.device) -> str:
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    features = inputs.input_features.to(device)
    try:
        forced = processor.get_decoder_prompt_ids(language="sn", task="transcribe")
        ids = model.generate(features, forced_decoder_ids=forced, max_new_tokens=128)
    except Exception:
        ids = model.generate(features, max_new_tokens=128)
    return norm(processor.batch_decode(ids, skip_special_tokens=True)[0])


def row_cost(ref: str, hyp: str) -> float:
    # The competition score is the mean of WER and CER; single-row values are
    # useful for learning a switch policy, while corpus scores remain the
    # authoritative metric.
    safe_ref = ref or " "
    safe_hyp = hyp or ""
    return 0.5 * float(wer(safe_ref, safe_hyp)) + 0.5 * float(cer(safe_ref, safe_hyp))


def score(refs: list[str], hyps: list[str]) -> dict[str, float]:
    safe_refs = [r or " " for r in refs]
    safe_hyps = [h or "" for h in hyps]
    w = float(wer(safe_refs, safe_hyps))
    c = float(cer(safe_refs, safe_hyps))
    return {"wer": w, "cer": c, "zindi": 1.0 - 0.5 * (w + c)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-root", type=Path, default=Path.home() / ".cache" / "huggingface" / "datasets")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "goal_2026_08_07" / "sna_policy_calibration.json")
    args = ap.parse_args()

    rows = load_rows(args.cache_root)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    rows = [rows[i] for i in order[: args.n]]
    refs = [norm(row.get("transcription", "")) for row in rows]
    audios = [audio_array(row) for row in rows]
    device = pick_device()
    results: dict[str, list[str]] = {}
    waxal_ref = local_snapshot(args.cache_root, WAXAL_SNA)
    badrex_ref = local_snapshot(args.cache_root, BADREX)
    mubarak_ref = local_snapshot(args.cache_root, MUBARAK)

    processor = AutoProcessor.from_pretrained(waxal_ref, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(waxal_ref, local_files_only=True).to(device).eval()
    results["waxal"] = [decode_mms(model, processor, audio, device) for audio in audios]
    del model, processor

    processor = AutoProcessor.from_pretrained(badrex_ref, local_files_only=True)
    model = Wav2Vec2BertForCTC.from_pretrained(badrex_ref, local_files_only=True).to(device).eval()
    results["badrex"] = [decode_badrex(model, processor, audio, device) for audio in audios]
    del model, processor

    processor = WhisperProcessor.from_pretrained(mubarak_ref, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(mubarak_ref, local_files_only=True).to(device).eval()
    results["mubarak"] = [decode_mubarak(model, processor, audio, device) for audio in audios]
    del model, processor

    rows_out = []
    for i, row in enumerate(rows):
        costs = {name: row_cost(refs[i], hyps[i]) for name, hyps in results.items()}
        rows_out.append(
            {
                "id": row["id"],
                "ref": refs[i],
                "waxal": results["waxal"][i],
                "badrex": results["badrex"][i],
                "mubarak": results["mubarak"][i],
                "cost": costs,
                "badrex_wins": costs["badrex"] < costs["waxal"],
                "mubarak_wins": costs["mubarak"] < costs["waxal"],
            }
        )

    refs = [r["ref"] for r in rows_out]
    policies = {
        "waxal": [r["waxal"] for r in rows_out],
        "badrex": [r["badrex"] for r in rows_out],
        "mubarak": [r["mubarak"] for r in rows_out],
        "best_oracle": [
            min(
                ((r["waxal"], r["cost"]["waxal"]), (r["badrex"], r["cost"]["badrex"]), (r["mubarak"], r["cost"]["mubarak"])),
                key=lambda pair: pair[1],
            )[0]
            for r in rows_out
        ],
    }
    payload = {
        "n": len(rows_out),
        "seed": args.seed,
        "device": str(device),
        "scores": {name: score(refs, hyps) for name, hyps in policies.items()},
        "counts": {
            "badrex_beats_waxal": sum(r["badrex_wins"] for r in rows_out),
            "mubarak_beats_waxal": sum(r["mubarak_wins"] for r in rows_out),
            "badrex_or_mubarak_beats_waxal": sum(r["badrex_wins"] or r["mubarak_wins"] for r in rows_out),
        },
        "rows": rows_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "scores": payload["scores"], "counts": payload["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
