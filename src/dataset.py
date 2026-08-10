"""PyTorch / HF datasets wrappers for WAXAL ASR fine-tuning."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from torch.utils.data import Dataset as TorchDataset

from src.config import (
    FORBIDDEN_TRAIN_SPLITS,
    HF_CONFIGS,
    HF_DATASET,
    LANGUAGES,
    MAX_AUDIO_SECONDS,
    TARGET_SR,
)
from src.text_norm import normalize_text

logger = logging.getLogger(__name__)


def _decode_audio_item(audio_obj: Any, target_sr: int = TARGET_SR) -> dict:
    """Decode HF audio dict (bytes/path/array) via soundfile/librosa — no torchcodec."""
    import io

    import librosa
    import soundfile as sf

    if isinstance(audio_obj, dict) and "array" in audio_obj and audio_obj["array"] is not None:
        array = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj.get("sampling_rate") or target_sr)
        if sr != target_sr:
            array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
        return {"array": array, "sampling_rate": target_sr}

    raw = None
    path = None
    if isinstance(audio_obj, dict):
        raw = audio_obj.get("bytes")
        path = audio_obj.get("path")
    elif isinstance(audio_obj, (bytes, bytearray)):
        raw = bytes(audio_obj)
    elif isinstance(audio_obj, str):
        path = audio_obj

    if raw is not None:
        array, sr = sf.read(io.BytesIO(raw), dtype="float32")
    elif path is not None:
        array, sr = sf.read(path, dtype="float32")
    else:
        raise ValueError(f"Cannot decode audio object: type={type(audio_obj)}")

    if getattr(array, "ndim", 1) > 1:
        array = array.mean(axis=-1)
    array = np.asarray(array, dtype=np.float32)
    if int(sr) != target_sr:
        array = librosa.resample(array, orig_sr=int(sr), target_sr=target_sr)
    return {"array": array, "sampling_rate": target_sr}


def load_hf_asr_split(
    lang: str,
    split: str,
    *,
    streaming: bool = False,
    max_samples: int | None = None,
    allow_test: bool = False,
) -> Dataset:
    """Load one language/split from google/WaxalNLP.

    Never call this with split='test' for training — callers must exclude test.
    Test access is explicit because the split has no legitimate role in model
    fitting or tuning. When explicitly enabled, test transcriptions are
    removed from returned examples so prediction code cannot accidentally
    score against them.
    When max_samples is small, use streaming + take to avoid downloading entire shards.
    Audio is decoded with soundfile/librosa (open-source) to avoid torchcodec dependency.
    """
    if split in FORBIDDEN_TRAIN_SPLITS and not allow_test:
        raise ValueError(
            f"Refusing to load forbidden split '{split}' without allow_test=True; "
            "Phase-1 test gold must not enter training, tuning, or diagnostics."
        )
    # Challenge langs in HF_CONFIGS; other WAXAL langs use {lang}_asr configs.
    config = HF_CONFIGS.get(lang, f"{lang}_asr")

    # Prefer explicit split parquet globs so we do NOT download train/unlabeled
    # when only test/validation is needed. When max_samples is set we still use
    # parquet (if available) then select+decode only N rows — streaming is a
    # fallback when no shards can be resolved.
    if not streaming:
        try:
            import json
            import os  # used later for WAXAL_EAGER_AUDIO too
            from pathlib import Path
            from urllib.request import Request, urlopen

            from huggingface_hub import hf_hub_download

            needle = f"{lang}-{split}-"
            local_files: list[str] = []

            # 1) Prefer already-cached hub snapshot files (no network).
            hub_ds = HF_DATASET.replace("/", "--")
            cache_root = Path(
                os.environ.get("HF_HUB_CACHE")
                or (Path.home() / ".cache/huggingface/hub")
            )
            snap_root = cache_root / f"datasets--{hub_ds}" / "snapshots"
            if snap_root.is_dir():
                for snap in sorted(snap_root.iterdir(), reverse=True):
                    asr_dir = snap / "data" / "ASR" / lang
                    if not asr_dir.is_dir():
                        continue
                    found = sorted(
                        p
                        for p in asr_dir.glob("*.parquet")
                        if needle in p.name and p.resolve().is_file()
                    )
                    if found:
                        local_files = [str(p.resolve()) for p in found]
                        logger.info(
                            "Loading %d cached parquet file(s) for %s/%s from %s",
                            len(local_files),
                            lang,
                            split,
                            asr_dir,
                        )
                        break

            # 2) Else list remote tree and download only matching shards.
            if not local_files:
                api_url = (
                    f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main/"
                    f"data/ASR/{lang}?recursive=false"
                )
                req = Request(api_url, headers={"User-Agent": "waxal-asr-solution/1.0"})
                with urlopen(req, timeout=30) as resp:
                    tree = json.loads(resp.read().decode("utf-8"))
                paths = sorted(
                    item["path"]
                    for item in tree
                    if isinstance(item, dict)
                    and str(item.get("path", "")).endswith(".parquet")
                    and needle in str(item.get("path", ""))
                )
                if paths:
                    logger.info(
                        "Loading %d parquet file(s) for %s/%s via explicit data_files",
                        len(paths),
                        lang,
                        split,
                    )
                    local_files = [
                        hf_hub_download(
                            HF_DATASET, p, repo_type="dataset", local_files_only=False
                        )
                        for p in paths
                    ]

            if local_files:
                ds = load_dataset("parquet", data_files={split: local_files}, split=split)
            else:
                logger.info("Streaming full split %s/%s (no explicit shards found)", config, split)
                stream = load_dataset(HF_DATASET, config, split=split, streaming=True)
                try:
                    stream = stream.cast_column("audio", Audio(decode=False))
                except Exception:
                    pass
                rows = []
                for ex in stream:
                    ex = dict(ex)
                    ex["audio"] = _decode_audio_item(ex["audio"], TARGET_SR)
                    rows.append(ex)
                    if max_samples is not None and len(rows) >= max_samples:
                        break
                out = Dataset.from_list(rows)
                if split in FORBIDDEN_TRAIN_SPLITS:
                    out = out.remove_columns(["transcription"] if "transcription" in out.column_names else [])
                return out
        except Exception as e:
            logger.warning("explicit parquet load failed (%s); falling back to config load", e)
            ds = load_dataset(HF_DATASET, config, split=split)
    else:
        ds = load_dataset(HF_DATASET, config, split=split, streaming=True)
        if split in FORBIDDEN_TRAIN_SPLITS:
            return ds.remove_columns(["transcription"] if "transcription" in ds.column_names else [])
        return ds

    # Full map-style: decode with soundfile into plain dicts (avoid Audio feature /
    # torchcodec encode path when writing mapped datasets).
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    if max_samples is not None:
        n = min(max_samples, len(ds))
        ds = ds.select(range(n))

    # Test transcripts are challenge gold. Remove them on every map-style
    # return path, including cached parquet, so prediction code cannot score
    # against them accidentally.
    if split in FORBIDDEN_TRAIN_SPLITS and "transcription" in ds.column_names:
        ds = ds.remove_columns(["transcription"])

    # Lazy path: keep encoded audio in parquet; decode one sample at a time.
    # Eager full-decode of train (5k–14k clips) can exceed tens of GB RAM.
    if os.environ.get("WAXAL_EAGER_AUDIO", "0") == "1":
        rows = []
        for i in range(len(ds)):
            ex = dict(ds[i])
            ex["audio"] = _decode_audio_item(ex["audio"], TARGET_SR)
            rows.append(ex)
        out = Dataset.from_list(rows)
        if split in FORBIDDEN_TRAIN_SPLITS:
            out = out.remove_columns(["transcription"] if "transcription" in out.column_names else [])
        return out

    # set_transform receives column-oriented batches (lists); decode on access only.
    def _lazy_decode(batch):
        batch = dict(batch)
        audio = batch.get("audio")
        if isinstance(audio, list):
            batch["audio"] = [_decode_audio_item(a, TARGET_SR) for a in audio]
        else:
            batch["audio"] = _decode_audio_item(audio, TARGET_SR)
        return batch

    ds.set_transform(_lazy_decode)
    return ds


def _apply_lazy_audio_decode(ds: Dataset) -> Dataset:
    """Decode HF audio bytes/path → array on access (soundfile/librosa).

    Must be re-applied after concatenate_datasets (transforms are dropped).
    """
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass

    def _lazy_decode(batch):
        batch = dict(batch)
        audio = batch.get("audio")
        if isinstance(audio, list):
            batch["audio"] = [_decode_audio_item(a, TARGET_SR) for a in audio]
        else:
            batch["audio"] = _decode_audio_item(audio, TARGET_SR)
        return batch

    ds.set_transform(_lazy_decode)
    return ds


def load_labeled_splits(
    languages: tuple[str, ...] = LANGUAGES,
    splits: tuple[str, ...] = ("train", "validation"),
    max_per_lang_split: int | None = None,
) -> Dataset:
    """Concatenate allowed labeled splits across languages.

    Raises if 'test' is requested (hard gate against test-gold leakage).
    """
    for s in splits:
        if s in FORBIDDEN_TRAIN_SPLITS:
            raise ValueError(
                f"Refusing to load forbidden split '{s}' into a training/tuning dataset"
            )
    parts = []
    for lang in languages:
        for split in splits:
            ds = load_hf_asr_split(lang, split, max_samples=max_per_lang_split)
            # Ensure language field present
            if "language" not in ds.column_names:
                ds = ds.add_column("language", [lang] * len(ds))
            parts.append(ds)
    out = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    # concatenate_datasets drops per-part set_transform — re-bind audio decode
    if os.environ.get("WAXAL_EAGER_AUDIO", "0") != "1":
        out = _apply_lazy_audio_decode(out)
    return out


@dataclass
class WhisperDataCollator:
    """Pad audio features and labels for Whisper fine-tuning."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Whisper: force decoder start if present as first label
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def prepare_whisper_example(
    batch: dict,
    processor: Any,
    max_seconds: float = MAX_AUDIO_SECONDS,
) -> dict:
    """Map function: waveform + text -> input_features + labels.

    Returns *only* model fields so datasets.map does not re-encode Audio
    (avoids torchcodec dependency on write).
    """
    audio = batch["audio"]
    if not isinstance(audio, dict) or "array" not in audio:
        audio = _decode_audio_item(audio, TARGET_SR)
    array = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    max_len = int(max_seconds * sr)
    if array.shape[0] > max_len:
        array = array[:max_len]

    inputs = processor.feature_extractor(
        array, sampling_rate=sr, return_tensors="np"
    )
    text = normalize_text(batch.get("transcription") or batch.get("Target") or "")
    return {
        "input_features": inputs.input_features[0],
        "labels": processor.tokenizer(text).input_ids,
    }


class IndexedAudioDataset(TorchDataset):
    """Map-style dataset that loads audio from HF by id for inference."""

    def __init__(
        self,
        frame: pd.DataFrame,
        id_to_audio: dict[str, dict],
        processor: Any,
        text_column: str | None = "Target",
    ):
        self.frame = frame.reset_index(drop=True)
        self.id_to_audio = id_to_audio
        self.processor = processor
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        sample_id = str(row["ID"] if "ID" in row.index else row["id"])
        audio = self.id_to_audio[sample_id]
        array = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate", TARGET_SR))
        max_len = int(MAX_AUDIO_SECONDS * sr)
        if array.shape[0] > max_len:
            array = array[:max_len]
        feats = self.processor.feature_extractor(
            array, sampling_rate=sr, return_tensors="pt"
        ).input_features[0]
        item = {
            "ID": sample_id,
            "input_features": feats,
            "language": row.get("language", ""),
        }
        if self.text_column and self.text_column in row.index:
            item["reference"] = normalize_text(row[self.text_column])
        return item
