#!/usr/bin/env python3
"""Alignment-safe, speaker-disjoint CTC adaptation for WAXAL.

The default recipe adapts the highest-gap challenge language (Luganda) from the
WAXAL 300M specialist using Harcuracy's corrected train transcripts.  It never
loads a test split and never crops a waveform while retaining the full label.

Safety properties:
  * train/validation are made from the source *train* split by speaker_id;
  * full waveforms are passed to the processor with truncation disabled;
  * CTC target length includes the blank frames required by adjacent repeats;
  * target/output geometry is checked before every forward with labels;
  * unknown target characters never silently become [UNK];
  * the tokenizer word delimiter and CTC blank IDs are checked explicitly.

Use --self-test for a dependency-light smoke test and --audit-only to build the
real data/model audit without starting training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# MPS can execute the acoustic model while CTC loss falls back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_norm import normalize_text

LOG = logging.getLogger("alignment_safe_waxal")
TARGET_SR = 16_000
DEFAULT_LANGUAGE = "lug"
DEFAULT_MODEL = "waxal-benchmarking/mms-300m-waxal-lug"
HARCURACY_DATASET = "Harcuracy/google_waxal_asr_challenge"
# Pin the corrected corpus used by this recipe so later edits cannot silently
# change labels under an existing experiment name.
HARCURACY_REVISION = "1b53b0eeb92b48576b353545a5e5644d1cb526be"
GOOGLE_DATASET = "google/WaxalNLP"
SUPPORTED_HARCURACY_LANGS = frozenset({"lin", "sna", "lug"})
FORBIDDEN_SPLITS = frozenset({"test"})
APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
        "\u02bb": "'",
        "`": "'",
        "\u00b4": "'",
    }
)
LAYER_RE = re.compile(r"(?:^|\.)encoder\.layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class SpeakerSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_speakers: frozenset[str]
    validation_speakers: frozenset[str]


@dataclass(frozen=True)
class TranscriptAudit:
    projected_text: tuple[str, ...]
    eligible_indices: tuple[int, ...]
    changed_rows: int
    empty_rows: int
    dropped_characters: dict[str, int]


def seed_everything(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_score(seed: int, value: str) -> int:
    payload = f"{seed}\0{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def speaker_disjoint_split(
    speaker_ids: Sequence[str], validation_fraction: float, seed: int
) -> SpeakerSplit:
    """Deterministically hold out whole speakers, targeting an utterance fraction."""
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("speaker validation fraction must be in (0, 0.5)")
    groups: dict[str, list[int]] = {}
    for index, raw_speaker in enumerate(speaker_ids):
        speaker = str(raw_speaker).strip()
        if not speaker or speaker.lower() in {"nan", "none", "null"}:
            raise ValueError(f"missing speaker_id at eligible row {index}")
        groups.setdefault(speaker, []).append(index)
    if len(groups) < 2:
        raise ValueError("speaker-disjoint validation requires at least two speakers")

    target_rows = max(1, round(len(speaker_ids) * validation_fraction))
    ordered = sorted(groups, key=lambda s: (stable_score(seed, s), s))
    validation_speakers: set[str] = set()
    validation_rows = 0
    # Add whole groups until the requested number of validation utterances is met.
    # Keep at least one speaker for training even for tiny smoke datasets.
    for speaker in ordered[:-1]:
        if validation_rows >= target_rows and validation_speakers:
            break
        validation_speakers.add(speaker)
        validation_rows += len(groups[speaker])

    train_speakers = set(groups) - validation_speakers
    train_indices = tuple(
        i for i, speaker in enumerate(speaker_ids) if str(speaker).strip() in train_speakers
    )
    validation_indices = tuple(
        i
        for i, speaker in enumerate(speaker_ids)
        if str(speaker).strip() in validation_speakers
    )
    if not train_indices or not validation_indices:
        raise AssertionError("speaker split unexpectedly produced an empty partition")
    if train_speakers & validation_speakers:
        raise AssertionError("speaker leakage between train and validation")
    if set(train_indices) & set(validation_indices):
        raise AssertionError("row leakage between train and validation")
    if len(train_indices) + len(validation_indices) != len(speaker_ids):
        raise AssertionError("speaker split lost rows")
    return SpeakerSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_speakers=frozenset(train_speakers),
        validation_speakers=frozenset(validation_speakers),
    )


def canonical_transcript(text: Any) -> str:
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).translate(APOSTROPHE_TRANSLATION)
    return normalize_text(value)


def tokenizer_literal_vocabulary(tokenizer: Any) -> set[str]:
    vocab = tokenizer.get_vocab()
    if vocab and isinstance(next(iter(vocab.values())), dict):
        target = getattr(tokenizer, "target_lang", None)
        if target is None or target not in vocab:
            raise ValueError("multilingual tokenizer has no active target language")
        vocab = vocab[target]
    return set(vocab)


def find_literal_token_id(tokenizer: Any, token: str) -> int | None:
    for token_id in range(len(tokenizer)):
        if tokenizer.convert_ids_to_tokens(token_id) == token:
            return token_id
    return None


def validate_tokenizer_contract(tokenizer: Any, model: nn.Module) -> dict[str, Any]:
    pipe_id = find_literal_token_id(tokenizer, "|")
    if pipe_id is None:
        raise ValueError("CTC tokenizer has no literal '|' word delimiter")
    tokenizer.word_delimiter_token = "|"
    tokenizer.word_delimiter_token_id = pipe_id
    if hasattr(tokenizer, "_word_delimiter_token"):
        tokenizer._word_delimiter_token = "|"

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("CTC tokenizer has no pad token; the blank ID is ambiguous")
    model_pad_id = getattr(model.config, "pad_token_id", None)
    if model_pad_id is None:
        model.config.pad_token_id = int(pad_id)
    elif int(model_pad_id) != int(pad_id):
        raise ValueError(
            f"model CTC blank id ({model_pad_id}) != tokenizer pad id ({pad_id})"
        )
    if pipe_id == int(pad_id):
        raise ValueError("word delimiter and CTC blank cannot share an ID")

    probe = tokenizer("a b", add_special_tokens=False).input_ids
    if pipe_id not in probe:
        raise ValueError(f"tokenizer does not encode spaces as '|': probe={probe}")
    head = getattr(model, "lm_head", None)
    if head is None or not isinstance(head, nn.Linear):
        raise TypeError("expected an AutoModelForCTC with a linear lm_head")
    if head.out_features != len(tokenizer):
        raise ValueError(
            f"lm_head outputs={head.out_features}, tokenizer size={len(tokenizer)}"
        )
    if int(model.config.vocab_size) != head.out_features:
        raise ValueError(
            f"config vocab_size={model.config.vocab_size}, lm_head={head.out_features}"
        )
    return {
        "vocab_size": len(tokenizer),
        "blank_id": int(pad_id),
        "word_delimiter_id": int(pipe_id),
        "unk_id": tokenizer.unk_token_id,
        "probe_ids": [int(x) for x in probe],
    }


def add_apostrophe_token(tokenizer: Any, model: nn.Module) -> bool:
    """Append an apostrophe output while preserving every pretrained CTC row."""
    if find_literal_token_id(tokenizer, "'") is not None:
        return False
    head = getattr(model, "lm_head", None)
    if head is None or not isinstance(head, nn.Linear):
        raise TypeError("cannot expand vocabulary: model has no linear lm_head")
    old_size = head.out_features
    if old_size != len(tokenizer):
        raise ValueError("refusing vocabulary expansion with an existing head/tokenizer mismatch")
    added = tokenizer.add_tokens(["'"])
    if added != 1 or len(tokenizer) != old_size + 1:
        raise RuntimeError(
            f"apostrophe vocabulary expansion failed: added={added}, size={len(tokenizer)}"
        )

    new_head = nn.Linear(head.in_features, len(tokenizer), bias=head.bias is not None)
    new_head.to(device=head.weight.device, dtype=head.weight.dtype)
    with torch.no_grad():
        new_head.weight[:old_size].copy_(head.weight)
        if head.bias is not None and new_head.bias is not None:
            new_head.bias[:old_size].copy_(head.bias)
        # Initialize the new grapheme from frequent letter rows rather than random
        # scale; subsequent corrected-label adaptation learns its acoustic evidence.
        seed_ids = [
            token_id
            for char in "aeiouln"
            if (token_id := find_literal_token_id(tokenizer, char)) is not None
            and token_id < old_size
        ]
        if seed_ids:
            new_head.weight[old_size].copy_(head.weight[seed_ids].mean(dim=0))
            if head.bias is not None and new_head.bias is not None:
                new_head.bias[old_size].copy_(head.bias[seed_ids].mean())
    model.lm_head = new_head
    model.config.vocab_size = len(tokenizer)
    return True


def project_transcript(
    text: Any, tokenizer: Any, unknown_policy: str
) -> tuple[str, Counter[str]]:
    """Project normalized text onto the literal CTC alphabet without [UNK] labels."""
    normalized = canonical_transcript(text)
    literals = tokenizer_literal_vocabulary(tokenizer)
    dropped: Counter[str] = Counter()
    output: list[str] = []
    for char in normalized:
        if char == " ":
            output.append(char)
        elif char in literals:
            output.append(char)
        elif unknown_policy == "error":
            raise ValueError(f"unsupported target character {char!r} in {normalized!r}")
        else:
            dropped[char] += 1
    projected = " ".join("".join(output).split())
    if projected:
        ids = tokenizer(projected, add_special_tokens=False).input_ids
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in ids:
            raise ValueError(f"[UNK] survived literal projection for {projected!r}")
    return projected, dropped


def audit_transcripts(
    metadata: Dataset, tokenizer: Any, language: str, unknown_policy: str
) -> TranscriptAudit:
    required = {"id", "speaker_id", "transcription", "language"}
    missing = required - set(metadata.column_names)
    if missing:
        raise ValueError(f"dataset missing required columns: {sorted(missing)}")
    seen_ids: set[str] = set()
    projected_all: list[str] = []
    eligible: list[int] = []
    changed_rows = 0
    empty_rows = 0
    dropped_total: Counter[str] = Counter()
    for index in range(len(metadata)):
        row = metadata[index]
        row_id = str(row["id"])
        if row_id in seen_ids:
            raise ValueError(f"duplicate source id: {row_id}")
        seen_ids.add(row_id)
        if str(row["language"]).strip().lower() != language:
            raise ValueError(
                f"language mismatch for {row_id}: {row['language']!r} != {language!r}"
            )
        original = canonical_transcript(row["transcription"])
        projected, dropped = project_transcript(
            row["transcription"], tokenizer, unknown_policy
        )
        projected_all.append(projected)
        dropped_total.update(dropped)
        if projected != original:
            changed_rows += 1
        if not projected:
            empty_rows += 1
            continue
        eligible.append(index)
    if not eligible:
        raise ValueError("no eligible labeled rows after transcript projection")
    return TranscriptAudit(
        projected_text=tuple(projected_all),
        eligible_indices=tuple(eligible),
        changed_rows=changed_rows,
        empty_rows=empty_rows,
        dropped_characters=dict(sorted(dropped_total.items())),
    )


def subset_with_targets(raw: Dataset, audit: TranscriptAudit) -> Dataset:
    selected = raw.select(list(audit.eligible_indices))
    targets = [audit.projected_text[index] for index in audit.eligible_indices]
    if "_ctc_text" in selected.column_names:
        selected = selected.remove_columns(["_ctc_text"])
    return selected.add_column("_ctc_text", targets)


def limit_indices(indices: Sequence[int], maximum: int | None, seed: int) -> tuple[int, ...]:
    if maximum is None or maximum >= len(indices):
        return tuple(indices)
    if maximum <= 0:
        raise ValueError("sample limits must be positive")
    return tuple(sorted(indices, key=lambda i: stable_score(seed, str(i)))[:maximum])


def load_source_train(
    *,
    source: str,
    language: str,
    revision: str | None,
    local_files_only: bool,
) -> tuple[Dataset, dict[str, Any]]:
    if "test" in FORBIDDEN_SPLITS:  # executable guard against later scope drift
        split = "train"
    else:  # pragma: no cover
        raise AssertionError("forbidden split policy was modified")
    if split in FORBIDDEN_SPLITS:
        raise AssertionError("test split cannot be loaded by this entry point")

    if source == "harcuracy":
        if language not in SUPPORTED_HARCURACY_LANGS:
            raise ValueError(
                f"Harcuracy corrected data supports {sorted(SUPPORTED_HARCURACY_LANGS)}, "
                f"not {language!r}"
            )
        dataset_id = HARCURACY_DATASET
        config = f"{language}_asr"
        resolved_revision = revision or HARCURACY_REVISION
    elif source == "google":
        dataset_id = GOOGLE_DATASET
        config = f"{language}_asr"
        resolved_revision = revision
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(source)

    if source == "harcuracy":
        path_prefix = f"{config}/train-"
    else:
        path_prefix = f"data/ASR/{language}/{language}-train-"

    # Do not use the repository's dataset configuration builder here.  Some
    # builders resolve/materialize every split even when split="train" is
    # requested. Resolve only filenames in the train prefix, then construct a
    # parquet dataset from that explicit allow-list.
    repo_cache_name = f"datasets--{dataset_id.replace('/', '--')}"
    hub_cache = Path(
        os.environ.get("HF_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface" / "hub")
    )
    snapshot_root = hub_cache / repo_cache_name / "snapshots"
    local_train_files: list[str] = []
    if snapshot_root.is_dir():
        snapshots = sorted(snapshot_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if resolved_revision:
            exact = snapshot_root / resolved_revision
            snapshots = [exact] if exact.is_dir() else snapshots
        for snapshot in snapshots:
            matches = sorted(
                path.resolve()
                for path in snapshot.rglob("*.parquet")
                if path.relative_to(snapshot).as_posix().startswith(path_prefix)
            )
            if matches:
                local_train_files = [str(path) for path in matches]
                break

    if not local_train_files:
        if local_files_only:
            raise FileNotFoundError(
                f"no cached train parquet files for {dataset_id}/{config} "
                f"under {snapshot_root}"
            )
        repo_files = HfApi().list_repo_files(
            dataset_id, repo_type="dataset", revision=resolved_revision
        )
        train_paths = sorted(
            path
            for path in repo_files
            if path.startswith(path_prefix) and path.endswith(".parquet")
        )
        if not train_paths:
            raise FileNotFoundError(
                f"no train parquet files matched {dataset_id}:{path_prefix}*.parquet"
            )
        local_train_files = [
            hf_hub_download(
                dataset_id,
                path,
                repo_type="dataset",
                revision=resolved_revision,
                local_files_only=False,
            )
            for path in train_paths
        ]

    bad_paths = [
        path
        for path in local_train_files
        if any(marker in Path(path).name for marker in ("-test-", "-validation-", "-unlabeled-"))
    ]
    if bad_paths:
        raise AssertionError(f"non-train shard entered allow-list: {bad_paths}")
    raw = load_dataset(
        "parquet", data_files={split: local_train_files}, split=split
    )
    if "audio" not in raw.column_names:
        raise ValueError("source dataset has no audio column")
    # Keep bytes/path encoded until each item reaches the collator.  This avoids
    # torchcodec and avoids materializing the multi-GB corpus in RAM.
    raw = raw.cast_column("audio", Audio(decode=False))
    provenance = {
        "dataset_id": dataset_id,
        "config": config,
        "source_split_loaded": split,
        "revision": resolved_revision,
        "corrected_transcripts": source == "harcuracy",
        "test_split_loaded": False,
        "explicit_train_shards_only": True,
        "train_shard_count": len(local_train_files),
    }
    return raw, provenance


def decode_audio(audio: Any, target_sr: int = TARGET_SR) -> np.ndarray:
    if isinstance(audio, dict) and audio.get("array") is not None:
        array = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio.get("sampling_rate") or target_sr)
    else:
        raw_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        path = audio.get("path") if isinstance(audio, dict) else None
        if raw_bytes is not None:
            array, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        elif path:
            array, sample_rate = sf.read(path, dtype="float32")
        else:
            raise ValueError(f"unsupported encoded audio value: {type(audio)!r}")
    if array.ndim > 1:
        array = array.mean(axis=-1)
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("audio is empty or contains non-finite samples")
    if int(sample_rate) != target_sr:
        import librosa

        array = librosa.resample(
            array, orig_sr=int(sample_rate), target_sr=target_sr
        ).astype(np.float32, copy=False)
    if len(array) < target_sr // 10:
        raise ValueError(f"audio shorter than 100 ms: {len(array)} samples")
    return np.ascontiguousarray(array, dtype=np.float32)


def ctc_required_length(label_ids: Sequence[int]) -> int:
    """Frames required by CTC: labels plus blanks between adjacent repeats."""
    if not label_ids:
        return 0
    repeats = sum(left == right for left, right in zip(label_ids, label_ids[1:]))
    return len(label_ids) + repeats


class AlignmentSafeCollator:
    def __init__(self, processor: Any, target_sr: int = TARGET_SR):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.target_sr = target_sr
        names = getattr(processor, "model_input_names", None) or getattr(
            processor.feature_extractor, "model_input_names", None
        )
        self.main_input_name = names[0] if names else "input_values"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        arrays = [decode_audio(feature["audio"], self.target_sr) for feature in features]
        raw_lengths = torch.tensor([len(array) for array in arrays], dtype=torch.long)
        batch = self.processor(
            arrays,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        if self.main_input_name not in batch:
            raise KeyError(
                f"processor did not return expected input {self.main_input_name!r}: "
                f"{list(batch.keys())}"
            )
        if self.main_input_name == "input_values":
            observed = batch["attention_mask"].sum(dim=-1).cpu()
            if not torch.equal(observed, raw_lengths):
                raise ValueError(
                    "processor changed waveform lengths despite truncation=False: "
                    f"raw={raw_lengths.tolist()} processed={observed.tolist()}"
                )

        encoded: list[list[int]] = []
        required: list[int] = []
        label_lengths: list[int] = []
        forbidden = {
            token_id
            for token_id in (
                self.tokenizer.unk_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
            )
            if token_id is not None
        }
        for feature in features:
            ids = [
                int(x)
                for x in self.tokenizer(
                    feature["_ctc_text"], add_special_tokens=False
                ).input_ids
            ]
            if not ids:
                raise ValueError(f"empty CTC labels for id={feature.get('id')}")
            illegal = forbidden.intersection(ids)
            if illegal:
                raise ValueError(
                    f"special/unknown IDs {sorted(illegal)} in target id={feature.get('id')}"
                )
            if min(ids) < 0 or max(ids) >= len(self.tokenizer):
                raise ValueError(f"target ID outside vocabulary for id={feature.get('id')}")
            encoded.append(ids)
            label_lengths.append(len(ids))
            required.append(ctc_required_length(ids))

        max_labels = max(label_lengths)
        labels = torch.full((len(encoded), max_labels), -100, dtype=torch.long)
        for row, ids in enumerate(encoded):
            labels[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        batch["labels"] = labels
        batch["_ctc_label_lengths"] = torch.tensor(label_lengths, dtype=torch.long)
        batch["_ctc_required_lengths"] = torch.tensor(required, dtype=torch.long)
        batch["_raw_audio_samples"] = raw_lengths
        return dict(batch)


def model_output_lengths(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    main_input_name = getattr(model, "main_input_name", "input_values")
    if main_input_name not in inputs:
        candidates = [k for k in ("input_values", "input_features") if k in inputs]
        if len(candidates) != 1:
            raise KeyError(f"cannot identify model input from {list(inputs)}")
        main_input_name = candidates[0]
    tensor = inputs[main_input_name]
    mask = inputs.get("attention_mask")
    if mask is not None:
        input_lengths = mask.to(dtype=torch.long).sum(dim=-1)
    elif tensor.ndim == 2:
        input_lengths = torch.full(
            (tensor.shape[0],), tensor.shape[-1], dtype=torch.long, device=tensor.device
        )
    else:
        input_lengths = torch.full(
            (tensor.shape[0],), tensor.shape[-2], dtype=torch.long, device=tensor.device
        )
    length_fn = getattr(model, "_get_feat_extract_output_lengths", None)
    if length_fn is None:
        raise TypeError(
            f"{type(model).__name__} does not expose exact CTC output-length geometry"
        )
    output_lengths = length_fn(input_lengths)
    return torch.as_tensor(output_lengths, dtype=torch.long, device=input_lengths.device)


def validate_ctc_geometry(
    model: nn.Module,
    model_inputs: dict[str, torch.Tensor],
    required_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
) -> torch.Tensor:
    output_lengths = model_output_lengths(model, model_inputs)
    required_lengths = required_lengths.to(output_lengths.device)
    label_lengths = label_lengths.to(output_lengths.device)
    if output_lengths.shape != required_lengths.shape:
        raise ValueError(
            f"length batch mismatch: output={tuple(output_lengths.shape)}, "
            f"required={tuple(required_lengths.shape)}"
        )
    bad = required_lengths > output_lengths
    if bad.any():
        rows = bad.nonzero(as_tuple=False).flatten().tolist()
        detail = [
            {
                "batch_row": int(i),
                "output_frames": int(output_lengths[i]),
                "label_tokens": int(label_lengths[i]),
                "ctc_required_frames": int(required_lengths[i]),
            }
            for i in rows
        ]
        raise ValueError(f"impossible CTC alignment; refusing batch: {detail}")
    return output_lengths


class AlignmentSafeTrainer(Trainer):
    """Trainer that refuses impossible CTC batches before loss computation."""

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        batch = dict(inputs)
        required = batch.pop("_ctc_required_lengths")
        label_lengths = batch.pop("_ctc_label_lengths")
        batch.pop("_raw_audio_samples")
        labels = batch.get("labels")
        if labels is None:
            raise ValueError("CTC training batch has no labels")
        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        validate_ctc_geometry(model, model_inputs, required, label_lengths)
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite CTC loss: {loss}")
        return (loss, outputs) if return_outputs else loss


def configure_trainable(model: nn.Module, mode: str, top_layers: int) -> dict[str, Any]:
    if mode not in {"head", "top", "full"}:
        raise ValueError(mode)
    if top_layers <= 0:
        raise ValueError("top_layers must be positive")
    layer_ids = [
        int(match.group(1))
        for name, _ in model.named_parameters()
        if (match := LAYER_RE.search(name)) is not None
    ]
    highest_layer = max(layer_ids) if layer_ids else None
    threshold = (
        max(0, highest_layer - top_layers + 1) if highest_layer is not None else None
    )
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    if mode != "full":
        for name, parameter in model.named_parameters():
            train = name.startswith("lm_head") or ".lm_head." in name
            match = LAYER_RE.search(name)
            if mode == "top" and match is not None and threshold is not None:
                train = train or int(match.group(1)) >= threshold
            parameter.requires_grad = train
    if mode == "full" and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError("trainable mode selected zero parameters")
    return {
        "mode": mode,
        "top_layers": top_layers if mode == "top" else None,
        "highest_encoder_layer": highest_layer,
        "first_trainable_encoder_layer": threshold if mode == "top" else None,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
    }


@torch.inference_mode()
def preflight_batches(
    model: nn.Module,
    dataset: Dataset,
    collator: AlignmentSafeCollator,
    device: torch.device,
    batches: int,
) -> dict[str, Any]:
    if batches <= 0:
        return {"batches": 0, "examples": 0}
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
    model.to(device).eval()
    checked_examples = 0
    minima: list[int] = []
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= batches:
            break
        batch = {k: v.to(device) for k, v in raw_batch.items()}
        required = batch.pop("_ctc_required_lengths")
        label_lengths = batch.pop("_ctc_label_lengths")
        batch.pop("_raw_audio_samples")
        labels = batch.pop("labels")
        del labels
        output_lengths = validate_ctc_geometry(
            model, batch, required, label_lengths
        )
        outputs = model(**batch)
        if outputs.logits.ndim != 3:
            raise ValueError(f"expected [batch,time,vocab] logits, got {outputs.logits.shape}")
        if outputs.logits.shape[-1] != model.config.vocab_size:
            raise ValueError("logit/tokenizer vocabulary mismatch during preflight")
        if int(output_lengths.max()) != int(outputs.logits.shape[1]):
            raise ValueError(
                "model length function disagrees with actual logits: "
                f"computed={int(output_lengths.max())}, logits={outputs.logits.shape[1]}"
            )
        minima.extend((output_lengths - required.to(device)).cpu().tolist())
        checked_examples += len(required)
    model.train()
    return {
        "batches": min(batches, checked_examples),
        "examples": checked_examples,
        "minimum_ctc_frame_margin": min(minima) if minima else None,
    }


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def json_ready_counter(counter: dict[str, int]) -> dict[str, int]:
    return {repr(key): int(value) for key, value in sorted(counter.items())}


def run_self_test() -> None:
    speakers = ["a", "a", "b", "b", "c", "d", "e", "f"]
    first = speaker_disjoint_split(speakers, 0.25, 42)
    second = speaker_disjoint_split(speakers, 0.25, 42)
    assert first == second
    assert first.train_speakers.isdisjoint(first.validation_speakers)
    assert len(first.train_indices) + len(first.validation_indices) == len(speakers)
    assert ctc_required_length([1, 1, 2, 3, 3]) == 7
    assert ctc_required_length([1, 2, 3]) == 3
    assert canonical_transcript("  Ng\u2019enda, WAKA! ") == "ng'enda waka"

    class DummyModel(nn.Module):
        main_input_name = "input_values"

        def _get_feat_extract_output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
            return torch.div(lengths - 1, 2, rounding_mode="floor") + 1

    dummy = DummyModel()
    inputs = {
        "input_values": torch.zeros(2, 12),
        "attention_mask": torch.tensor(
            [[1] * 12, [1] * 9 + [0] * 3], dtype=torch.long
        ),
    }
    lengths = validate_ctc_geometry(
        dummy,
        inputs,
        required_lengths=torch.tensor([6, 5]),
        label_lengths=torch.tensor([5, 4]),
    )
    assert lengths.tolist() == [6, 5]
    try:
        validate_ctc_geometry(
            dummy,
            inputs,
            required_lengths=torch.tensor([7, 5]),
            label_lengths=torch.tensor([6, 4]),
        )
    except ValueError as exc:
        assert "impossible CTC alignment" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("impossible CTC target was not rejected")
    print("SELF_TEST_OK speaker_split transcript_normalization ctc_repeat_geometry")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alignment-safe, speaker-disjoint WAXAL CTC adaptation"
    )
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument(
        "--data-source", choices=("harcuracy", "google"), default="harcuracy"
    )
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help=f"defaults to pinned Harcuracy revision {HARCURACY_REVISION}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "checkpoints" / "alignment-safe-lug-harcuracy-v1",
    )
    parser.add_argument("--speaker-validation-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainable-mode", choices=("head", "top", "full"), default="top")
    parser.add_argument("--top-layers", type=int, default=6)
    parser.add_argument("--epochs", type=float, default=6.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--preflight-batches", type=int, default=32)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--add-apostrophe-token", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--unknown-character-policy", choices=("drop", "error"), default="drop"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.self_test:
        run_self_test()
        return 0
    if args.bf16 and args.fp16:
        raise ValueError("choose at most one of --bf16 and --fp16")
    if args.language == "lug" and args.model_id == DEFAULT_MODEL:
        LOG.info("highest-gap-first recipe: Luganda")
    seed_everything(args.seed)
    device = choose_device(args.device)

    LOG.info("loading processor/model %s", args.model_id)
    processor = AutoProcessor.from_pretrained(
        args.model_id, local_files_only=args.local_files_only
    )
    model = AutoModelForCTC.from_pretrained(
        args.model_id, local_files_only=args.local_files_only
    )
    apostrophe_added = False
    if args.add_apostrophe_token:
        apostrophe_added = add_apostrophe_token(processor.tokenizer, model)
    tokenizer_contract = validate_tokenizer_contract(processor.tokenizer, model)
    # Invalid alignments must fail loudly rather than becoming zero-loss examples.
    if hasattr(model.config, "ctc_zero_infinity"):
        model.config.ctc_zero_infinity = False
    trainable = configure_trainable(model, args.trainable_mode, args.top_layers)
    LOG.info(
        "trainable %.2fM / %.2fM (%.2f%%)",
        trainable["trainable_parameters"] / 1e6,
        trainable["total_parameters"] / 1e6,
        100 * trainable["trainable_fraction"],
    )

    raw, provenance = load_source_train(
        source=args.data_source,
        language=args.language,
        revision=args.dataset_revision,
        local_files_only=args.local_files_only,
    )
    metadata = raw.remove_columns(["audio"])
    transcript_audit = audit_transcripts(
        metadata,
        processor.tokenizer,
        args.language,
        args.unknown_character_policy,
    )
    prepared = subset_with_targets(raw, transcript_audit)
    eligible_meta = prepared.remove_columns(["audio"])
    speakers = [str(value) for value in eligible_meta["speaker_id"]]
    split = speaker_disjoint_split(
        speakers, args.speaker_validation_fraction, args.seed
    )
    train_indices = limit_indices(
        split.train_indices, args.max_train_samples, args.seed + 1
    )
    validation_indices = limit_indices(
        split.validation_indices, args.max_validation_samples, args.seed + 2
    )
    train_dataset = prepared.select(list(train_indices))
    validation_dataset = prepared.select(list(validation_indices))
    train_speakers = set(train_dataset["speaker_id"])
    validation_speakers = set(validation_dataset["speaker_id"])
    if train_speakers & validation_speakers:
        raise AssertionError("speaker leakage after sample limiting")
    train_ids = set(train_dataset["id"])
    validation_ids = set(validation_dataset["id"])
    if train_ids & validation_ids:
        raise AssertionError("ID leakage after sample limiting")

    LOG.info(
        "source=%s rows=%d eligible=%d train=%d/%d speakers val=%d/%d speakers",
        provenance["dataset_id"],
        len(raw),
        len(prepared),
        len(train_dataset),
        len(train_speakers),
        len(validation_dataset),
        len(validation_speakers),
    )
    collator = AlignmentSafeCollator(processor)
    preflight = preflight_batches(
        model,
        validation_dataset,
        collator,
        device,
        min(args.preflight_batches, len(validation_dataset)),
    )
    LOG.info("preflight=%s", preflight)

    run_manifest: dict[str, Any] = {
        "language": args.language,
        "highest_gap_language_first": args.language == "lug",
        "model_id": args.model_id,
        "data": provenance,
        "speaker_split": {
            "seed": args.seed,
            "validation_fraction_requested": args.speaker_validation_fraction,
            "train_rows": len(train_dataset),
            "validation_rows": len(validation_dataset),
            "train_speakers": len(train_speakers),
            "validation_speakers": len(validation_speakers),
            "speaker_overlap": len(train_speakers & validation_speakers),
            "id_overlap": len(train_ids & validation_ids),
            "constructed_from_source_split": "train",
            "official_validation_used_for_training": False,
        },
        "transcripts": {
            "corrected_source_preferred": args.data_source == "harcuracy",
            "source_rows": len(raw),
            "eligible_rows": len(prepared),
            "changed_by_vocab_projection": transcript_audit.changed_rows,
            "empty_rows_rejected": transcript_audit.empty_rows,
            "unknown_character_policy": args.unknown_character_policy,
            "dropped_characters": json_ready_counter(
                transcript_audit.dropped_characters
            ),
        },
        "tokenizer": {
            **tokenizer_contract,
            "apostrophe_token_added": apostrophe_added,
            "apostrophe_token_id": find_literal_token_id(processor.tokenizer, "'"),
        },
        "alignment": {
            "waveform_truncation": False,
            "full_labels_with_cropped_audio": False,
            "ctc_repeat_aware_validation": True,
            "model_geometry_function": "_get_feat_extract_output_lengths",
            "runtime_validation_before_every_labeled_forward": True,
            "ctc_zero_infinity": getattr(model.config, "ctc_zero_infinity", None),
            "preflight": preflight,
        },
        "trainable": trainable,
        "device": str(device),
        "seed": args.seed,
        "audit_only": args.audit_only,
    }

    if args.audit_only:
        print(json.dumps(run_manifest, indent=2, ensure_ascii=False))
        LOG.info("audit-only complete; no optimizer steps were run")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "runs"),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=device.type == "cuda",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        max_grad_norm=1.0,
        use_cpu=device.type == "cpu",
    )
    trainer = AlignmentSafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    trainer.train()
    best = args.output_dir / "best"
    trainer.save_model(str(best))
    processor.save_pretrained(str(best))
    run_manifest["training"] = {
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
    }
    run_manifest["output"] = str(best)
    (args.output_dir / "train_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOG.info("saved alignment-safe checkpoint: %s", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
