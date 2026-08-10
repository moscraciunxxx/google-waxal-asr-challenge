#!/usr/bin/env python3
"""Reference-free Lingala specialist routing with strict promotion gates.

The router compares text hypotheses only.  Competition references are used to
measure validation performance and to select two conservative thresholds inside
speaker-disjoint outer folds; they are never features and are never consulted
when routing Phase-2 rows.  The language model and lexicon are built only from
Lingala *train* parquet shards.  No test parquet or test transcript is read.

The script never builds a submission.  It emits a Lingala route cache only when
all promotion gates pass and both 444-row specialist caches are exact matches
for the Phase-2 Lingala route.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from jiwer import process_characters, process_words
from rapidfuzz.distance import Levenshtein
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import score_pairs
from src.text_norm import normalize_text


SOURCE = ROOT / "outputs" / "goal_2026_08_08" / "sulaiman_public_descendants"
OUT = ROOT / "outputs" / "goal_2026_08_08" / "lin_specialist_ensemble"
ROUTE_INDEX = ROOT / "outputs" / "beat075" / "public_visible_index.csv"

VALIDATION_W2V = SOURCE / "validation_w2vbert-lingala-sd3.csv"
VALIDATION_WHISPER = SOURCE / "validation_whisper-small-lingala-cased-2.csv"
VALIDATION_MANIFEST = SOURCE / "validation_manifest_lin.csv"
PHASE2_W2V = SOURCE / "phase2_cache_w2vbert-lingala-sd3.csv"
PHASE2_WHISPER = SOURCE / "phase2_cache_whisper-small-lingala-cased-2.csv"

EXPECTED_VALIDATION_ROWS = 80
EXPECTED_PHASE2_ROWS = 444
OUTER_FOLDS = 5
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260808
HASH_HALF_SEED = "20260808"

# Fixed safety guard.  This is deliberately not tuned per fold: it recognizes
# short CTC collapse (many singleton tokens) only when the independent Whisper
# transcript is much longer and has a better train-only character LM score.
CAT_MAX_WORDS = 8
CAT_SINGLETON_MIN = 0.30
CAT_WHISPER_LENGTH_RATIO_MIN = 2.50

# The only fold-tuned part of the policy is a small, declared grid for close
# W2V/Whisper disagreements.  Smaller edit distance and larger OOV advantage
# are preferred on metric ties.
LEX_EDIT_GRID = (0.01, 0.02, 0.04, 0.08)
LEX_OOV_ADV_GRID = (0.00, 0.02, 0.04, 0.08)


@dataclass(frozen=True)
class RuleConfig:
    max_normalized_edit: float
    min_w2v_oov_advantage: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_or_dot(value: str) -> str:
    return normalize_text(value) or "."


def strict_csv(path: Path, required: Sequence[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: missing {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), None)
    missing = sorted(set(required) - set(header or []))
    if missing:
        raise RuntimeError(f"{label}: missing columns {missing}; header={header}")
    frame = pd.read_csv(
        path,
        usecols=list(required),
        dtype={column: str for column in required},
        keep_default_na=False,
    )
    if "ID" in frame and frame.ID.duplicated().any():
        raise RuntimeError(f"{label}: duplicate IDs")
    return frame


def metric(frame: pd.DataFrame, hypothesis: Sequence[str]) -> dict[str, float | int]:
    result = score_pairs(frame.reference.astype(str).tolist(), list(hypothesis))
    return {
        "n": int(len(frame)),
        "wer": float(result["wer"]),
        "cer": float(result["cer"]),
        "zindi": float(1.0 - result["score"]),
    }


def metric_for_column(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    return metric(frame, frame[column].astype(str).tolist())


def delta_metric(
    frame: pd.DataFrame, candidate_column: str, baseline_column: str = "w2v"
) -> dict[str, Any]:
    baseline = metric_for_column(frame, baseline_column)
    candidate = metric_for_column(frame, candidate_column)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta_zindi": float(candidate["zindi"] - baseline["zindi"]),
    }


def find_waxal_snapshot() -> tuple[Path, list[Path], list[Path]]:
    pattern = str(
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--google--WaxalNLP"
        / "snapshots"
        / "*"
        / "data"
        / "ASR"
        / "lin"
    )
    candidates: list[tuple[Path, list[Path], list[Path]]] = []
    for raw_dir in sorted(glob.glob(pattern)):
        directory = Path(raw_dir)
        train = sorted(directory.glob("lin-train-*.parquet"))
        validation = sorted(directory.glob("lin-validation-*.parquet"))
        if train and validation:
            candidates.append((directory, train, validation))
    if not candidates:
        raise FileNotFoundError(
            "No cached google/WaxalNLP Lingala train+validation parquet snapshot found"
        )
    # Prefer the most complete snapshot, then a deterministic path tie-break.
    return max(candidates, key=lambda item: (len(item[1]) + len(item[2]), str(item[0])))


def load_train_text_and_validation_metadata(
    validation_ids: set[str],
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    snapshot, train_files, validation_files = find_waxal_snapshot()
    if any("test" in path.name or "unlabeled" in path.name for path in train_files + validation_files):
        raise RuntimeError("forbidden split entered parquet projection")

    train_ids: list[str] = []
    train_texts: list[str] = []
    for path in train_files:
        table = pq.read_table(path, columns=["id", "transcription"])
        train_ids.extend(str(value) for value in table.column("id").to_pylist())
        train_texts.extend(normalize_text(value) for value in table.column("transcription").to_pylist())
    if len(train_ids) != len(set(train_ids)):
        raise RuntimeError("duplicate train IDs in cached Lingala shards")
    overlap = validation_ids & set(train_ids)
    if overlap:
        raise RuntimeError(f"train/validation ID leakage: {sorted(overlap)[:5]}")

    validation_parts = []
    for path in validation_files:
        validation_parts.append(
            pq.read_table(path, columns=["id", "speaker_id", "gender"]).to_pandas()
        )
    metadata = pd.concat(validation_parts, ignore_index=True).rename(columns={"id": "ID"})
    metadata = metadata[metadata.ID.astype(str).isin(validation_ids)].copy()
    metadata.ID = metadata.ID.astype(str)
    metadata.speaker_id = metadata.speaker_id.astype(str)
    if metadata.ID.duplicated().any() or set(metadata.ID) != validation_ids:
        raise RuntimeError("validation metadata does not map one-to-one to exact n=80 IDs")
    if metadata.speaker_id.map(lambda value: not value.strip()).any():
        raise RuntimeError("missing speaker_id in exact validation sample")

    audit = {
        "dataset": "google/WaxalNLP",
        "snapshot_revision": snapshot.parents[2].name,
        "snapshot_language_dir": str(snapshot),
        "train_shards": [str(path) for path in train_files],
        "validation_metadata_shards": [str(path) for path in validation_files],
        "train_rows": len(train_texts),
        "train_ids_sha256": sha256_lines(train_ids),
        "test_or_unlabeled_shards_read": False,
        "audio_columns_read": False,
        "validation_reference_column_read_from_parquet": False,
        "train_validation_id_overlap": 0,
    }
    return train_texts, metadata, audit


class TrainLanguageModel:
    def __init__(self, texts: Sequence[str]) -> None:
        self.word_counts: Counter[str] = Counter()
        self.context_counts: Counter[str] = Counter()
        self.trigram_counts: Counter[str] = Counter()
        chars: set[str] = set()
        for raw in texts:
            text = normalize_text(raw)
            words = text.split()
            self.word_counts.update(words)
            chars.update(text)
            padded = "^^" + text + "$"
            for index in range(2, len(padded)):
                self.context_counts[padded[index - 2 : index]] += 1
                self.trigram_counts[padded[index - 2 : index + 1]] += 1
        self.total_words = sum(self.word_counts.values())
        self.vocab_size = max(1, len(self.word_counts))
        self.char_vocab_size = max(1, len(chars))

    def features(self, raw: str) -> dict[str, float | int]:
        text = normalize_text(raw)
        words = text.split()
        denominator = max(1, len(words))
        oov_rate = sum(word not in self.word_counts for word in words) / denominator
        singleton_rate = sum(len(word) == 1 for word in words) / denominator
        repeated_word_rate = (
            max(Counter(words).values()) / denominator if words else 0.0
        )
        unigram_logprob = sum(
            math.log(
                (self.word_counts[word] + 0.5)
                / (self.total_words + 0.5 * self.vocab_size)
            )
            for word in words
        ) / denominator
        padded = "^^" + text + "$"
        char_terms = []
        for index in range(2, len(padded)):
            context = padded[index - 2 : index]
            trigram = padded[index - 2 : index + 1]
            char_terms.append(
                math.log(
                    (self.trigram_counts[trigram] + 0.2)
                    / (self.context_counts[context] + 0.2 * self.char_vocab_size)
                )
            )
        return {
            "words": len(words),
            "characters": len(text),
            "oov_rate": float(oov_rate),
            "singleton_token_rate": float(singleton_rate),
            "repeated_word_rate": float(repeated_word_rate),
            "unigram_logprob_per_word": float(unigram_logprob),
            "char_trigram_logprob": float(sum(char_terms) / max(1, len(char_terms))),
        }


def add_reference_free_features(frame: pd.DataFrame, lm: TrainLanguageModel) -> pd.DataFrame:
    result = frame.copy()
    for model in ("w2v", "whisper", "incumbent"):
        features = pd.DataFrame([lm.features(value) for value in result[model]])
        features.columns = [f"{model}_{column}" for column in features.columns]
        result = pd.concat([result.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    result["w2v_whisper_normalized_edit"] = [
        Levenshtein.normalized_distance(left, right)
        for left, right in zip(result.w2v, result.whisper)
    ]
    result["whisper_w2v_word_ratio"] = result.whisper_words / np.maximum(result.w2v_words, 1)
    result["w2v_oov_advantage"] = result.w2v_oov_rate - result.whisper_oov_rate
    result["whisper_char_lm_advantage"] = (
        result.whisper_char_trigram_logprob - result.w2v_char_trigram_logprob
    )
    result["w2v_incumbent_normalized_edit"] = [
        Levenshtein.normalized_distance(left, right)
        for left, right in zip(result.w2v, result.incumbent)
    ]
    result["whisper_incumbent_normalized_edit"] = [
        Levenshtein.normalized_distance(left, right)
        for left, right in zip(result.whisper, result.incumbent)
    ]
    return result


def catastrophic_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame.w2v_words <= CAT_MAX_WORDS)
        & (frame.w2v_singleton_token_rate >= CAT_SINGLETON_MIN)
        & (frame.whisper_w2v_word_ratio >= CAT_WHISPER_LENGTH_RATIO_MIN)
        & (frame.whisper_char_lm_advantage > 0.0)
    )


def whisper_mask(frame: pd.DataFrame, config: RuleConfig) -> pd.Series:
    catastrophic = catastrophic_mask(frame)
    lexical = (
        (frame.w2v_whisper_normalized_edit <= config.max_normalized_edit)
        & (frame.w2v_oov_advantage > config.min_w2v_oov_advantage)
        & (frame.whisper_char_lm_advantage > 0.0)
    )
    return catastrophic | lexical


def apply_rule(frame: pd.DataFrame, config: RuleConfig, output: str) -> pd.Series:
    choose_whisper = whisper_mask(frame, config)
    values = np.where(choose_whisper, frame.whisper, frame.w2v)
    return pd.Series(values, index=frame.index, name=output)


def all_configs() -> list[RuleConfig]:
    return [
        RuleConfig(edit, oov)
        for edit in LEX_EDIT_GRID
        for oov in LEX_OOV_ADV_GRID
    ]


def select_config(frame: pd.DataFrame, indices: Sequence[int]) -> tuple[RuleConfig, list[dict[str, Any]]]:
    subset = frame.loc[list(indices)]
    ranked: list[tuple[tuple[float, int, float, float], RuleConfig, dict[str, Any]]] = []
    baseline = metric_for_column(subset, "w2v")
    for config in all_configs():
        hypotheses = apply_rule(subset, config, "candidate")
        candidate = metric(subset, hypotheses.tolist())
        switches = int((hypotheses != subset.w2v).sum())
        delta = float(candidate["zindi"] - baseline["zindi"])
        record = {
            "config": asdict(config),
            "n": len(subset),
            "switches": switches,
            "zindi": candidate["zindi"],
            "delta_vs_w2v": delta,
        }
        # Prefer higher score, then fewer switches, then the stricter rule.
        key = (float(candidate["zindi"]), -switches, -config.max_normalized_edit, config.min_w2v_oov_advantage)
        ranked.append((key, config, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1], [item[2] for item in ranked]


def nested_group_oof(frame: pd.DataFrame) -> tuple[pd.Series, list[dict[str, Any]]]:
    groups = frame.speaker_id.astype(str).to_numpy()
    if len(set(groups)) < OUTER_FOLDS:
        raise RuntimeError("insufficient speakers for five-fold grouped evaluation")
    splitter = GroupKFold(n_splits=OUTER_FOLDS)
    output = pd.Series(index=frame.index, dtype=object, name="nested_oof_router")
    folds: list[dict[str, Any]] = []
    for fold_index, (train_pos, test_pos) in enumerate(splitter.split(frame, groups=groups)):
        train_indices = frame.index[train_pos]
        test_indices = frame.index[test_pos]
        train_speakers = set(frame.loc[train_indices, "speaker_id"])
        test_speakers = set(frame.loc[test_indices, "speaker_id"])
        if train_speakers & test_speakers:
            raise RuntimeError(f"speaker leakage in outer fold {fold_index}")
        config, _ = select_config(frame, train_indices)
        output.loc[test_indices] = apply_rule(frame.loc[test_indices], config, "candidate")
        train_part = frame.loc[train_indices].copy()
        train_part["router"] = apply_rule(train_part, config, "router")
        test_part = frame.loc[test_indices].copy()
        test_part["router"] = output.loc[test_indices]
        folds.append(
            {
                "fold": fold_index,
                "config": asdict(config),
                "train_rows": len(train_part),
                "test_rows": len(test_part),
                "train_speakers": len(train_speakers),
                "test_speakers": len(test_speakers),
                "speaker_overlap": 0,
                "train": delta_metric(train_part, "router"),
                "test": delta_metric(test_part, "router"),
                "test_override_ids": test_part.loc[test_part.router != test_part.w2v, "ID"].tolist(),
            }
        )
    if output.isna().any():
        raise RuntimeError("nested grouped OOF did not predict every validation row")
    return output, folds


def hypothesis_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    words = process_words(reference, hypothesis)
    chars = process_characters(reference, hypothesis)
    return (
        int(words.substitutions + words.deletions + words.insertions),
        int(chars.substitutions + chars.deletions + chars.insertions),
    )


def oracle(frame: pd.DataFrame, candidates: Sequence[str], output: str) -> tuple[pd.Series, dict[str, Any]]:
    total_words = sum(len(value.split()) for value in frame.reference)
    total_chars = sum(len(value) for value in frame.reference)
    chosen: list[str] = []
    sources: list[str] = []
    for row in frame.itertuples(index=False):
        ranked = []
        for order, candidate in enumerate(candidates):
            word_errors, char_errors = hypothesis_error_counts(
                str(row.reference), str(getattr(row, candidate))
            )
            contribution = 0.5 * word_errors / total_words + 0.5 * char_errors / total_chars
            ranked.append((contribution, order, candidate, str(getattr(row, candidate))))
        _, _, source, text = min(ranked)
        chosen.append(text)
        sources.append(source)
    series = pd.Series(chosen, index=frame.index, name=output)
    return series, {
        "metric": metric(frame, chosen),
        "selection_counts": dict(Counter(sources)),
        "definition": "minimum additive official WER/CER error contribution using references; non-deployable",
    }


def bootstrap_rows(frame: pd.DataFrame, candidate: str) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(frame)
    scores = np.empty(BOOTSTRAP_DRAWS)
    deltas = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        positions = rng.integers(0, n, size=n)
        sample = frame.iloc[positions]
        base = metric_for_column(sample, "w2v")["zindi"]
        routed = metric_for_column(sample, candidate)["zindi"]
        scores[draw] = routed
        deltas[draw] = routed - base
    return summarize_bootstrap(scores, deltas, "paired row bootstrap")


def bootstrap_speakers(frame: pd.DataFrame, candidate: str) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    speakers = np.array(sorted(frame.speaker_id.unique()))
    scores = np.empty(BOOTSTRAP_DRAWS)
    deltas = np.empty(BOOTSTRAP_DRAWS)
    groups = {speaker: np.flatnonzero(frame.speaker_id.to_numpy() == speaker) for speaker in speakers}
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        positions = np.concatenate([groups[speaker] for speaker in sampled])
        sample = frame.iloc[positions]
        base = metric_for_column(sample, "w2v")["zindi"]
        routed = metric_for_column(sample, candidate)["zindi"]
        scores[draw] = routed
        deltas[draw] = routed - base
    return summarize_bootstrap(scores, deltas, "paired speaker-block bootstrap")


def summarize_bootstrap(scores: np.ndarray, deltas: np.ndarray, method: str) -> dict[str, Any]:
    return {
        "method": method,
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED if "row" in method else BOOTSTRAP_SEED + 1,
        "score_p05": float(np.quantile(scores, 0.05)),
        "score_p50": float(np.quantile(scores, 0.50)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "delta_mean": float(deltas.mean()),
        "delta_p05": float(np.quantile(deltas, 0.05)),
        "delta_p50": float(np.quantile(deltas, 0.50)),
        "delta_p95": float(np.quantile(deltas, 0.95)),
        "probability_delta_positive": float(np.mean(deltas > 0.0)),
        "probability_delta_nonnegative": float(np.mean(deltas >= 0.0)),
    }


def hash_half(speaker: str) -> str:
    value = hashlib.sha256((HASH_HALF_SEED + speaker).encode("utf-8")).digest()[0] % 2
    return f"speaker_hash_half_{value}"


def read_validation() -> tuple[pd.DataFrame, dict[str, Any], TrainLanguageModel]:
    manifest = strict_csv(
        VALIDATION_MANIFEST,
        ["ID", "language", "split", "sample_position", "fold", "reference"],
        "validation manifest",
    )
    w2v = strict_csv(
        VALIDATION_W2V,
        ["ID", "language", "split", "sample_position", "fold", "reference", "incumbent", "candidate"],
        "W2V validation",
    ).rename(columns={"candidate": "w2v"})
    whisper = strict_csv(
        VALIDATION_WHISPER,
        ["ID", "language", "split", "sample_position", "fold", "reference", "incumbent", "candidate"],
        "Whisper validation",
    ).rename(columns={"candidate": "whisper"})
    if len(manifest) != EXPECTED_VALIDATION_ROWS or manifest.ID.nunique() != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("validation manifest is not exact n=80")
    for label, candidate in (("W2V", w2v), ("Whisper", whisper)):
        if candidate.ID.tolist() != manifest.ID.tolist():
            raise RuntimeError(f"{label}: validation ID/order mismatch")
        for column in ("language", "split", "sample_position", "fold", "reference"):
            if candidate[column].tolist() != manifest[column].tolist():
                raise RuntimeError(f"{label}: {column} mismatch against manifest")
    if w2v.incumbent.tolist() != whisper.incumbent.tolist():
        raise RuntimeError("incumbent hypotheses differ between validation caches")
    if set(manifest.language) != {"lin"} or set(manifest.split) != {"validation"}:
        raise RuntimeError("non-Lingala or non-validation row entered exact sample")
    if manifest.fold.value_counts().to_dict() != {"tune": 40, "holdout": 40}:
        raise RuntimeError("immutable tune/holdout fold geometry changed")

    frame = manifest.copy()
    frame["incumbent"] = w2v.incumbent.map(normalize_or_dot)
    frame["w2v"] = w2v.w2v.map(normalize_or_dot)
    frame["whisper"] = whisper.whisper.map(normalize_or_dot)
    if frame[["reference", "incumbent", "w2v", "whisper"]].map(lambda value: not str(value).strip()).any().any():
        raise RuntimeError("empty validation reference or hypothesis")

    train_texts, metadata, data_audit = load_train_text_and_validation_metadata(set(frame.ID))
    frame = frame.merge(metadata, on="ID", validate="one_to_one")
    lm = TrainLanguageModel(train_texts)
    frame = add_reference_free_features(frame, lm)
    audit = {
        **data_audit,
        "validation_rows": len(frame),
        "validation_ids_sha256": sha256_lines(frame.ID),
        "validation_unique_speakers": int(frame.speaker_id.nunique()),
        "validation_gender_values": dict(Counter(frame.gender.astype(str))),
        "w2v_whisper_identical_rows": int((frame.w2v == frame.whisper).sum()),
        "w2v_incumbent_identical_rows": int((frame.w2v == frame.incumbent).sum()),
        "whisper_incumbent_identical_rows": int((frame.whisper == frame.incumbent).sum()),
        "train_lexicon_words": lm.total_words,
        "train_lexicon_types": lm.vocab_size,
    }
    return frame, audit, lm


def cache_status(path: Path, expected_ids: list[str], label: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    expected_set = set(expected_ids)
    if not path.is_file():
        return None, {
            "label": label,
            "path": str(path),
            "exists": False,
            "rows": 0,
            "complete_exact": False,
            "missing_ids": expected_ids,
            "foreign_ids": [],
        }
    frame = strict_csv(path, ["ID", "Target"], label)
    present = set(frame.ID)
    empty = frame.Target.map(lambda value: not normalize_text(value).strip())
    status = {
        "label": label,
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "rows": len(frame),
        "unique_ids": int(frame.ID.nunique()),
        "empty_targets": int(empty.sum()),
        "missing_ids": sorted(expected_set - present),
        "foreign_ids": sorted(present - expected_set),
        "ordered_exact": frame.ID.tolist() == expected_ids,
    }
    status["complete_exact"] = bool(
        len(frame) == len(expected_ids)
        and frame.ID.nunique() == len(expected_ids)
        and not empty.any()
        and not status["missing_ids"]
        and not status["foreign_ids"]
        and status["ordered_exact"]
    )
    return frame, status


def inspect_phase2_route() -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    route = strict_csv(
        ROUTE_INDEX,
        ["ID", "decode_lang", "split", "prediction"],
        "public-visible route index",
    )
    route = route[(route.decode_lang == "lin") & (route.split == "new")].copy()
    route = route.sort_values("ID", kind="stable").reset_index(drop=True)
    if len(route) != EXPECTED_PHASE2_ROWS or route.ID.nunique() != EXPECTED_PHASE2_ROWS:
        raise RuntimeError(f"expected exact 444-row Phase-2 Lingala route, got {len(route)}")
    if route.prediction.map(lambda value: not normalize_text(value).strip()).any():
        raise RuntimeError("empty incumbent prediction in Phase-2 Lingala route")
    expected_ids = route.ID.tolist()
    w2v, w2v_status = cache_status(PHASE2_W2V, expected_ids, "Phase-2 W2V cache")
    whisper, whisper_status = cache_status(PHASE2_WHISPER, expected_ids, "Phase-2 Whisper cache")
    presence = pd.DataFrame({"ID": expected_ids})
    presence["w2v_present"] = presence.ID.isin(set() if w2v is None else set(w2v.ID))
    presence["whisper_present"] = presence.ID.isin(set() if whisper is None else set(whisper.ID))
    presence.to_csv(OUT / "route_id_audit.csv", index=False)
    return route, w2v, whisper, {
        "expected_rows": EXPECTED_PHASE2_ROWS,
        "route_ids_sha256": sha256_lines(expected_ids),
        "route_index": str(ROUTE_INDEX),
        "route_index_sha256": sha256_file(ROUTE_INDEX),
        "w2v": w2v_status,
        "whisper": whisper_status,
        "both_caches_complete_exact": bool(
            w2v_status["complete_exact"] and whisper_status["complete_exact"]
        ),
        "route_id_audit": str(OUT / "route_id_audit.csv"),
    }


def build_fused_cache(
    route: pd.DataFrame,
    w2v: pd.DataFrame,
    whisper: pd.DataFrame,
    lm: TrainLanguageModel,
    config: RuleConfig,
) -> dict[str, Any]:
    frame = route[["ID", "prediction"]].rename(columns={"prediction": "incumbent"})
    frame = frame.merge(w2v.rename(columns={"Target": "w2v"}), on="ID", validate="one_to_one")
    frame = frame.merge(
        whisper.rename(columns={"Target": "whisper"}), on="ID", validate="one_to_one"
    )
    frame = add_reference_free_features(frame, lm)
    frame["choose_whisper"] = whisper_mask(frame, config)
    frame["source"] = np.where(frame.choose_whisper, "whisper", "w2v")
    frame["Target"] = np.where(frame.choose_whisper, frame.whisper, frame.w2v)
    if frame.ID.tolist() != route.ID.tolist() or frame.Target.map(lambda value: not str(value).strip()).any():
        raise RuntimeError("fused route cache failed ID/order/target invariants")
    output = OUT / "phase2_cache_lin_specialist_ensemble.csv"
    decisions = OUT / "phase2_router_decisions.csv"
    frame[["ID", "Target"]].to_csv(output, index=False)
    decision_columns = [
        "ID",
        "source",
        "choose_whisper",
        "w2v_words",
        "whisper_words",
        "w2v_singleton_token_rate",
        "w2v_oov_rate",
        "whisper_oov_rate",
        "w2v_whisper_normalized_edit",
        "whisper_w2v_word_ratio",
        "whisper_char_lm_advantage",
    ]
    frame[decision_columns].to_csv(decisions, index=False)
    return {
        "built": True,
        "cache": str(output),
        "cache_sha256": sha256_file(output),
        "rows": len(frame),
        "unique_ids": int(frame.ID.nunique()),
        "whisper_overrides": int(frame.choose_whisper.sum()),
        "w2v_rows": int((~frame.choose_whisper).sum()),
        "override_ids": frame.loc[frame.choose_whisper, "ID"].tolist(),
        "decisions": str(decisions),
        "submission_built": False,
    }


def main() -> None:
    global BOOTSTRAP_DRAWS
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        raise ValueError("--bootstrap-draws must be at least 100")
    BOOTSTRAP_DRAWS = args.bootstrap_draws
    OUT.mkdir(parents=True, exist_ok=True)

    frame, input_audit, lm = read_validation()
    nested_oof, outer_folds = nested_group_oof(frame)
    frame["nested_oof_router"] = nested_oof
    deployment_config, tuning_table = select_config(frame, frame.index)
    frame["deployable_router"] = apply_rule(frame, deployment_config, "deployable_router")
    frame["catastrophic_guard"] = catastrophic_mask(frame)
    frame["deployable_choose_whisper"] = frame.deployable_router != frame.w2v
    frame["nested_oof_choose_whisper"] = frame.nested_oof_router != frame.w2v

    oracle_two, oracle_two_report = oracle(frame, ["w2v", "whisper"], "oracle_w2v_whisper")
    oracle_three, oracle_three_report = oracle(
        frame, ["w2v", "whisper", "incumbent"], "oracle_three_way"
    )
    frame["oracle_w2v_whisper"] = oracle_two
    frame["oracle_three_way"] = oracle_three

    pure = {
        "w2vbert_lingala_sd3": metric_for_column(frame, "w2v"),
        "whisper_small_lingala_cased_2": metric_for_column(frame, "whisper"),
        "incumbent": metric_for_column(frame, "incumbent"),
    }
    nested_result = delta_metric(frame, "nested_oof_router")
    deployment_result = delta_metric(frame, "deployable_router")

    robust_halves: dict[str, Any] = {}
    for fold_name in ("tune", "holdout"):
        part = frame[frame.fold == fold_name]
        robust_halves[f"manifest_{fold_name}"] = delta_metric(part, "nested_oof_router")
    frame["speaker_hash_half"] = frame.speaker_id.map(hash_half)
    for half_name in sorted(frame.speaker_hash_half.unique()):
        part = frame[frame.speaker_hash_half == half_name]
        result = delta_metric(part, "nested_oof_router")
        result["speakers"] = int(part.speaker_id.nunique())
        robust_halves[half_name] = result

    row_bootstrap = bootstrap_rows(frame, "nested_oof_router")
    speaker_bootstrap = bootstrap_speakers(frame, "nested_oof_router")
    outer_nonnegative = all(
        fold["test"]["delta_zindi"] >= -1e-12 for fold in outer_folds
    )
    promotion_checks = {
        "nested_oof_strictly_beats_pure_w2v": nested_result["delta_zindi"] > 0.0,
        "manifest_tune_half_positive": robust_halves["manifest_tune"]["delta_zindi"] > 0.0,
        "manifest_holdout_half_positive": robust_halves["manifest_holdout"]["delta_zindi"] > 0.0,
        "speaker_hash_half_0_positive": robust_halves["speaker_hash_half_0"]["delta_zindi"] > 0.0,
        "speaker_hash_half_1_positive": robust_halves["speaker_hash_half_1"]["delta_zindi"] > 0.0,
        "no_negative_outer_speaker_fold": outer_nonnegative,
        "speaker_bootstrap_p05_positive": speaker_bootstrap["delta_p05"] > 0.0,
    }
    promoted = all(promotion_checks.values())

    tune_speakers = set(frame.loc[frame.fold == "tune", "speaker_id"])
    holdout_speakers = set(frame.loc[frame.fold == "holdout", "speaker_id"])
    catastrophic_rows = frame[frame.catastrophic_guard]
    anomalies = {
        "immutable_manifest_tune_holdout_speaker_overlap": {
            "count": len(tune_speakers & holdout_speakers),
            "speakers": sorted(tune_speakers & holdout_speakers),
            "note": "Promotion relies on speaker-disjoint outer OOF, not this overlapping split alone.",
        },
        "catastrophic_guard_rows": catastrophic_rows.ID.tolist(),
        "catastrophic_guard_unique_speakers": int(catastrophic_rows.speaker_id.nunique()),
        "catastrophic_guard_speakers": sorted(catastrophic_rows.speaker_id.unique()),
        "all_deployable_validation_override_ids": frame.loc[
            frame.deployable_choose_whisper, "ID"
        ].tolist(),
        "nested_oof_validation_override_ids": frame.loc[
            frame.nested_oof_choose_whisper, "ID"
        ].tolist(),
        "incumbent_selected_by_deployable_router": 0,
        "incumbent_note": (
            "The incumbent was evaluated and included in the three-way oracle. "
            "No stable reference-free incumbent selector beat the W2V/Whisper policy, "
            "so the promoted deployable rule conservatively never selects it."
        ),
        "model_card_provenance": (
            "The specialist cards do not disclose enough train/evaluation provenance to "
            "exclude validation exposure; matched local evaluation and Phase-2 transfer remain necessary."
        ),
    }

    # Save validation evidence before inspecting test prediction caches.
    validation_columns = [
        "ID",
        "speaker_id",
        "gender",
        "fold",
        "reference",
        "incumbent",
        "w2v",
        "whisper",
        "nested_oof_router",
        "deployable_router",
        "oracle_w2v_whisper",
        "oracle_three_way",
        "catastrophic_guard",
        "nested_oof_choose_whisper",
        "deployable_choose_whisper",
        "speaker_hash_half",
        "w2v_words",
        "whisper_words",
        "w2v_singleton_token_rate",
        "w2v_oov_rate",
        "whisper_oov_rate",
        "w2v_whisper_normalized_edit",
        "whisper_w2v_word_ratio",
        "whisper_char_lm_advantage",
    ]
    validation_path = OUT / "validation_router_audit.csv"
    frame[validation_columns].to_csv(validation_path, index=False)
    (OUT / "anomaly_audit.json").write_text(json.dumps(anomalies, indent=2) + "\n")

    route, w2v_cache, whisper_cache, route_audit = inspect_phase2_route()
    if promoted and route_audit["both_caches_complete_exact"]:
        assert w2v_cache is not None and whisper_cache is not None
        fused = build_fused_cache(route, w2v_cache, whisper_cache, lm, deployment_config)
    else:
        fused = {
            "built": False,
            "reason": (
                "promotion_failed"
                if not promoted
                else "one_or_both_444_row_specialist_caches_incomplete"
            ),
            "submission_built": False,
        }

    report = {
        "protocol": {
            "task": "reference-free Lingala specialist selective routing",
            "validation": "exact existing seed=42 n=80 google/WaxalNLP Lingala validation CSVs",
            "selection": (
                "five-fold speaker-disjoint outer OOF; fold-local threshold selection; "
                "fixed train-only-LM catastrophic guard"
            ),
            "features_at_deployment": [
                "hypothesis text agreement",
                "train-only word OOV rates",
                "train-only character trigram log-probability",
                "word-length ratios",
                "singleton-token collapse rate",
            ],
            "reference_used_as_deployment_feature": False,
            "test_labels_read": False,
            "submission_built": False,
        },
        "inputs": {
            **input_audit,
            "validation_manifest": str(VALIDATION_MANIFEST),
            "validation_manifest_sha256": sha256_file(VALIDATION_MANIFEST),
            "validation_w2v": str(VALIDATION_W2V),
            "validation_w2v_sha256": sha256_file(VALIDATION_W2V),
            "validation_whisper": str(VALIDATION_WHISPER),
            "validation_whisper_sha256": sha256_file(VALIDATION_WHISPER),
        },
        "metrics": {
            "pure": pure,
            "nested_speaker_oof_router": nested_result,
            "deployable_full_fit_router": deployment_result,
            "oracle_w2v_whisper": oracle_two_report,
            "oracle_w2v_whisper_incumbent": oracle_three_report,
            "robust_halves_nested_oof": robust_halves,
            "outer_speaker_folds": outer_folds,
            "bootstrap": {
                "rows": row_bootstrap,
                "speakers": speaker_bootstrap,
            },
        },
        "router": {
            "fixed_catastrophic_guard": {
                "w2v_max_words": CAT_MAX_WORDS,
                "w2v_min_singleton_token_rate": CAT_SINGLETON_MIN,
                "min_whisper_to_w2v_word_ratio": CAT_WHISPER_LENGTH_RATIO_MIN,
                "requires_positive_whisper_char_lm_advantage": True,
            },
            "deployment_config": asdict(deployment_config),
            "candidate_grid": {
                "normalized_edit": list(LEX_EDIT_GRID),
                "w2v_oov_advantage": list(LEX_OOV_ADV_GRID),
            },
            "full_data_tuning_table": tuning_table,
            "promotion_rule": (
                "nested speaker-OOF must strictly beat pure W2V; immutable tune/holdout "
                "and deterministic speaker-hash halves must each be positive; no outer "
                "speaker fold may be negative; speaker-block bootstrap p05 must be positive"
            ),
            "promotion_checks": promotion_checks,
            "promoted": promoted,
            "recommendation": (
                "promote_reference_free_router"
                if promoted
                else "retain_pure_w2vbert_lingala_sd3_as_primary"
            ),
        },
        "anomalies": anomalies,
        "phase2_route_audit": route_audit,
        "fused_route_cache": fused,
        "artifacts": {
            "validation_router_audit": str(validation_path),
            "anomaly_audit": str(OUT / "anomaly_audit.json"),
            "route_id_audit": str(OUT / "route_id_audit.csv"),
        },
    }
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "promoted": promoted,
                "pure_w2v": pure["w2vbert_lingala_sd3"],
                "pure_whisper": pure["whisper_small_lingala_cased_2"],
                "nested_speaker_oof_router": nested_result,
                "deployable_full_fit_router": deployment_result,
                "oracle_w2v_whisper": oracle_two_report,
                "oracle_three_way": oracle_three_report,
                "speaker_bootstrap": speaker_bootstrap,
                "deployment_config": asdict(deployment_config),
                "phase2_both_caches_complete": route_audit["both_caches_complete_exact"],
                "fused_route_cache": fused,
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
