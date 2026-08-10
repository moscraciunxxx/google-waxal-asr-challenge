#!/usr/bin/env python3
"""Leakage-safe Runyankole Sunbird/production ensemble evaluation.

This script is intentionally disjoint from the existing inference scripts.  It:

* reconstructs the seed-42 WAXAL Runyankole validation sample by exact ID;
* regenerates the production MMS-1B FT-v1 greedy and KenLM beam hypotheses;
* joins those hypotheses to the cached, correctly prompted Sunbird hypotheses;
* evaluates train-only lexical correction, sentence selection, word-lattice
  fusion, and a reference-free selector;
* evaluates learned/tuned policies with speaker-disjoint nested CV; and
* optionally applies only a CV-promoted policy to the current Phase-2 pair.

No test transcript is loaded or used.  All generated files are constrained to
``outputs/goal_2026_08_08/nyn_ensemble`` by default.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import kenlm
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pyctcdecode import build_ctcdecoder
from rapidfuzz.distance import Levenshtein
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from src.lexicon_correct import LexiconCorrector
from src.metrics import score_pairs
from src.submission import check_phase2_submission
from src.text_norm import normalize_text

LOGGER = logging.getLogger("nyn_sunbird_ensemble")

DEFAULT_OUT = ROOT / "outputs" / "goal_2026_08_08" / "nyn_ensemble"
DEFAULT_SUNBIRD_VAL = (
    ROOT
    / "outputs"
    / "goal_2026_08_08"
    / "sunbird51_nyn_seed42_n120"
    / "nyn_hyps.csv"
)
DEFAULT_SUNBIRD_PHASE2 = (
    ROOT / "outputs" / "goal_2026_08_08" / "sunbird51_phase2" / "hyps_nyn.csv"
)
DEFAULT_PHASE2_BASE = (
    ROOT
    / "outputs"
    / "goal_2026_08_07"
    / "badrex_tiers"
    / "submission_phase2_badrex_sna_sim99_lug_splitjoin.csv"
)
CURRENT_SUNBIRD_SUBMISSION = (
    ROOT
    / "outputs"
    / "goal_2026_08_08"
    / "sunbird51_phase2"
    / "submission_phase2_sim99_sunbird51_nyn.csv"
)
DEFAULT_FT = ROOT / "checkpoints" / "mms-nyn-ft-v1"
TRAIN_ARPA = ROOT / "data" / "lms" / "nyn_2gram.arpa"
DOMAIN_ARPA = ROOT / "data" / "lms_phase2_domain" / "nyn_merged_2gram.arpa"
TRAIN_UNIGRAMS = ROOT / "data" / "lms" / "nyn_unigrams.txt"
TRAIN_COUNTS = ROOT / "data" / "lms" / "nyn_counts.json"
SUNBIRD_ID = "Sunbird/asr-whisper-51-african-languages"
SUNBIRD_NYN_TOKEN = 50322


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cached_parquets(lang: str, split: str) -> list[Path]:
    """Find Hub snapshot parquet shards without creating a datasets cache lock."""
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    root = cache / "datasets--google--WaxalNLP" / "snapshots"
    pattern = f"*/data/ASR/{lang}/{lang}-{split}-*.parquet"
    paths = sorted(p.resolve() for p in root.glob(pattern) if p.resolve().is_file())
    if not paths:
        raise FileNotFoundError(
            f"No cached WAXAL parquet matched {root / pattern}. "
            "Download the dataset once before running this offline evaluator."
        )
    return paths


def load_validation_frame() -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in _cached_parquets("nyn", "validation")]
    frame = pd.concat(frames, ignore_index=True)
    required = {"id", "speaker_id", "transcription", "gender", "audio"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Runyankole validation parquet missing columns: {sorted(missing)}")
    frame["id"] = frame.id.astype(str)
    if frame.id.duplicated().any():
        raise ValueError("Duplicate Runyankole validation IDs")
    return frame


def decode_audio(encoded: dict) -> tuple[np.ndarray, int]:
    raw = encoded.get("bytes") if isinstance(encoded, dict) else None
    path = encoded.get("path") if isinstance(encoded, dict) else None
    if raw is not None:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    elif path:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    else:
        raise ValueError("Audio row has neither bytes nor a readable path")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=-1)
    audio = np.asarray(audio, dtype=np.float32)
    if int(sr) != TARGET_SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=int(sr), target_sr=TARGET_SR)
        sr = TARGET_SR
    peak = float(np.max(np.abs(audio)) + 1e-9)
    return audio / peak, int(sr)


def exact_seeded_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    indices = list(range(len(frame)))
    random.Random(seed).shuffle(indices)
    indices = indices[: min(n, len(indices))]
    return frame.iloc[indices].copy().reset_index(drop=True)


def load_exact_sunbird(path: Path, rows: pd.DataFrame) -> dict[str, str]:
    cached = pd.read_csv(path, dtype={"ID": str})
    if "hypothesis" not in cached or "reference" not in cached:
        raise ValueError(f"Sunbird cache lacks hypothesis/reference columns: {path}")
    if cached.ID.duplicated().any():
        raise ValueError(f"Duplicate IDs in Sunbird cache: {path}")
    expected = rows.id.astype(str).tolist()
    if set(cached.ID) != set(expected):
        missing = sorted(set(expected) - set(cached.ID))
        extra = sorted(set(cached.ID) - set(expected))
        raise ValueError(f"Sunbird exact-ID mismatch; missing={missing[:5]} extra={extra[:5]}")
    ref_by_id = dict(zip(cached.ID, cached.reference.map(normalize_text)))
    for row in rows.itertuples(index=False):
        ref = normalize_text(row.transcription)
        if ref_by_id[row.id] != ref:
            raise ValueError(f"Reference mismatch for {row.id}")
    return dict(zip(cached.ID, cached.hypothesis.map(lambda x: normalize_text(x) or ".")))


def _decoder(processor, arpa: Path, alpha: float, beta: float):
    vocab = processor.tokenizer.get_vocab()
    labels = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    unigrams = [line.strip() for line in TRAIN_UNIGRAMS.read_text().splitlines() if line.strip()]
    return build_ctcdecoder(
        labels,
        kenlm_model_path=str(arpa),
        unigrams=unigrams,
        alpha=alpha,
        beta=beta,
    )


def length_guard(greedy: str, beam: str, low: float = 0.5, high: float = 2.0) -> str:
    gw = max(1, len(greedy.split()))
    bw = max(1, len(beam.split()))
    return beam if beam.strip() and low <= bw / gw <= high else greedy


def confidence_stats(logits: np.ndarray, blank_id: int) -> dict[str, float]:
    tensor = torch.from_numpy(logits).float()
    log_probs = torch.log_softmax(tensor, dim=-1)
    top2 = torch.topk(log_probs, k=2, dim=-1).values
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(-1)
    denom = math.log(max(2, logits.shape[-1]))
    argmax = tensor.argmax(-1)
    return {
        "prod_mean_logp": float(top2[:, 0].mean()),
        "prod_mean_margin": float((top2[:, 0] - top2[:, 1]).mean()),
        "prod_mean_entropy": float((entropy / denom).mean()),
        "prod_blank_ratio": float((argmax == blank_id).float().mean()),
        "prod_frames": float(logits.shape[0]),
    }


@torch.inference_mode()
def decode_production(
    rows: pd.DataFrame,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    alpha: float,
    beta: float,
    beam_width: int,
    force: bool,
) -> pd.DataFrame:
    final_path = out_dir / "production_exact_id.csv"
    partial_path = out_dir / "production_exact_id.partial.csv"
    expected_ids = rows.id.astype(str).tolist()
    if final_path.exists() and not force:
        cached = pd.read_csv(final_path, dtype={"ID": str})
        if cached.ID.tolist() == expected_ids:
            LOGGER.info("Reusing exact-ID production cache %s", final_path)
            return cached

    done = pd.DataFrame()
    if partial_path.exists() and not force:
        done = pd.read_csv(partial_path, dtype={"ID": str})
        done = done[done.ID.isin(expected_ids)].drop_duplicates("ID", keep="last")
    by_id = {str(row.ID): row._asdict() for row in done.itertuples(index=False)}

    LOGGER.info("Loading production checkpoint %s on %s", checkpoint, device)
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(
        str(checkpoint), local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()
    train_decoder = _decoder(processor, TRAIN_ARPA, alpha, beta)
    domain_decoder = _decoder(processor, DOMAIN_ARPA, alpha, beta)
    blank_id = int(model.config.pad_token_id)

    started = time.time()
    for position, row in enumerate(rows.itertuples(index=False), start=1):
        uid = str(row.id)
        if uid in by_id:
            continue
        audio, sr = decode_audio(row.audio)
        inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        logits = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
        greedy = normalize_text(processor.decode(torch.tensor(logits.argmax(-1)))) or "."
        beam_train_raw = normalize_text(
            train_decoder.decode(logits, beam_width=beam_width).replace("|", " ")
        ) or "."
        beam_domain_raw = normalize_text(
            domain_decoder.decode(logits, beam_width=beam_width).replace("|", " ")
        ) or "."
        record = {
            "ID": uid,
            "production_greedy": greedy,
            "production_beam": length_guard(greedy, beam_train_raw),
            "production_domain_beam": length_guard(greedy, beam_domain_raw),
            "duration_sec": float(len(audio) / sr),
            **confidence_stats(logits, blank_id),
        }
        by_id[uid] = record
        ordered = pd.DataFrame([by_id[item] for item in expected_ids if item in by_id])
        ordered.to_csv(partial_path, index=False)
        completed = len(by_id)
        if completed % 10 == 0 or completed == len(rows):
            LOGGER.info(
                "production %d/%d (%.2fs/utterance)",
                completed,
                len(rows),
                (time.time() - started) / max(1, completed - len(done)),
            )

    result = pd.DataFrame([by_id[item] for item in expected_ids])
    result.to_csv(final_path, index=False)
    partial_path.unlink(missing_ok=True)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


@torch.inference_mode()
def decode_sunbird_if_requested(
    rows: pd.DataFrame, out_dir: Path, device: torch.device
) -> dict[str, str]:
    """Regenerate Sunbird exact-ID hypotheses when explicitly requested."""
    processor = WhisperProcessor.from_pretrained(SUNBIRD_ID, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        SUNBIRD_ID, local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()
    forced = [
        (1, SUNBIRD_NYN_TOKEN),
        (2, processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")),
        (3, processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")),
    ]
    output: dict[str, str] = {}
    for position, row in enumerate(rows.itertuples(index=False), start=1):
        audio, _ = decode_audio(row.audio)
        features = processor(
            audio, sampling_rate=TARGET_SR, do_normalize=True, return_tensors="pt"
        ).input_features.to(device)
        ids = model.generate(
            features,
            forced_decoder_ids=forced,
            num_beams=1,
            do_sample=False,
            max_new_tokens=256,
        )
        text = processor.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        output[str(row.id)] = normalize_text(text) or "."
        if position % 10 == 0 or position == len(rows):
            LOGGER.info("Sunbird %d/%d", position, len(rows))
    pd.DataFrame(
        {"ID": list(output), "hypothesis": list(output.values())}
    ).to_csv(out_dir / "sunbird_exact_id_regenerated.csv", index=False)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return output


def metric(refs: Sequence[str], hyps: Sequence[str]) -> dict[str, float | int]:
    score = score_pairs(list(refs), list(hyps))
    return {
        "n": len(refs),
        "wer": float(score["wer"]),
        "cer": float(score["cer"]),
        "zindi": float(1.0 - score["score"]),
    }


def row_errors(reference: str, hypothesis: str) -> tuple[int, int, int, int, float]:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    ref_chars = list(normalize_text(reference))
    hyp_chars = list(normalize_text(hypothesis))
    we = int(Levenshtein.distance(ref_words, hyp_words))
    ce = int(Levenshtein.distance(ref_chars, hyp_chars))
    nw = max(1, len(ref_words))
    nc = max(1, len(ref_chars))
    local = 0.5 * (we / nw + ce / nc)
    return we, ce, nw, nc, float(local)


class CountBigramLM:
    def __init__(self, path: Path, smoothing: float = 0.1):
        data = json.loads(path.read_text())
        self.uni = {str(k): int(v) for k, v in data["uni"].items()}
        self.bi = {str(k): int(v) for k, v in data["bi"].items()}
        self.vocab = set(self.uni) - {"<s>", "</s>"}
        self.vocab_size = max(2, len(self.vocab))
        self.smoothing = smoothing

    def transition(self, previous: str, word: str) -> float:
        count = self.bi.get(f"{previous}\t{word}", 0)
        denom = self.uni.get(previous, 0) + self.smoothing * self.vocab_size
        return math.log10((count + self.smoothing) / max(self.smoothing, denom))


def _align_words(first: Sequence[str], second: Sequence[str]) -> list[tuple[str | None, str | None]]:
    n, m = len(first), len(second)
    costs = np.zeros((n + 1, m + 1), dtype=np.int32)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    costs[:, 0] = np.arange(n + 1)
    costs[0, :] = np.arange(m + 1)
    back[1:, 0] = 1
    back[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = costs[i - 1, j - 1] + int(first[i - 1] != second[j - 1])
            delete = costs[i - 1, j] + 1
            insert = costs[i, j - 1] + 1
            options = (diag, delete, insert)
            choice = int(np.argmin(options))
            costs[i, j] = options[choice]
            back[i, j] = choice
    aligned: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i or j:
        move = int(back[i, j])
        if i and j and move == 0:
            aligned.append((first[i - 1], second[j - 1]))
            i -= 1
            j -= 1
        elif i and (j == 0 or move == 1):
            aligned.append((first[i - 1], None))
            i -= 1
        else:
            aligned.append((None, second[j - 1]))
            j -= 1
    return list(reversed(aligned))


def rover_fuse(
    sunbird: str,
    production: str,
    lm: CountBigramLM,
    lm_weight: float,
    sunbird_bias: float,
    deletion_penalty: float,
    beam_size: int = 32,
) -> str:
    slots = _align_words(sunbird.split(), production.split())
    # state: score, tokens, previous token
    beam: list[tuple[float, tuple[str, ...], str]] = [(0.0, (), "<s>")]
    for sun_word, prod_word in slots:
        choices: list[tuple[str | None, float]] = []
        if sun_word == prod_word:
            choices = [(sun_word, 0.0)]
        else:
            if sun_word is not None:
                choices.append((sun_word, sunbird_bias))
            else:
                choices.append((None, sunbird_bias + deletion_penalty))
            if prod_word is not None:
                choices.append((prod_word, 0.0))
            else:
                choices.append((None, deletion_penalty))
        expanded: dict[tuple[str, ...], tuple[float, str]] = {}
        for score, tokens, previous in beam:
            for word, source_score in choices:
                if word is None:
                    new_tokens = tokens
                    new_score = score + source_score
                    new_previous = previous
                else:
                    new_tokens = tokens + (word,)
                    new_score = score + source_score + lm_weight * lm.transition(previous, word)
                    new_previous = word
                old = expanded.get(new_tokens)
                if old is None or new_score > old[0]:
                    expanded[new_tokens] = (new_score, new_previous)
        ranked = sorted(
            ((score, tokens, previous) for tokens, (score, previous) in expanded.items()),
            key=lambda item: item[0],
            reverse=True,
        )
        beam = ranked[:beam_size]
    if not beam:
        return sunbird or production or "."
    return normalize_text(" ".join(beam[0][1])) or "."


def _lexical_stats(text: str, vocabulary: set[str]) -> dict[str, float]:
    words = normalize_text(text).split()
    chars = normalize_text(text)
    n_words = max(1, len(words))
    return {
        "words": float(len(words)),
        "chars": float(len(chars)),
        "oov": float(sum(word not in vocabulary for word in words) / n_words),
        "apostrophe": float(chars.count("'") / max(1, len(chars))),
        "unique": float(len(set(words)) / n_words),
        "repeat": float(sum(a == b for a, b in zip(words, words[1:])) / max(1, len(words) - 1)),
    }


def make_features(
    frame: pd.DataFrame,
    train_kenlm: kenlm.Model,
    domain_kenlm: kenlm.Model,
    vocabulary: set[str],
) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for row in frame.itertuples(index=False):
        sun = normalize_text(row.sunbird) or "."
        prod = normalize_text(row.production_beam) or "."
        ss = _lexical_stats(sun, vocabulary)
        ps = _lexical_stats(prod, vocabulary)
        sw, pw = sun.split(), prod.split()
        train_s = float(train_kenlm.score(sun, bos=True, eos=True) / max(1, len(sw)))
        train_p = float(train_kenlm.score(prod, bos=True, eos=True) / max(1, len(pw)))
        domain_s = float(domain_kenlm.score(sun, bos=True, eos=True) / max(1, len(sw)))
        domain_p = float(domain_kenlm.score(prod, bos=True, eos=True) / max(1, len(pw)))
        rec = {
            "sun_words": ss["words"],
            "prod_words": ps["words"],
            "log_word_ratio": math.log((ss["words"] + 1.0) / (ps["words"] + 1.0)),
            "log_char_ratio": math.log((ss["chars"] + 1.0) / (ps["chars"] + 1.0)),
            "word_disagreement": float(Levenshtein.normalized_distance(sw, pw)),
            "char_disagreement": float(Levenshtein.normalized_distance(sun, prod)),
            "sun_oov": ss["oov"],
            "prod_oov": ps["oov"],
            "oov_delta": ss["oov"] - ps["oov"],
            "sun_apostrophe": ss["apostrophe"],
            "prod_apostrophe": ps["apostrophe"],
            "sun_unique": ss["unique"],
            "prod_unique": ps["unique"],
            "sun_repeat": ss["repeat"],
            "prod_repeat": ps["repeat"],
            "train_lm_sun": train_s,
            "train_lm_prod": train_p,
            "train_lm_delta": train_s - train_p,
            "domain_lm_sun": domain_s,
            "domain_lm_prod": domain_p,
            "domain_lm_delta": domain_s - domain_p,
        }
        for name in (
            "duration_sec",
            "prod_mean_logp",
            "prod_mean_margin",
            "prod_mean_entropy",
            "prod_blank_ratio",
            "prod_frames",
        ):
            rec[name] = float(getattr(row, name, float("nan")))
        records.append(rec)
    features = pd.DataFrame(records)
    return features.replace([np.inf, -np.inf], np.nan)


BASIC_FEATURES = [
    "sun_words",
    "prod_words",
    "log_word_ratio",
    "log_char_ratio",
    "word_disagreement",
    "char_disagreement",
    "sun_oov",
    "prod_oov",
    "oov_delta",
    "sun_apostrophe",
    "prod_apostrophe",
    "sun_unique",
    "prod_unique",
    "sun_repeat",
    "prod_repeat",
]
LM_FEATURES = BASIC_FEATURES + [
    "train_lm_sun",
    "train_lm_prod",
    "train_lm_delta",
    "domain_lm_sun",
    "domain_lm_prod",
    "domain_lm_delta",
]
ALL_FEATURES = LM_FEATURES + [
    "duration_sec",
    "prod_mean_logp",
    "prod_mean_margin",
    "prod_mean_entropy",
    "prod_blank_ratio",
    "prod_frames",
]


def _impute(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(train), train, medians)
    test = np.where(np.isfinite(test), test, medians)
    return train, test


def _group_splits(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    folds = min(n_splits, len(unique))
    if folds < 2:
        raise ValueError("At least two distinct speakers are required for grouped CV")
    return list(GroupKFold(n_splits=folds).split(np.zeros(len(groups)), groups=groups))


@dataclass(frozen=True)
class SelectorConfig:
    feature_set: str
    c: float
    class_weight: str
    threshold: float

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_set": self.feature_set,
            "C": self.c,
            "class_weight": None if self.class_weight == "none" else self.class_weight,
            "threshold": self.threshold,
        }


def _columns(name: str) -> list[str]:
    return {"basic": BASIC_FEATURES, "lm": LM_FEATURES, "all": ALL_FEATURES}[name]


def _fit_probability(
    features: pd.DataFrame,
    labels: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    config: SelectorConfig,
) -> np.ndarray:
    cols = _columns(config.feature_set)
    x_train = features.iloc[train_index][cols].to_numpy(dtype=float)
    x_test = features.iloc[test_index][cols].to_numpy(dtype=float)
    x_train, x_test = _impute(x_train, x_test)
    y_train = labels[train_index]
    if len(np.unique(y_train)) < 2:
        return np.full(len(test_index), float(y_train[0]))
    class_weight = None if config.class_weight == "none" else config.class_weight
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.c,
            class_weight=class_weight,
            max_iter=2000,
            random_state=42,
        ),
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def _selector_grid(feature_sets: Iterable[str]) -> list[SelectorConfig]:
    return [
        SelectorConfig(feature_set, c, weight, threshold)
        for feature_set in feature_sets
        for c in (0.05, 0.2, 1.0, 5.0)
        for weight in ("none", "balanced")
        for threshold in (0.30, 0.40, 0.50, 0.60, 0.70)
    ]


def _hyp_from_probability(frame: pd.DataFrame, indices: np.ndarray, p: np.ndarray, threshold: float):
    choose_sun = p >= threshold
    sun = frame.iloc[indices].sunbird.to_numpy(dtype=str)
    prod = frame.iloc[indices].production_beam.to_numpy(dtype=str)
    return np.where(choose_sun, sun, prod), choose_sun


def tune_selector(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    labels: np.ndarray,
    indices: np.ndarray,
    groups: np.ndarray,
    feature_sets: Iterable[str],
    inner_splits: int = 4,
) -> tuple[SelectorConfig, dict[str, object]]:
    local_groups = groups[indices]
    splits = _group_splits(local_groups, inner_splits)
    best: tuple[float, SelectorConfig, np.ndarray] | None = None
    refs = frame.iloc[indices].reference.tolist()
    for config in _selector_grid(feature_sets):
        probabilities = np.zeros(len(indices), dtype=float)
        for train_local, valid_local in splits:
            train_global = indices[train_local]
            valid_global = indices[valid_local]
            base_config = SelectorConfig(config.feature_set, config.c, config.class_weight, 0.5)
            probabilities[valid_local] = _fit_probability(
                features, labels, train_global, valid_global, base_config
            )
        hyps, chosen = _hyp_from_probability(frame, indices, probabilities, config.threshold)
        zindi = float(metric(refs, hyps.tolist())["zindi"])
        candidate = (zindi, config, chosen)
        if best is None or zindi > best[0] + 1e-12:
            best = candidate
        elif best is not None and abs(zindi - best[0]) <= 1e-12:
            # Prefer simpler and more conservative configurations on ties.
            complexity = (config.feature_set == "all", config.feature_set == "lm", config.c)
            incumbent = (
                best[1].feature_set == "all",
                best[1].feature_set == "lm",
                best[1].c,
            )
            if complexity < incumbent:
                best = candidate
    assert best is not None
    return best[1], {
        "inner_oof_zindi": best[0],
        "inner_sunbird_fraction": float(best[2].mean()),
        "config": best[1].as_dict(),
    }


def nested_selector_cv(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    feature_sets: Iterable[str],
    outer_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    outer = _group_splits(groups, outer_splits)
    probabilities = np.zeros(len(frame), dtype=float)
    folds = np.full(len(frame), -1, dtype=int)
    details: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(outer):
        config, tuning = tune_selector(
            frame, features, labels, train, groups, feature_sets, inner_splits=4
        )
        probabilities[test] = _fit_probability(features, labels, train, test, config)
        folds[test] = fold
        held_hyp, chosen = _hyp_from_probability(
            frame, test, probabilities[test], config.threshold
        )
        train_speakers = set(groups[train])
        test_speakers = set(groups[test])
        overlap = train_speakers & test_speakers
        if overlap:
            raise AssertionError(f"Speaker leakage in fold {fold}: {sorted(overlap)[:3]}")
        details.append(
            {
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "train_speakers": len(train_speakers),
                "test_speakers": len(test_speakers),
                "speaker_overlap": 0,
                "heldout": metric(frame.iloc[test].reference.tolist(), held_hyp.tolist()),
                "heldout_sunbird_fraction": float(chosen.mean()),
                "tuning": tuning,
            }
        )
    return probabilities, folds, details


def cross_validated_candidate_tuning(
    frame: pd.DataFrame,
    candidate_columns: Sequence[str],
    groups: np.ndarray,
    outer_splits: int = 5,
) -> tuple[list[str], list[dict[str, object]]]:
    output = [""] * len(frame)
    details: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(_group_splits(groups, outer_splits)):
        refs_train = frame.iloc[train].reference.tolist()
        scored = [
            (float(metric(refs_train, frame.iloc[train][column].tolist())["zindi"]), column)
            for column in candidate_columns
        ]
        best_score, best_column = max(scored, key=lambda item: (item[0], -candidate_columns.index(item[1])))
        for index in test:
            output[index] = str(frame.iloc[index][best_column])
        details.append(
            {
                "fold": fold,
                "selected": best_column,
                "train_zindi": best_score,
                "heldout": metric(frame.iloc[test].reference.tolist(), frame.iloc[test][best_column].tolist()),
                "speaker_overlap": len(set(groups[train]) & set(groups[test])),
            }
        )
    if any(not item for item in output):
        raise AssertionError("Cross-validated candidate tuning left rows unassigned")
    return output, details


def fit_final_selector(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[SelectorConfig, object]:
    indices = np.arange(len(frame))
    config, _ = tune_selector(frame, features, labels, indices, groups, ["basic", "lm"])
    cols = _columns(config.feature_set)
    x = features[cols].to_numpy(dtype=float)
    x, _ = _impute(x, x.copy())
    class_weight = None if config.class_weight == "none" else config.class_weight
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.c,
            class_weight=class_weight,
            max_iter=2000,
            random_state=42,
        ),
    )
    model.fit(x, labels)
    return config, (model, np.nanmedian(features[cols].to_numpy(dtype=float), axis=0), cols)


def apply_final_selector(bundle, config: SelectorConfig, features: pd.DataFrame) -> np.ndarray:
    model, medians, cols = bundle
    x = features[cols].to_numpy(dtype=float)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x = np.where(np.isfinite(x), x, medians)
    return model.predict_proba(x)[:, 1]


def bootstrap_delta_by_speaker(
    frame: pd.DataFrame,
    challenger: Sequence[str],
    baseline: Sequence[str],
    seed: int = 42,
    repeats: int = 1000,
) -> dict[str, float]:
    groups = frame.speaker_id.astype(str).to_numpy()
    speakers = np.unique(groups)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    challenger = np.asarray(challenger, dtype=str)
    baseline = np.asarray(baseline, dtype=str)
    for _ in range(repeats):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == speaker) for speaker in sampled])
        refs = frame.iloc[indices].reference.tolist()
        c = float(metric(refs, challenger[indices].tolist())["zindi"])
        b = float(metric(refs, baseline[indices].tolist())["zindi"])
        deltas.append(c - b)
    return {
        "mean": float(np.mean(deltas)),
        "p2_5": float(np.quantile(deltas, 0.025)),
        "p50": float(np.quantile(deltas, 0.5)),
        "p97_5": float(np.quantile(deltas, 0.975)),
        "p_delta_gt_0": float(np.mean(np.asarray(deltas) > 0)),
        "p_delta_ge_0": float(np.mean(np.asarray(deltas) >= 0)),
    }


def override_anomaly_reasons(
    sunbird: str,
    production: str,
    sun_oov: float,
    production_oov: float,
    char_disagreement: float,
) -> list[str]:
    """Reference-free fail-closed checks for a production override.

    The localized expansion rule is deliberately conditional on low global
    disagreement.  It catches a small hallucinated phrase inside an otherwise
    matching sentence without rejecting cases where production legitimately
    restores a long span omitted by Sunbird.
    """
    sun_words = normalize_text(sunbird).split()
    prod_words = normalize_text(production).split()
    reasons: list[str] = []
    if prod_words and len(prod_words[-1]) <= 1:
        reasons.append("dangling_one_character_terminal")
    if (
        len(prod_words) >= 2
        and prod_words[-1] != prod_words[-2]
        and (
            prod_words[-2].startswith(prod_words[-1])
            or prod_words[-1].startswith(prod_words[-2])
        )
    ):
        reasons.append("trailing_prefix_fragment")
    if char_disagreement <= 0.20:
        operations = difflib.SequenceMatcher(
            a=sun_words, b=prod_words, autojunk=False
        ).get_opcodes()
        if any(
            tag == "replace" and i2 - i1 == 1 and j2 - j1 >= 3
            for tag, i1, i2, j1, j2 in operations
        ):
            reasons.append("localized_one_to_three_expansion")
    if production_oov > sun_oov + 0.10:
        reasons.append("production_oov_blowup_gt_0.10")
    return reasons


def anomaly_guard(frame: pd.DataFrame, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    reasons = [
        override_anomaly_reasons(
            row.sunbird,
            row.production_beam,
            float(features.iloc[index].sun_oov),
            float(features.iloc[index].prod_oov),
            float(features.iloc[index].char_disagreement),
        )
        for index, row in enumerate(frame.itertuples(index=False))
    ]
    return np.asarray([not item for item in reasons], dtype=bool), [";".join(item) for item in reasons]


def assemble_phase2(
    args,
    validation: pd.DataFrame,
    validation_features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    train_kenlm: kenlm.Model,
    domain_kenlm: kenlm.Model,
    vocabulary: set[str],
    nested_metric: dict[str, object],
    nested_probabilities: np.ndarray,
    validation_guard_pass: np.ndarray,
    out_dir: Path,
) -> dict[str, object]:
    if not (args.phase2_base.exists() and args.phase2_sunbird.exists()):
        return {"status": "skipped", "reason": "missing Phase-2 source files"}
    base = pd.read_csv(args.phase2_base, dtype={"ID": str})
    sun = pd.read_csv(args.phase2_sunbird, dtype={"ID": str})
    if "prediction" not in sun:
        raise ValueError(f"Phase-2 Sunbird cache lacks prediction: {args.phase2_sunbird}")
    if base.ID.duplicated().any() or sun.ID.duplicated().any():
        raise ValueError("Duplicate ID in Phase-2 source")
    base_map = base.set_index("ID").Target.map(lambda x: normalize_text(x) or ".")
    sun_map = sun.set_index("ID").prediction.map(lambda x: normalize_text(x) or ".")
    ids = [uid for uid in sun.ID.astype(str) if uid in base_map]
    phase = pd.DataFrame(
        {
            "ID": ids,
            "production_beam": [base_map[uid] for uid in ids],
            "sunbird": [sun_map[uid] for uid in ids],
        }
    )
    phase_features = make_features(phase, train_kenlm, domain_kenlm, vocabulary)
    config, bundle = fit_final_selector(
        validation, validation_features, labels, groups
    )
    probabilities = apply_final_selector(bundle, config, phase_features)
    phase_guard_pass, phase_guard_reasons = anomaly_guard(phase, phase_features)

    all_sunbird_zindi = float(nested_metric["all_sunbird_zindi"])
    selector_zindi = float(nested_metric["selector_zindi"])
    score_pass = selector_zindi > all_sunbird_zindi + args.min_cv_promotion
    uncertainty_pass = (
        float(nested_metric["bootstrap_p_delta_gt_0"]) >= args.min_bootstrap_probability
    )
    # Selector probability is P(Sunbird), so production is the low-probability branch.
    unguarded_production = probabilities < config.threshold
    guarded_production = unguarded_production & phase_guard_pass
    selected = np.where(guarded_production, phase.production_beam, phase.sunbird)
    selected_source = np.where(guarded_production, "production_beam", "sunbird")
    novel_rows = int(np.sum(selected != phase.sunbird.to_numpy(dtype=str)))
    production_fraction = novel_rows / max(1, len(phase))
    matched_k = max(1, int(round(len(validation) * production_fraction)))
    eligible = np.flatnonzero(validation_guard_pass)
    matched_indices = eligible[np.argsort(nested_probabilities[eligible])[:matched_k]]
    matched_hyp = validation.sunbird.to_numpy(dtype=str).copy()
    matched_hyp[matched_indices] = validation.production_beam.to_numpy(dtype=str)[matched_indices]
    matched_metric = metric(validation.reference.tolist(), matched_hyp.tolist())
    matched_delta = float(matched_metric["zindi"] - all_sunbird_zindi)
    matched_bootstrap = bootstrap_delta_by_speaker(
        validation,
        matched_hyp,
        validation.sunbird.to_numpy(dtype=str),
        seed=args.seed,
        repeats=args.bootstrap,
    )
    geometry_pass = bool(
        matched_delta > args.min_cv_promotion
        and matched_bootstrap["p_delta_ge_0"] >= args.min_bootstrap_probability
    )
    promoted = bool(score_pass and uncertainty_pass and geometry_pass and novel_rows > 0)
    deployed = selected
    deployed_source = selected_source
    detail = phase.copy()
    detail["sunbird_probability"] = probabilities
    detail["unguarded_selector_source"] = np.where(
        unguarded_production, "production_beam", "sunbird"
    )
    detail["anomaly_guard_pass"] = phase_guard_pass
    detail["anomaly_guard_reasons"] = phase_guard_reasons
    detail["selector_source"] = selected_source
    detail["deployed_source"] = deployed_source
    detail["deployed_prediction"] = deployed
    detail.to_csv(out_dir / "phase2_policy_detail.csv", index=False)

    unguarded_output = out_dir / "submission_phase2_nyn_cv_ensemble.csv"
    unguarded_output.unlink(missing_ok=True)
    output = out_dir / "submission_phase2_nyn_cv_ensemble_guarded.csv"
    if not promoted:
        output.unlink(missing_ok=True)
        return {
            "status": "rejected_no_submission_written",
            "source_rows": len(phase),
            "promoted_selector": False,
            "score_margin_pass": score_pass,
            "uncertainty_pass": uncertainty_pass,
            "geometry_pass": geometry_pass,
            "novel_rows_vs_current_sunbird": novel_rows,
            "promotion_margin_required": args.min_cv_promotion,
            "bootstrap_probability_required": args.min_bootstrap_probability,
            "selector_config": config.as_dict(),
            "unguarded_production_rows": int(unguarded_production.sum()),
            "guarded_production_rows": int(guarded_production.sum()),
            "guard_rejected_ids": phase.loc[unguarded_production & ~phase_guard_pass, "ID"].tolist(),
            "density_matched_validation": {
                "phase2_production_fraction": production_fraction,
                "validation_production_rows": matched_k,
                "metrics": matched_metric,
                "delta_vs_current_sunbird": matched_delta,
                "bootstrap": matched_bootstrap,
            },
        }

    candidate = base.copy()
    deployed_map = dict(zip(ids, deployed))
    candidate.loc[candidate.ID.isin(deployed_map), "Target"] = candidate.loc[
        candidate.ID.isin(deployed_map), "ID"
    ].map(deployed_map)
    candidate.to_csv(output, index=False)
    validation_result = check_phase2_submission(output, strict=True)
    if not validation_result["ok"]:
        raise RuntimeError(f"Phase-2 ensemble candidate invalid: {validation_result['errors']}")
    current_sunbird_changes = None
    if CURRENT_SUNBIRD_SUBMISSION.exists():
        incumbent = pd.read_csv(CURRENT_SUNBIRD_SUBMISSION, dtype={"ID": str}).set_index("ID")
        candidate_by_id = candidate.set_index("ID").Target.astype(str)
        current_sunbird_changes = int(
            sum(
                normalize_text(candidate_by_id[uid]) != normalize_text(incumbent.loc[uid, "Target"])
                for uid in ids
            )
        )
    return {
        "status": "written",
        "output": str(output),
        "output_sha256": sha256(output),
        "source_rows": len(phase),
        "promoted_selector": promoted,
        "score_margin_pass": score_pass,
        "uncertainty_pass": uncertainty_pass,
        "geometry_pass": geometry_pass,
        "promotion_margin_required": args.min_cv_promotion,
        "bootstrap_probability_required": args.min_bootstrap_probability,
        "selector_config": config.as_dict(),
        "unguarded_production_rows": int(unguarded_production.sum()),
        "guarded_production_rows": int(guarded_production.sum()),
        "guard_rejected": [
            {"ID": str(phase.iloc[index].ID), "reasons": phase_guard_reasons[index]}
            for index in np.flatnonzero(unguarded_production & ~phase_guard_pass)
        ],
        "surviving_override_ids": phase.loc[guarded_production, "ID"].astype(str).tolist(),
        "deployed_sunbird_rows": int(np.sum(np.char.startswith(deployed_source.astype(str), "sunbird"))),
        "novel_rows_vs_current_sunbird": current_sunbird_changes,
        "density_matched_validation": {
            "phase2_production_fraction": production_fraction,
            "validation_production_rows": matched_k,
            "metrics": matched_metric,
            "delta_vs_current_sunbird": matched_delta,
            "bootstrap": matched_bootstrap,
        },
        "changed_vs_base": int(
            sum(normalize_text(deployed_map[uid]) != normalize_text(base_map[uid]) for uid in ids)
        ),
        "validation": validation_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_FT)
    parser.add_argument("--sunbird-cache", type=Path, default=DEFAULT_SUNBIRD_VAL)
    parser.add_argument("--force-production", action="store_true")
    parser.add_argument("--regenerate-sunbird", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--min-cv-promotion", type=float, default=0.001)
    parser.add_argument("--min-bootstrap-probability", type=float, default=0.95)
    parser.add_argument("--phase2-base", type=Path, default=DEFAULT_PHASE2_BASE)
    parser.add_argument("--phase2-sunbird", type=Path, default=DEFAULT_SUNBIRD_PHASE2)
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = args.out_dir.resolve()
    expected_root = DEFAULT_OUT.resolve()
    if out_dir != expected_root and expected_root not in out_dir.parents:
        raise ValueError(f"Output must remain under {expected_root}; got {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    full = load_validation_frame()
    rows = exact_seeded_rows(full, args.n, args.seed)
    if args.regenerate_sunbird:
        sunbird = decode_sunbird_if_requested(rows, out_dir, device)
    else:
        sunbird = load_exact_sunbird(args.sunbird_cache, rows)
    production = decode_production(
        rows,
        args.checkpoint,
        out_dir,
        device,
        args.alpha,
        args.beta,
        args.beam_width,
        args.force_production,
    )

    frame = pd.DataFrame(
        {
            "ID": rows.id.astype(str),
            "speaker_id": rows.speaker_id.astype(str),
            "gender": rows.gender.astype(str),
            "reference": rows.transcription.map(lambda x: normalize_text(x) or "."),
            "sunbird": [sunbird[uid] for uid in rows.id.astype(str)],
        }
    ).merge(production, on="ID", how="left", validate="one_to_one")
    if frame.isna().any().any():
        raise ValueError(f"Null in exact-ID joined frame: {frame.isna().sum().to_dict()}")

    counts = json.loads(TRAIN_COUNTS.read_text())["uni"]
    vocabulary = set(counts) - {"<s>", "</s>"}
    corrector = LexiconCorrector({word: int(count) for word, count in counts.items() if word in vocabulary})
    frame["sunbird_lexicon"] = frame.sunbird.map(corrector.correct_text)
    frame["production_lexicon"] = frame.production_beam.map(corrector.correct_text)

    train_kenlm = kenlm.Model(str(TRAIN_ARPA))
    domain_kenlm = kenlm.Model(str(DOMAIN_ARPA))
    features = make_features(frame, train_kenlm, domain_kenlm, vocabulary)
    for column in features:
        frame[f"feature_{column}"] = features[column]

    frame["lm_sentence_select"] = np.where(
        features.train_lm_delta.to_numpy() >= 0,
        frame.sunbird.to_numpy(dtype=str),
        frame.production_beam.to_numpy(dtype=str),
    )
    for low, high in ((0.70, 1.40), (0.80, 1.25), (0.85, 1.40)):
        ratio = (features.sun_words + 1.0) / (features.prod_words + 1.0)
        column = f"ratio_guard_{str(low).replace('.', 'p')}_{str(high).replace('.', 'p')}"
        frame[column] = np.where(
            (ratio >= low) & (ratio <= high), frame.sunbird, frame.production_beam
        )

    count_lm = CountBigramLM(TRAIN_COUNTS)
    rover_columns: list[str] = []
    for lm_weight in (0.0, 0.15, 0.30):
        for sun_bias in (-0.30, 0.0, 0.30, 0.60):
            for deletion_penalty in (-0.5, -1.0, -2.0):
                name = (
                    f"rover_lm{lm_weight:+.2f}_sun{sun_bias:+.2f}_del{deletion_penalty:+.1f}"
                    .replace("+", "p")
                    .replace("-", "m")
                    .replace(".", "p")
                )
                frame[name] = [
                    rover_fuse(sun, prod, count_lm, lm_weight, sun_bias, deletion_penalty)
                    for sun, prod in zip(frame.sunbird, frame.production_beam)
                ]
                rover_columns.append(name)

    refs = frame.reference.tolist()
    baseline_columns = [
        "production_greedy",
        "production_beam",
        "production_domain_beam",
        "sunbird",
        "production_lexicon",
        "sunbird_lexicon",
        "lm_sentence_select",
        "ratio_guard_0p7_1p4",
        "ratio_guard_0p8_1p25",
        "ratio_guard_0p85_1p4",
    ]
    metrics = {column: metric(refs, frame[column].tolist()) for column in baseline_columns}

    sun_local, prod_local = [], []
    for ref, sun, prod in zip(frame.reference, frame.sunbird, frame.production_beam):
        sun_local.append(row_errors(ref, sun)[-1])
        prod_local.append(row_errors(ref, prod)[-1])
    labels = (np.asarray(sun_local) < np.asarray(prod_local)).astype(int)
    groups = frame.speaker_id.astype(str).to_numpy()

    probabilities, folds, selector_folds = nested_selector_cv(
        frame,
        features,
        labels,
        groups,
        feature_sets=["basic", "lm"],
        outer_splits=args.outer_folds,
    )
    # Each outer fold selected its own threshold; recover held-out decisions.
    unguarded_selector_choose = np.zeros(len(frame), dtype=bool)
    for detail in selector_folds:
        fold = int(detail["fold"])
        threshold = float(detail["tuning"]["config"]["threshold"])
        unguarded_selector_choose[folds == fold] = probabilities[folds == fold] >= threshold
    validation_guard_pass, validation_guard_reasons = anomaly_guard(frame, features)
    validation_guard_rejected = (~unguarded_selector_choose) & (~validation_guard_pass)
    selector_choose = unguarded_selector_choose | (~validation_guard_pass)
    unguarded_selector_hyp = np.where(
        unguarded_selector_choose, frame.sunbird, frame.production_beam
    )
    selector_hyp = np.where(selector_choose, frame.sunbird, frame.production_beam)
    frame["nested_selector_probability"] = probabilities
    frame["outer_fold"] = folds
    frame["anomaly_guard_pass"] = validation_guard_pass
    frame["anomaly_guard_reasons"] = validation_guard_reasons
    frame["diagnostic_unguarded_nested_selector"] = unguarded_selector_hyp
    frame["nested_selector_source"] = np.where(selector_choose, "sunbird", "production_beam")
    frame["nested_selector_hypothesis"] = selector_hyp
    metrics["diagnostic_unguarded_nested_selector"] = metric(
        refs, unguarded_selector_hyp.tolist()
    )
    metrics["nested_speaker_cv_selector"] = metric(refs, selector_hyp.tolist())

    diagnostic_probabilities, diagnostic_folds, diagnostic_details = nested_selector_cv(
        frame,
        features,
        labels,
        groups,
        feature_sets=["basic", "lm", "all"],
        outer_splits=args.outer_folds,
    )
    diagnostic_choose = np.zeros(len(frame), dtype=bool)
    for detail in diagnostic_details:
        fold = int(detail["fold"])
        threshold = float(detail["tuning"]["config"]["threshold"])
        diagnostic_choose[diagnostic_folds == fold] = (
            diagnostic_probabilities[diagnostic_folds == fold] >= threshold
        )
    diagnostic_hyp = np.where(diagnostic_choose, frame.sunbird, frame.production_beam)
    frame["diagnostic_nested_selector_with_acoustic"] = diagnostic_hyp
    metrics["diagnostic_nested_selector_with_acoustic"] = metric(
        refs, diagnostic_hyp.tolist()
    )

    static_candidates = [
        "production_beam",
        "sunbird",
        "production_lexicon",
        "sunbird_lexicon",
        "lm_sentence_select",
        "ratio_guard_0p7_1p4",
        "ratio_guard_0p8_1p25",
        "ratio_guard_0p85_1p4",
        *rover_columns,
    ]
    tuned_hyp, static_folds = cross_validated_candidate_tuning(
        frame, static_candidates, groups, outer_splits=args.outer_folds
    )
    frame["speaker_cv_static_fusion"] = tuned_hyp
    metrics["speaker_cv_static_fusion"] = metric(refs, tuned_hyp)

    oracle = np.where(
        np.asarray(sun_local) < np.asarray(prod_local), frame.sunbird, frame.production_beam
    )
    frame["diagnostic_oracle"] = oracle
    metrics["diagnostic_oracle_not_deployable"] = metric(refs, oracle.tolist())

    exact_out = out_dir / "same_id_hypotheses_and_cv.csv"
    frame.to_csv(exact_out, index=False)

    selector_metric = metrics["nested_speaker_cv_selector"]
    sunbird_metric = metrics["sunbird"]
    nested_summary = {
        "selector_zindi": selector_metric["zindi"],
        "all_sunbird_zindi": sunbird_metric["zindi"],
        "delta_vs_all_sunbird": float(selector_metric["zindi"] - sunbird_metric["zindi"]),
    }
    confidence = bootstrap_delta_by_speaker(
        frame,
        selector_hyp,
        frame.sunbird.to_numpy(dtype=str),
        seed=args.seed,
        repeats=args.bootstrap,
    )
    nested_summary["bootstrap_p_delta_gt_0"] = confidence["p_delta_gt_0"]

    phase2 = {"status": "skipped", "reason": "--skip-phase2"}
    if not args.skip_phase2:
        phase2 = assemble_phase2(
            args,
            frame,
            features,
            labels,
            groups,
            train_kenlm,
            domain_kenlm,
            vocabulary,
            nested_summary,
            probabilities,
            validation_guard_pass,
            out_dir,
        )

    report = {
        "protocol": {
            "language": "nyn",
            "n": len(frame),
            "seed": args.seed,
            "device": str(device),
            "validation_source": [str(path) for path in _cached_parquets("nyn", "validation")],
            "exact_sunbird_cache": str(args.sunbird_cache),
            "production_checkpoint": str(args.checkpoint),
            "production_recipe": {
                "train_arpa": str(TRAIN_ARPA),
                "domain_arpa": str(DOMAIN_ARPA),
                "alpha": args.alpha,
                "beta": args.beta,
                "beam_width": args.beam_width,
                "length_guard": [0.5, 2.0],
            },
            "unique_speakers": int(frame.speaker_id.nunique()),
            "outer_folds": args.outer_folds,
            "speaker_overlap_every_fold": 0,
            "selector_features_reference_free": True,
            "lexicon_source": str(TRAIN_COUNTS),
            "test_transcripts_used": False,
            "public_geometry": {
                "nyn_public_visible_rows": 256,
                "public_sensitive_rows": 1607,
                "expanded_submission_rows": 2392,
                "nyn_share_of_public_sensitive_rows": 256 / 1607,
                "note": "Utterance share only; corpus WER/CER influence is token-weighted.",
            },
        },
        "metrics": metrics,
        "pairwise": {
            "sunbird_local_wins": int(np.sum(np.asarray(sun_local) < np.asarray(prod_local))),
            "production_local_wins": int(np.sum(np.asarray(prod_local) < np.asarray(sun_local))),
            "local_ties": int(np.sum(np.asarray(prod_local) == np.asarray(sun_local))),
        },
        "nested_selector": {
            **nested_summary,
            "guard": {
                "reference_free": True,
                "rules": {
                    "dangling_terminal_max_characters": 1,
                    "trailing_prefix_fragment": True,
                    "localized_one_to_three_expansion_max_global_char_disagreement": 0.20,
                    "production_oov_increase_max": 0.10,
                },
                "unguarded_metrics": metrics["diagnostic_unguarded_nested_selector"],
                "guarded_metrics": metrics["nested_speaker_cv_selector"],
                "heldout_production_overrides_rejected": int(validation_guard_rejected.sum()),
                "heldout_rejected_ids": frame.loc[validation_guard_rejected, "ID"].astype(str).tolist(),
                "speaker_disjoint_predictions_preserved": True,
            },
            "sunbird_rows": int(selector_choose.sum()),
            "production_rows": int((~selector_choose).sum()),
            "folds": selector_folds,
            "bootstrap_delta_vs_all_sunbird": confidence,
        },
        "diagnostic_selector_with_acoustic_features": {
            "metrics": metrics["diagnostic_nested_selector_with_acoustic"],
            "not_deployable_reason": "Phase-2 production CTC confidence was not cached",
            "folds": diagnostic_details,
        },
        "static_fusion_cv": {
            "metrics": metrics["speaker_cv_static_fusion"],
            "folds": static_folds,
        },
        "phase2": phase2,
        "candidate_ranking_vs_current_sunbird51": sorted(
            [
                {
                    "candidate": "current_sunbird51",
                    "zindi": metrics["sunbird"]["zindi"],
                    "delta": 0.0,
                    "novel": False,
                    "status": "incumbent",
                },
                {
                    "candidate": "nested_speaker_cv_selector_guarded_deployable",
                    "zindi": metrics["nested_speaker_cv_selector"]["zindi"],
                    "delta": metrics["nested_speaker_cv_selector"]["zindi"]
                    - metrics["sunbird"]["zindi"],
                    "novel": bool(phase2.get("novel_rows_vs_current_sunbird", 0)),
                    "status": "pass" if phase2.get("promoted_selector") else "reject",
                },
                {
                    "candidate": "existing_word_ratio_guard_0.85_1.40",
                    "zindi": metrics["ratio_guard_0p85_1p4"]["zindi"],
                    "delta": metrics["ratio_guard_0p85_1p4"]["zindi"]
                    - metrics["sunbird"]["zindi"],
                    "novel": False,
                    "status": "existing_non_novel",
                },
                {
                    "candidate": "speaker_cv_static_fusion",
                    "zindi": metrics["speaker_cv_static_fusion"]["zindi"],
                    "delta": metrics["speaker_cv_static_fusion"]["zindi"]
                    - metrics["sunbird"]["zindi"],
                    "novel": True,
                    "status": "reject",
                },
            ],
            key=lambda item: item["zindi"],
            reverse=True,
        ),
        "artifacts": {
            "exact_hypotheses_and_cv": str(exact_out),
            "exact_hypotheses_and_cv_sha256": sha256(exact_out),
            "production_cache": str(out_dir / "production_exact_id.csv"),
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
