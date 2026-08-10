#!/usr/bin/env python3
"""Exact-ID Luganda model/fusion evaluation on validation audio only.

The script deliberately separates three things that earlier experiments mixed:

* acoustic hypotheses, all decoded on the same fixed proxy IDs;
* reference provenance (original WAXAL versus Harcuracy's corrected labels);
* reference-free selection policies that can be applied to competition audio.

No test references are loaded or accepted.  Persistent writes are confined to
``outputs/goal_2026_08_08/luganda_fusion``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from pyctcdecode import build_ctcdecoder
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForCTC,
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

OUT = ROOT / "outputs" / "goal_2026_08_08" / "luganda_fusion"
PROXY = ROOT / "data" / "proxy_val_index.csv"
PRODUCTION_SUBMISSION = (
    ROOT
    / "outputs"
    / "goal_2026_08_07"
    / "badrex_tiers"
    / "submission_phase2_badrex_sna_sim99_lug_splitjoin.csv"
)
ROUTE_INDEX = ROOT / "outputs" / "beat075" / "public_visible_index.csv"
CORRECTED_PARQUET = (
    "https://huggingface.co/datasets/Harcuracy/"
    "google_waxal_asr_challenge/resolve/main/"
    "lug_asr/validation-00000-of-00001.parquet"
)
SUNBIRD_CACHE = (
    ROOT
    / "outputs"
    / "goal_2026_08_08"
    / "sunbird51_routes_full"
    / "lug_hyps.csv"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    path: str
    adapter: str | None = None
    note: str = ""


# Distinct local Luganda model states.  The three whisper-per-lang checkpoint
# directories are byte-identical, so only ``best`` is decoded and the inventory
# records the deduplication.
MODEL_SPECS = (
    ModelSpec(
        "okwija_mms_adapter_waxal_lug",
        "ctc_ambiguous_beam",
        "checkpoints/okwija-mms-adapter-waxal-lug",
        note="ID 28 is duplicated by | and ᵑ; decode protocol explicitly assigns it to the word delimiter",
    ),
    ModelSpec(
        "sulaimank_w2vbert_grain_lg_v2",
        "auto_ctc",
        "outputs/goal_2026_08_08/luganda_fusion/models/w2v-bert-grain-lg-v2",
        note="open Wav2Vec2-BERT Grain model; card WER .0299/CER .0077 on non-WAXAL Grain eval",
    ),
    ModelSpec("mms_ft_v2", "ctc", "checkpoints/mms-lug-ft-v2"),
    ModelSpec("mms_ft_v3", "ctc_beam", "checkpoints/mms-lug-ft-v3"),
    ModelSpec("mms_ft_v4", "ctc_beam", "checkpoints/mms-lug-ft-v4"),
    ModelSpec("mms_ft_v5b", "ctc", "checkpoints/mms-lug-ft-v5b"),
    ModelSpec("mms_ft_v5c", "ctc", "checkpoints/mms-lug-ft-v5c"),
    ModelSpec(
        "mms_ft_pseudo_public_v1",
        "ctc",
        "checkpoints/mms-lug-ft-pseudo-public-v1",
        note="HF train plus pseudo labels; validation remains label-free at inference",
    ),
    ModelSpec(
        "mms300_continue_legit",
        "ctc",
        "checkpoints/mms300-continue-legit/lug/best",
    ),
    ModelSpec(
        "nolimitsxl_base",
        "nolimits",
        "checkpoints/nolimitsxl-mms1b-waxal-base",
    ),
    ModelSpec(
        "nolimitsxl_lora",
        "nolimits",
        "checkpoints/nolimitsxl-mms1b-waxal-base",
        adapter="checkpoints/nolimitsxl-mms1b-waxal",
    ),
    ModelSpec(
        "whisper_per_lang_legit",
        "whisper",
        "checkpoints/whisper-per-lang-legit/lug/best",
        note="best, checkpoint-30 and checkpoint-100 are byte-identical",
    ),
    ModelSpec(
        "whisper_protocol_beat_mms",
        "whisper",
        "checkpoints/whisper-protocol-beat-mms/lug/best",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(refs: list[str], hyps: list[str]) -> dict[str, float | int]:
    score = score_pairs(refs, hyps)
    return {
        "n": int(score["n"]),
        "wer": float(score["wer"]),
        "cer": float(score["cer"]),
        "error": float(score["score"]),
        "zindi": float(1.0 - score["score"]),
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


def load_proxy_examples() -> tuple[list[str], dict[str, dict]]:
    proxy = pd.read_csv(PROXY, dtype=str).fillna("")
    lug = proxy.loc[proxy["language"] == "lug"].copy()
    ids = lug["id"].astype(str).tolist()
    if len(ids) != len(set(ids)) or not ids:
        raise RuntimeError("Luganda proxy IDs are empty or non-unique")

    dataset = load_hf_asr_split("lug", "validation")
    wanted = set(ids)
    by_id: dict[str, dict] = {}
    for position in range(len(dataset)):
        example = dataset[position]
        uid = str(example.get("id") or example.get("ID") or "")
        if uid in wanted:
            by_id[uid] = example
            wanted.remove(uid)
            if not wanted:
                break
    if wanted:
        raise RuntimeError(f"Missing proxy validation IDs: {sorted(wanted)[:5]}")
    return ids, by_id


def corrected_reference_map(ids: list[str], refresh: bool) -> dict[str, str]:
    """Read only id/transcription columns from the public corrected Parquet."""
    cache = OUT / "corrected_validation_labels.csv"
    if cache.exists() and not refresh:
        frame = pd.read_csv(cache, dtype=str).fillna("")
    else:
        # HTTPFile performs range requests; PyArrow projects away embedded audio.
        with fsspec.open(CORRECTED_PARQUET, "rb", block_size=1 << 20) as remote:
            table = pq.ParquetFile(remote).read(columns=["id", "transcription"])
        frame = table.to_pandas()
        frame.to_csv(cache, index=False)
    if frame["id"].duplicated().any():
        raise RuntimeError("Corrected validation labels contain duplicate IDs")
    mapping = {
        str(row.id): normalize_text(str(row.transcription)) or "."
        for row in frame.itertuples(index=False)
    }
    missing = [uid for uid in ids if uid not in mapping]
    if missing:
        raise RuntimeError(f"Corrected validation labels missing IDs: {missing[:5]}")
    return mapping


def load_splitjoin_counts() -> tuple[dict[str, int], dict[str, int]]:
    payload = json.loads((ROOT / "data" / "lms" / "lug_counts.json").read_text())
    unigrams = {
        str(word): int(count)
        for word, count in payload["uni"].items()
        if not str(word).startswith("<")
    }
    bigrams = {str(pair): int(count) for pair, count in payload["bi"].items()}
    return unigrams, bigrams


def apply_splitjoin(text: str, counts: tuple[dict[str, int], dict[str, int]]) -> str:
    return normalize_text(feat_D_join_lug_splits(text, *counts)) or "."


def decoder_for(
    processor,
    arpa: Path,
    alpha: float = 0.3,
    beta: float = 0.5,
    *,
    vocab_size: int | None = None,
    label_overrides: dict[int, str] | None = None,
):
    """Build labels by numeric ID, preserving deliberate duplicate-ID repair."""
    size = int(vocab_size or len(processor.tokenizer))
    labels = [str(processor.tokenizer.convert_ids_to_tokens(i)) for i in range(size)]
    for index, token in (label_overrides or {}).items():
        labels[int(index)] = token
    unigrams = [
        line.strip()
        for line in (ROOT / "data" / "lms" / "lug_unigrams.txt").read_text().splitlines()
        if line.strip()
    ]
    return build_ctcdecoder(
        labels,
        kenlm_model_path=str(arpa),
        unigrams=unigrams,
        alpha=alpha,
        beta=beta,
    )


def normalize_nolimits_tokenizer(processor) -> None:
    # The repository appends <pad>/<unk>, but the acoustic head uses [PAD]=49.
    processor.tokenizer.pad_token = "[PAD]"
    processor.tokenizer.unk_token = "[UNK]"
    processor.tokenizer.word_delimiter_token = "|"


def decode_ambiguous_delimiter_ids(tokenizer, token_ids: list[int], blank: int) -> str:
    """CTC collapse for Okwija, resolving duplicate ID 28 as a word boundary.

    AutoTokenizer's inverse vocabulary resolves ID 28 to ``ᵑ`` and therefore
    ``tokenizer.decode`` emits no spaces.  The model's tokenizer metadata names
    ID 28 as ``word_delimiter_token_id`` and its encoder maps spaces to ID 28,
    which is the decisive direction of the mapping for CTC inference.
    """
    output: list[str] = []
    previous: int | None = None
    ignored = {int(blank), 0, 29, 31, 32}
    for value in token_ids:
        index = int(value)
        if index == previous:
            continue
        previous = index
        if index in ignored:
            continue
        if index == 28:
            output.append(" ")
        else:
            output.append(str(tokenizer.convert_ids_to_tokens(index)))
    return normalize_text("".join(output)) or "."


@torch.inference_mode()
def decode_ctc(
    spec: ModelSpec,
    ids: list[str],
    examples: dict[str, dict],
    device: torch.device,
) -> pd.DataFrame:
    path = ROOT / spec.path
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    ambiguous_delimiter = spec.kind == "ctc_ambiguous_beam"
    if spec.kind == "nolimits":
        normalize_nolimits_tokenizer(processor)
    elif spec.kind == "auto_ctc":
        tokenizer = processor.tokenizer
        delimiter = int(tokenizer.word_delimiter_token_id)
        if str(tokenizer.convert_ids_to_tokens(delimiter)) != "|":
            raise RuntimeError(
                f"{spec.name}: invalid word delimiter mapping at ID {delimiter}"
            )
    elif ambiguous_delimiter:
        tokenizer = processor.tokenizer
        if int(tokenizer.word_delimiter_token_id) != 28:
            raise RuntimeError("Okwija word delimiter is not ID 28")
        if int(tokenizer.pad_token_id) != 30:
            raise RuntimeError("Okwija tokenizer blank/pad is not ID 30")
        duplicate_28 = sorted(
            token for token, index in tokenizer.get_vocab().items() if int(index) == 28
        )
        if duplicate_28 != ["|", "ᵑ"]:
            raise RuntimeError(f"Unexpected Okwija ID-28 aliases: {duplicate_28}")
        probe = decode_ambiguous_delimiter_ids(
            tokenizer, [2, 2, 30, 28, 28, 3], blank=30
        )
        if probe != "a b":
            raise RuntimeError(f"Okwija delimiter repair failed probe: {probe!r}")
    else:
        fix_mms_tokenizer(processor, "lug")
    model_class = AutoModelForCTC if spec.kind == "auto_ctc" else Wav2Vec2ForCTC
    model = model_class.from_pretrained(
        str(path), local_files_only=True, low_cpu_mem_usage=True
    )
    if spec.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(ROOT / spec.adapter),
            local_files_only=True,
            is_trainable=False,
        )
    model.to(device).eval()

    domain_decoder = None
    standard_decoder = None
    if spec.kind in {"ctc_beam", "ctc_ambiguous_beam"}:
        # ID 0 is an unassigned hole which AutoTokenizer aliases to [UNK],
        # duplicating the real [UNK] at ID 29.  It must keep its logit column
        # but needs a unique inert label for pyctcdecode's alphabet invariant.
        overrides = {0: "<unused_0>", 28: "|"} if ambiguous_delimiter else None
        domain_decoder = decoder_for(
            processor,
            ROOT / "data" / "lms_phase2_domain" / "lug_merged_2gram.arpa",
            vocab_size=int(model.config.vocab_size),
            label_overrides=overrides,
        )
        standard_decoder = decoder_for(
            processor,
            ROOT / "data" / "lms" / "lug_2gram.arpa",
            vocab_size=int(model.config.vocab_size),
            label_overrides=overrides,
        )

    rows = []
    started = time.time()
    for position, uid in enumerate(ids, start=1):
        array = prepare_audio(examples[uid])
        batch = processor(
            array, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        )
        kwargs = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        logits = model(**kwargs).logits[0]
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        ids_greedy = torch.argmax(logits, dim=-1)
        # MMS target alphabets store several ordinary graphemes (for example
        # doubled vowels/consonants) as added tokens marked ``special`` by the
        # tokenizer.  ``skip_special_tokens=True`` silently deletes them and
        # corrupts CER, so CTC decoding must retain the full target alphabet.
        raw_processor_decode = normalize_text(
            processor.decode(ids_greedy)
        ) or "."
        if ambiguous_delimiter:
            greedy = decode_ambiguous_delimiter_ids(
                processor.tokenizer,
                ids_greedy.detach().cpu().tolist(),
                blank=int(model.config.pad_token_id),
            )
        else:
            greedy = raw_processor_decode
        blank = int(processor.tokenizer.pad_token_id or 0)
        nonblank = ids_greedy != blank
        selected = log_probs[
            torch.arange(log_probs.shape[0], device=log_probs.device), ids_greedy
        ]
        row = {
            "ID": uid,
            "hypothesis": greedy,
            "mean_logprob": float(selected.mean().item()),
            "nonblank_logprob": float(
                selected[nonblank].mean().item()
                if bool(nonblank.any())
                else selected.mean().item()
            ),
            "n_frames": int(logits.shape[0]),
        }
        if ambiguous_delimiter:
            row["raw_processor_decode_invalid"] = raw_processor_decode
            row["delimiter_id"] = 28
            row["blank_id"] = int(model.config.pad_token_id)
        if domain_decoder is not None:
            numpy_logits = logits.float().cpu().numpy()
            row["domain_beam"] = normalize_text(
                domain_decoder.decode(numpy_logits, beam_width=100).replace("|", " ")
            ) or "."
            row["standard_beam"] = normalize_text(
                standard_decoder.decode(numpy_logits, beam_width=100).replace("|", " ")
            ) or "."
        rows.append(row)
        if position % 10 == 0 or position == len(ids):
            print(
                f"{spec.name}: {position}/{len(ids)} "
                f"({(time.time() - started) / position:.2f}s/utt)",
                flush=True,
            )

    del model, processor
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return pd.DataFrame(rows)


@torch.inference_mode()
def decode_whisper(
    spec: ModelSpec,
    ids: list[str],
    examples: dict[str, dict],
    device: torch.device,
) -> pd.DataFrame:
    path = ROOT / spec.path
    processor = WhisperProcessor.from_pretrained(str(path), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(path), local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    try:
        model.generation_config.max_length = None
    except Exception:
        pass

    rows = []
    started = time.time()
    for position, uid in enumerate(ids, start=1):
        array = prepare_audio(examples[uid])
        features = processor(
            array, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(device)
        generated = model.generate(
            features,
            do_sample=False,
            num_beams=1,
            max_new_tokens=256,
        )
        text = normalize_text(
            processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        ) or "."
        rows.append({"ID": uid, "hypothesis": text})
        if position % 10 == 0 or position == len(ids):
            print(
                f"{spec.name}: {position}/{len(ids)} "
                f"({(time.time() - started) / position:.2f}s/utt)",
                flush=True,
            )
    del model, processor
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return pd.DataFrame(rows)


def cache_hypotheses(
    spec: ModelSpec,
    ids: list[str],
    examples: dict[str, dict],
    device: torch.device,
    refresh: bool,
) -> pd.DataFrame:
    output = OUT / f"hyps_{spec.name}.csv"
    if output.exists() and not refresh:
        frame = pd.read_csv(output, dtype=str).fillna("")
    elif spec.kind in {
        "ctc",
        "ctc_beam",
        "ctc_ambiguous_beam",
        "nolimits",
        "auto_ctc",
    }:
        frame = decode_ctc(spec, ids, examples, device)
        frame.to_csv(output, index=False)
    elif spec.kind == "whisper":
        frame = decode_whisper(spec, ids, examples, device)
        frame.to_csv(output, index=False)
    else:
        raise ValueError(f"Unknown model kind: {spec.kind}")
    if frame["ID"].astype(str).tolist() != ids:
        raise RuntimeError(f"{spec.name}: cache does not match exact proxy ID order")
    return frame


def load_sunbird(ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(SUNBIRD_CACHE, dtype=str).fillna("")
    frame = frame.set_index("ID").reindex(ids).reset_index()
    if frame["hypothesis"].eq("").any():
        raise RuntimeError("Sunbird cache is missing exact Luganda proxy IDs")
    output = OUT / "hyps_sunbird51.csv"
    frame[["ID", "hypothesis"]].to_csv(output, index=False)
    return frame[["ID", "hypothesis"]]


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def word_count_ratio(candidate: str, baseline: str) -> float:
    return len(normalize_text(candidate).split()) / max(
        1, len(normalize_text(baseline).split())
    )


def medoid_choice(options: list[tuple[str, str]], incumbent: str) -> str:
    """Return an observed hypothesis minimizing disagreement; tie to incumbent."""
    unique: dict[str, str] = {}
    for name, text in options:
        unique.setdefault(normalize_text(text) or ".", name)
    texts = list(unique)
    if len(texts) == 1:
        return texts[0]
    costs = {
        text: sum(1.0 - similarity(text, other) for other in texts if other != text)
        for text in texts
    }
    best_cost = min(costs.values())
    tied = [text for text, cost in costs.items() if abs(cost - best_cost) < 1e-12]
    normalized_incumbent = normalize_text(incumbent) or "."
    if normalized_incumbent in tied:
        return normalized_incumbent
    return tied[0]


def make_candidates(
    ids: list[str],
    frames: dict[str, pd.DataFrame],
    counts: tuple[dict[str, int], dict[str, int]],
) -> tuple[dict[str, list[str]], dict[str, dict]]:
    columns: dict[str, list[str]] = {}
    metadata: dict[str, dict] = {}
    for name, frame in frames.items():
        greedy = [normalize_text(text) or "." for text in frame["hypothesis"]]
        columns[name] = greedy
        metadata[name] = {"type": "model_greedy", "source": name}
        joined = [apply_splitjoin(text, counts) for text in greedy]
        if joined != greedy:
            key = f"{name}_splitjoin"
            columns[key] = joined
            metadata[key] = {"type": "text_rule", "source": name}
        for beam_column, suffix in (
            ("domain_beam", "domain_beam"),
            ("standard_beam", "standard_beam"),
        ):
            if beam_column in frame:
                beam = [normalize_text(text) or "." for text in frame[beam_column]]
                key = f"{name}_{suffix}"
                columns[key] = beam
                metadata[key] = {"type": "ctc_lm_beam", "source": name}
                joined_beam = [apply_splitjoin(text, counts) for text in beam]
                if joined_beam != beam:
                    joined_key = f"{key}_splitjoin"
                    columns[joined_key] = joined_beam
                    metadata[joined_key] = {"type": "ctc_lm_beam_text_rule", "source": key}

    if "mms_ft_v3_splitjoin" not in columns:
        raise RuntimeError("Production candidate mms_ft_v3_splitjoin is unavailable")
    production = columns["mms_ft_v3_splitjoin"]

    # Reference-free conservative selectors.  These never inspect labels.
    if "mms_ft_v3_domain_beam" in columns and "mms_ft_v4_domain_beam" in columns:
        beam3 = columns["mms_ft_v3_domain_beam"]
        beam4 = columns["mms_ft_v4_domain_beam"]
        v4 = columns["mms_ft_v4_splitjoin"]

        columns["select_beam_exact_agreement"] = [
            b3 if normalize_text(b3) == normalize_text(b4) else base
            for base, b3, b4 in zip(production, beam3, beam4)
        ]
        metadata["select_beam_exact_agreement"] = {
            "type": "reference_free_selector",
            "rule": "use ft-v3 domain beam only when ft-v3 and ft-v4 domain beams exactly agree",
        }

        columns["select_beam_cross_support_095"] = [
            b3
            if (
                similarity(b3, b4) >= 0.95
                and similarity(b3, v4) >= similarity(base, v4)
                and 0.80 <= word_count_ratio(b3, base) <= 1.25
            )
            else base
            for base, b3, b4, v4 in zip(production, beam3, beam4, v4)
        ]
        metadata["select_beam_cross_support_095"] = {
            "type": "reference_free_selector",
            "rule": "beam pair similarity >=.95, beam no farther from v4 than production, length .80-1.25",
        }

        columns["medoid_top_mms"] = [
            medoid_choice(
                [
                    ("production", base),
                    ("v4", h4),
                    ("beam3", b3),
                    ("beam4", b4),
                ],
                base,
            )
            for base, h4, b3, b4 in zip(production, v4, beam3, beam4)
        ]
        metadata["medoid_top_mms"] = {
            "type": "reference_free_fusion",
            "rule": "character-distance medoid of production, v4, ft-v3 beam and ft-v4 beam; ties favor production",
        }

    if (
        "sunbird51" in columns
        and "mms_ft_v3_domain_beam" in columns
        and "mms_ft_v4_splitjoin" in columns
    ):
        sunbird = columns["sunbird51"]
        beam3 = columns["mms_ft_v3_domain_beam"]
        v4 = columns["mms_ft_v4_splitjoin"]
        columns["select_sunbird_two_support_098"] = [
            sun
            if (
                similarity(sun, beam) >= 0.98
                and similarity(sun, h4) >= 0.98
                and 0.85 <= word_count_ratio(sun, base) <= 1.15
            )
            else base
            for base, sun, beam, h4 in zip(production, sunbird, beam3, v4)
        ]
        metadata["select_sunbird_two_support_098"] = {
            "type": "reference_free_selector",
            "rule": "Sunbird must agree >=.98 with domain beam and v4 and pass length guard",
        }

    # Novel acoustic selectors: use Okwija only as an independent agreement
    # signal, never as an unconditional replacement (its matched score loses).
    required = {
        "okwija_mms_adapter_waxal_lug_standard_beam",
        "mms_ft_v3_domain_beam",
        "mms_ft_v4_domain_beam",
        "mms_ft_v4_splitjoin",
    }
    if required.issubset(columns):
        okwija = columns["okwija_mms_adapter_waxal_lug_standard_beam"]
        beam3 = columns["mms_ft_v3_domain_beam"]
        beam4 = columns["mms_ft_v4_domain_beam"]
        v4 = columns["mms_ft_v4_splitjoin"]

        columns["select_v4_okwija_support"] = [
            h4
            if (
                similarity(h4, okw) >= similarity(base, okw) + 0.01
                and similarity(h4, b3) >= similarity(base, b3)
                and 0.90 <= word_count_ratio(h4, base) <= 1.10
            )
            else base
            for base, h4, okw, b3 in zip(production, v4, okwija, beam3)
        ]
        metadata["select_v4_okwija_support"] = {
            "type": "reference_free_selector",
            "rule": "use v4 only when it is >=.01 closer to Okwija than production, no farther from beam, length .90-1.10",
            "novel_acoustic_signal": "Okwija",
        }

        columns["select_beam_okwija_v4_support"] = [
            b3
            if (
                similarity(b3, okw) >= 0.92
                and similarity(b3, b4) >= 0.98
                and similarity(b3, h4) >= similarity(base, h4)
                and 0.85 <= word_count_ratio(b3, base) <= 1.15
            )
            else base
            for base, h4, okw, b3, b4 in zip(
                production, v4, okwija, beam3, beam4
            )
        ]
        metadata["select_beam_okwija_v4_support"] = {
            "type": "reference_free_selector",
            "rule": "beam requires Okwija similarity >=.92, v4-beam similarity >=.98, v4 support and length .85-1.15",
            "novel_acoustic_signal": "Okwija",
            "public_caveat": "full domain beam was already public-exhausted; only this Okwija-gated subset is novel",
        }

    # Verify all outputs preserve the exact evaluation envelope.
    for name, hypotheses in columns.items():
        if len(hypotheses) != len(ids):
            raise RuntimeError(f"{name}: candidate length mismatch")
    return columns, metadata


def inventory(specs: tuple[ModelSpec, ...]) -> dict:
    rows = []
    for spec in specs:
        path = ROOT / spec.path
        model_file = path / "model.safetensors"
        rows.append(
            {
                **asdict(spec),
                "exists": path.exists(),
                "has_config": (path / "config.json").exists(),
                "model_bytes": model_file.stat().st_size if model_file.exists() else 0,
            }
        )
    empty = ROOT / "checkpoints" / "waxal-lug-lmhead-ft"
    corrector = ROOT / "checkpoints" / "mms300-corrector" / "lug"
    return {
        "decoded_distinct_states": rows,
        "deduplicated": {
            "whisper_per_lang_checkpoint_30": "byte-identical to whisper_per_lang_legit/best",
            "whisper_per_lang_checkpoint_100": "byte-identical to whisper_per_lang_legit/best",
        },
        "non_model_luganda_paths": {
            str(empty.relative_to(ROOT)): "empty directory",
            str(corrector.relative_to(ROOT)): "corrector artifacts; no ASR config/model pair",
        },
        "cached_external_hypotheses": {
            "sunbird51": str(SUNBIRD_CACHE.relative_to(ROOT)),
        },
    }


def production_audit() -> dict:
    submission = pd.read_csv(PRODUCTION_SUBMISSION, dtype=str).fillna("")
    routes = pd.read_csv(ROUTE_INDEX, dtype=str).fillna("")
    lug_ids = set(routes.loc[routes["decode_lang"] == "lug", "ID"])
    return {
        "submission": str(PRODUCTION_SUBMISSION.relative_to(ROOT)),
        "submission_sha256": sha256(PRODUCTION_SUBMISSION),
        "rows": len(submission),
        "unique_ids": int(submission["ID"].nunique()),
        "empty_targets": int(submission["Target"].map(normalize_text).eq("").sum()),
        "public_visible_luganda_rows": len(lug_ids),
        "production_proxy_definition": "mms-lug-ft-v3 greedy plus train-lexicon split/join",
        "test_reference_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--refresh-models", action="store_true")
    parser.add_argument("--refresh-corrected-labels", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ids, examples = load_proxy_examples()
    original_refs = {
        uid: normalize_text(str(examples[uid].get("transcription") or "")) or "."
        for uid in ids
    }
    corrected_refs = corrected_reference_map(ids, args.refresh_corrected_labels)
    refs_original = [original_refs[uid] for uid in ids]
    refs_corrected = [corrected_refs[uid] for uid in ids]

    label_audit = pd.DataFrame(
        {
            "ID": ids,
            "original_reference": refs_original,
            "corrected_reference": refs_corrected,
        }
    )
    label_audit["changed"] = (
        label_audit["original_reference"] != label_audit["corrected_reference"]
    )
    label_audit.to_csv(OUT / "reference_audit.csv", index=False)

    selected = set(args.models) if args.models else {spec.name for spec in MODEL_SPECS}
    unknown = selected - {spec.name for spec in MODEL_SPECS}
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    device = pick_device(args.device)
    print(f"device={device} exact_proxy_n={len(ids)}", flush=True)

    frames: dict[str, pd.DataFrame] = {}
    for spec in MODEL_SPECS:
        if spec.name not in selected:
            continue
        path = ROOT / spec.path
        if not (path / "config.json").exists():
            print(f"skip {spec.name}: no local config", flush=True)
            continue
        frames[spec.name] = cache_hypotheses(
            spec, ids, examples, device, args.refresh_models
        )
    frames["sunbird51"] = load_sunbird(ids)

    # A partial invocation may use previously completed exact-ID caches so that
    # selectors remain reproducible and long model sweeps are resumable.
    for spec in MODEL_SPECS:
        if spec.name in frames:
            continue
        cache = OUT / f"hyps_{spec.name}.csv"
        if cache.exists():
            frame = pd.read_csv(cache, dtype=str).fillna("")
            if frame["ID"].astype(str).tolist() == ids:
                frames[spec.name] = frame

    counts = load_splitjoin_counts()
    candidates, candidate_meta = make_candidates(ids, frames, counts)
    metrics: dict[str, dict] = {}
    production_key = "mms_ft_v3_splitjoin"
    production_corrected = metric(refs_corrected, candidates[production_key])
    for name, hypotheses in candidates.items():
        corrected = metric(refs_corrected, hypotheses)
        original = metric(refs_original, hypotheses)
        changed_vs_production = sum(
            normalize_text(hyp) != normalize_text(base)
            for hyp, base in zip(hypotheses, candidates[production_key])
        )
        metrics[name] = {
            "original_labels": original,
            "corrected_labels": corrected,
            "delta_zindi_vs_production_corrected": float(
                corrected["zindi"] - production_corrected["zindi"]
            ),
            "changed_rows_vs_production": changed_vs_production,
            **candidate_meta.get(name, {}),
        }

    ranking = sorted(
        metrics,
        key=lambda name: metrics[name]["corrected_labels"]["zindi"],
        reverse=True,
    )
    offline_passers = [
        name
        for name in ranking
        if metrics[name]["corrected_labels"]["zindi"]
        > production_corrected["zindi"]
        and metrics[name]["corrected_labels"]["wer"]
        <= production_corrected["wer"]
    ]
    public_exhausted = {
        "mms_ft_v3_domain_beam",
        "mms_ft_v3_domain_beam_splitjoin",
        "mms_ft_v3_standard_beam",
        "mms_ft_v3_standard_beam_splitjoin",
    }
    novel_passers = [name for name in offline_passers if name not in public_exhausted]
    selector_passers = [
        name
        for name in novel_passers
        if candidate_meta.get(name, {}).get("type")
        in {"reference_free_selector", "reference_free_fusion"}
    ]
    raw_model_passers = [
        name
        for name in novel_passers
        if candidate_meta.get(name, {}).get("type")
        in {"model_greedy", "text_rule"}
    ]

    v5b_confirmation_path = OUT / "v5b_n150.json"
    v5b_confirmation = (
        json.loads(v5b_confirmation_path.read_text())
        if v5b_confirmation_path.exists()
        else None
    )
    # Deployment requires evidence beyond the 40-row proxy because the same
    # proxy overstated the already-submitted full beam by ~0.0225.  No raw
    # novel model survives the independent 150-row check; selector positives
    # remain exploratory because they are tiny subsets gated around that
    # publicly failed beam signal.
    deployment_passers: list[str] = []

    details = pd.DataFrame({"ID": ids})
    details["original_reference"] = refs_original
    details["corrected_reference"] = refs_corrected
    for name in ranking:
        details[name] = candidates[name]
    details.to_csv(OUT / "matched_hypotheses.csv", index=False)

    report = {
        "protocol": {
            "split": "validation",
            "language": "lug",
            "exact_ids": ids,
            "n": len(ids),
            "test_labels_used": False,
            "corrected_label_source": CORRECTED_PARQUET,
            "corrected_label_mapping": "immutable example id",
            "corrected_label_limitations": "single external correction pass; not independently human re-verified here",
        },
        "reference_audit": {
            "n_original": len(refs_original),
            "n_corrected": len(refs_corrected),
            "n_changed_after_normalization": int(label_audit["changed"].sum()),
            "all_ids_matched": True,
        },
        "production": production_audit(),
        "inventory": inventory(MODEL_SPECS),
        "production_candidate": production_key,
        "production_corrected_metrics": production_corrected,
        "ranking_corrected": ranking,
        "offline_passes_production": offline_passers,
        "public_exhausted_candidates": sorted(public_exhausted),
        "novel_passes_production": novel_passers,
        "exploratory_selector_passes": selector_passers,
        "fixed_proxy_raw_model_passes": raw_model_passers,
        "deployment_passes_production": deployment_passers,
        "deployment_verdict": (
            "No novel Luganda candidate passes the production/public-transfer gate. "
            "Do not rebuild or submit a Luganda replacement from this sweep."
        ),
        "independent_confirmation": {
            "v5b_seed42_n150": v5b_confirmation,
            "interpretation": "v5b loses production on n=150 despite a small n=40 proxy gain",
        },
        "public_evidence": {
            "artifact": "outputs/beat075/PUBLIC_RESULT_beam_q7wH165R.md",
            "full_domain_beam_changed_rows": 402,
            "public_score_delta": 0.000010656,
            "wer_delta": 0.0,
            "verdict": "full Luganda domain beam is exhausted and cannot be promoted",
        },
        "external_candidate_audit": {
            "sulaimank/w2v-bert-2.0-lg-CV-Fleurs-300": {
                "access": "manual gated",
                "parameters": 605733751,
                "decision": "not accessible; base checkpoint was not evaluated",
            },
            "cdli/whisper-large-v3_finetuned_ugandan_luganda_waxal_7_standard_speech_v1.0": {
                "access": "open and already cached",
                "card": "WAXAL test standard WER 0.13 using Whisper language=sw",
                "prior_matched_artifact": "outputs/proxy_whisper_large_lug.json",
                "prior_matched_zindi": 0.8359964829988922,
                "decision": "not re-decoded; prior same-proxy evidence loses the production family",
            },
            "sulaimank/w2v-bert-grain-lg-v2": {
                "access": "open",
                "card": "Grain eval WER 0.0299 CER 0.0077",
                "matched_candidate": "sulaimank_w2vbert_grain_lg_v2",
                "decision": "evaluated and rejected on exact WAXAL IDs",
            },
        },
        "okwija_tokenizer_audit": {
            "model_vocab_size": 33,
            "word_delimiter_id": 28,
            "duplicate_vocab_aliases_at_28": ["|", "ᵑ"],
            "blank_pad_id": 30,
            "unassigned_hole_id": 0,
            "repair": "manual CTC collapse maps 28 to space; beam label 0 is inert and 28 is |",
            "raw_processor_decode_accepted": False,
        },
        "best_candidate": ranking[0],
        "metrics": metrics,
        "artifacts": {
            "matched_hypotheses": str((OUT / "matched_hypotheses.csv").relative_to(ROOT)),
            "reference_audit": str((OUT / "reference_audit.csv").relative_to(ROOT)),
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "reference_audit": report["reference_audit"],
        "production": production_corrected,
        "ranking": [
            {
                "name": name,
                **metrics[name]["corrected_labels"],
                "delta": metrics[name]["delta_zindi_vs_production_corrected"],
                "changed": metrics[name]["changed_rows_vs_production"],
            }
            for name in ranking
        ],
        "offline_passes_production": offline_passers,
        "novel_passes_production": novel_passers,
        "deployment_passes_production": deployment_passers,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
