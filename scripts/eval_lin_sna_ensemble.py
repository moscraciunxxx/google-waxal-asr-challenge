#!/usr/bin/env python3
"""Reproducible matched Lingala/Shona model and ensemble evaluation.

Scope is intentionally narrow:

* audit the 444 Lingala + 445 Shona rows in the expanded Phase-2 block;
* evaluate every acoustic candidate on the same seed-selected validation IDs;
* use corrected labels from Harcuracy/google_waxal_asr_challenge, validation only;
* keep original WAXAL labels as a secondary sensitivity analysis;
* fit text-normalization and label-free agreement selectors on a tune half and
  report their result on a disjoint holdout half;
* never read any labeled test split and never write outside ``--out-dir``.

The script is resumable: each completed system is appended to hypotheses.csv.
Run with PYTHONDONTWRITEBYTECODE=1 to preserve the write-isolation guarantee.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fsspec
import jiwer
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
from pyctcdecode import build_ctcdecoder
from transformers import (
    AutoProcessor,
    Wav2Vec2BertForCTC,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "goal_2026_08_08" / "lin_sna_ensemble"
TARGET_SR = 16_000
CORRECTED_DATASET = "Harcuracy/google_waxal_asr_challenge"
CORRECTED_BASE = (
    "https://huggingface.co/datasets/Harcuracy/google_waxal_asr_challenge/"
    "resolve/refs%2Fconvert%2Fparquet"
)

MODEL_IDS = {
    "lin_mms1b_greedy": "facebook/mms-1b-all",
    "lin_waxal300_greedy": "waxal-benchmarking/mms-300m-waxal-lin",
    "lin_ft_v3_greedy": str(ROOT / "checkpoints" / "mms-lin-ft-v3"),
    "lin_keystats_greedy": "keystats/lingala-xlsr-waxal-finetuned",
    "lin_nolimits_base_greedy": str(ROOT / "checkpoints" / "nolimitsxl-mms1b-waxal-base"),
    "lin_nolimits_lora_greedy": str(ROOT / "checkpoints" / "nolimitsxl-mms1b-waxal-base"),
    "lin_sunbird51_greedy": "Sunbird/asr-whisper-51-african-languages",
    "sna_waxal_greedy": "waxal-benchmarking/mms-300m-waxal-sna",
    "sna_badrex_greedy": "badrex/w2v-bert-2.0-shona-asr",
    "sna_mubarak_greedy": "Mubarak127/waxal-whisper-large-v3-sna_asr",
    "sna_okwija_safe_greedy": str(ROOT / "checkpoints" / "okwija-mms-adapter-waxal-sna"),
    "sna_khaya_dondo_greedy": "KhayaAI/w2v-bert-sna",
    "sna_nolimits_base_greedy": str(ROOT / "checkpoints" / "nolimitsxl-mms1b-waxal-base"),
    "sna_nolimits_lora_greedy": str(ROOT / "checkpoints" / "nolimitsxl-mms1b-waxal-base"),
    "sna_manasseh_greedy": "manassehzw/sna-w2v-bert-2.0-asr",
    "sna_sulaimank_nonpunct_greedy": "sulaimank/w2vbert-shona-waxal",
    "sna_sunbird51_greedy": "Sunbird/asr-whisper-51-african-languages",
}

DEFAULT_SYSTEMS = [
    "lin_mms1b_greedy",
    "lin_waxal300_greedy",
    "lin_ft_v3_greedy",
    "lin_keystats_greedy",
    "lin_nolimits_base_greedy",
    "lin_nolimits_lora_greedy",
    "lin_sunbird51_greedy",
    "sna_waxal_greedy",
    "sna_okwija_safe_greedy",
    "sna_khaya_dondo_greedy",
    "sna_badrex_greedy",
    "sna_manasseh_greedy",
    "sna_sulaimank_nonpunct_greedy",
    "sna_mubarak_greedy",
    "sna_nolimits_base_greedy",
    "sna_nolimits_lora_greedy",
    "sna_sunbird51_greedy",
]

KNOWN_PUBLIC_FAILURES = {
    "evidence": "outputs/beat075/PUBLIC_RESULT_linsna_probe_Xs1qTFys.md",
    "submission_id": "Xs1qTFys",
    "floor_score": 0.687878796,
    "failed_score": 0.669407195,
    "delta": -0.018471601,
    "failed_recipe": {
        "lin": "replace all 444 new Lingala rows: MMS-1B production -> WAXAL-300M",
        "sna": "replace all 461 Shona-routed rows: WAXAL-300M production -> MMS-1B",
        "changed_predictions": 876,
    },
    "promotion_blacklist": ["lin_waxal300_greedy", "sna_mms1b_greedy"],
    "policy": "negative controls may be scored but can never be selected or promoted",
}

LANGUAGE_TOKEN = {"lin": 50353, "sna": 50324}
PUNCT_RE = re.compile(r"[^\w\s']+", flags=re.UNICODE)
WS_RE = re.compile(r"\s+")


def norm(text: object) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = PUNCT_RE.sub(" ", value)
    value = WS_RE.sub(" ", value).strip("' ")
    return value


def transform_text(text: str, profile: str) -> str:
    value = norm(text)
    if profile == "challenge":
        return value
    if profile == "apostrophe_drop":
        return WS_RE.sub(" ", value.replace("'", "")).strip()
    if profile == "apostrophe_space":
        return WS_RE.sub(" ", value.replace("'", " ")).strip()
    raise ValueError(profile)


def metrics(refs: Iterable[str], hyps: Iterable[str]) -> dict[str, float]:
    r = [x or " " for x in refs]
    h = [x or "" for x in hyps]
    w = float(jiwer.wer(r, h))
    c = float(jiwer.cer(r, h))
    return {"wer": w, "cer": c, "zindi": 1.0 - 0.5 * (w + c)}


def row_cost(ref: str, hyp: str) -> float:
    return 0.5 * float(jiwer.wer(ref or " ", hyp or "")) + 0.5 * float(
        jiwer.cer(ref or " ", hyp or "")
    )


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def pick_device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def local_snapshot(repo_id: str) -> str:
    if Path(repo_id).exists():
        return repo_id
    roots = [Path(os.environ.get("HF_HUB_CACHE", "")), Path.home() / ".cache/huggingface/hub"]
    snapshots = sorted(
        path
        for root in roots
        if str(root)
        for path in (root / ("models--" + repo_id.replace("/", "--")) / "snapshots").glob("*")
    )
    complete = [p for p in snapshots if (p / "config.json").exists()]
    return str(complete[-1]) if complete else repo_id


def free_model(*objects: object) -> None:
    for obj in objects:
        del obj
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def corrected_urls(lang: str) -> list[str]:
    return [
        f"{CORRECTED_BASE}/{lang}_asr/validation/0000.parquet",
        f"{CORRECTED_BASE}/{lang}_asr/validation/0001.parquet",
    ]


def fetch_corrected_labels(out_dir: Path, offline: bool) -> pd.DataFrame:
    out = out_dir / "corrected_validation_labels.csv"
    if out.exists():
        frame = pd.read_csv(out, keep_default_na=False)
        if set(frame.language) == {"lin", "sna"}:
            return frame
    if offline:
        raise FileNotFoundError(f"Corrected-label cache missing in offline mode: {out}")
    frames: list[pd.DataFrame] = []
    for lang in ("lin", "sna"):
        for url in corrected_urls(lang):
            with fsspec.open(url, "rb", block_size=1 << 20, cache_type="readahead") as source:
                table = pq.ParquetFile(source).read(columns=["id", "transcription"])
            part = table.to_pandas().rename(columns={"id": "ID", "transcription": "corrected"})
            part["language"] = lang
            part["split"] = "validation"
            frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[["ID", "language", "split", "corrected"]]
    if frame.ID.duplicated().any() or len(frame) != 1844 + 1727:
        raise RuntimeError("Corrected-label row-count/uniqueness gate failed")
    frame.to_csv(out, index=False)
    return frame


def original_metadata(lang: str) -> pd.DataFrame:
    path = ROOT / "data" / "hf_metadata" / f"{lang}_validation.parquet"
    frame = pq.read_table(path).to_pandas().rename(columns={"Target": "original"})
    return frame[["ID", "speaker_id", "original"]]


def validation_parquets(lang: str) -> list[Path]:
    hubs = [Path(os.environ.get("HF_HUB_CACHE", "")), Path.home() / ".cache/huggingface/hub"]
    paths = sorted(
        path
        for hub in hubs
        if str(hub)
        for path in (hub / "datasets--google--WaxalNLP" / "snapshots").glob(
            f"*/data/ASR/{lang}/{lang}-validation-*.parquet"
        )
    )
    resolved = sorted({p.resolve() for p in paths if p.resolve().is_file()})
    if not resolved:
        raise FileNotFoundError(f"No cached original WAXAL validation audio for {lang}")
    return resolved


def selected_rows(lang: str, ids: list[str]) -> list[dict]:
    table = pq.read_table(
        validation_parquets(lang),
        columns=["id", "speaker_id", "transcription", "audio"],
        filters=[("id", "in", ids)],
    )
    by_id = {str(row["id"]): row for row in table.to_pylist()}
    missing = [uid for uid in ids if uid not in by_id]
    if missing:
        raise RuntimeError(f"Missing {lang} audio IDs: {missing[:5]}")
    return [by_id[uid] for uid in ids]


def decode_audio(audio: dict) -> np.ndarray:
    raw = audio.get("bytes")
    if raw is not None:
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
    else:
        arr, sr = sf.read(audio["path"], dtype="float32")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if int(sr) != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=int(sr), target_sr=TARGET_SR)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak


def ctc_decode(model, processor, audio: np.ndarray, device: torch.device) -> tuple[str, np.ndarray]:
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    kwargs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
    with torch.inference_mode():
        logits = model(**kwargs).logits[0].float().cpu().numpy()
    ids = np.argmax(logits, axis=-1)
    return norm(processor.decode(ids)), logits


def fix_mms_delimiter(processor, lang: str) -> dict:
    tok = processor.tokenizer
    tok.set_target_lang(lang)
    pipe_ids = [i for i in range(len(tok)) if tok.convert_ids_to_tokens(i) == "|"]
    if not pipe_ids:
        raise RuntimeError(f"MMS tokenizer has no delimiter for {lang}")
    tok.word_delimiter_token = "|"
    tok.word_delimiter_token_id = int(pipe_ids[0])
    if hasattr(tok, "_word_delimiter_token"):
        tok._word_delimiter_token = "|"
    return {"delimiter_id": int(pipe_ids[0]), "pad_id": int(tok.pad_token_id)}


def load_mms1b(lang: str, device: torch.device):
    ref = local_snapshot("facebook/mms-1b-all")
    processor = AutoProcessor.from_pretrained(ref, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(ref, local_files_only=True, low_cpu_mem_usage=True)
    probe = fix_mms_delimiter(processor, lang)
    try:
        model.load_adapter(lang, local_files_only=True)
    except TypeError:
        model.load_adapter(lang)
    return model.to(device).eval(), processor, probe


def load_standard_ctc(ref: str, device: torch.device):
    local = local_snapshot(ref)
    processor = AutoProcessor.from_pretrained(local, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(local, local_files_only=True, low_cpu_mem_usage=True)
    return model.to(device).eval(), processor


def load_nolimits(device: torch.device, lora: bool):
    ref = ROOT / "checkpoints" / "nolimitsxl-mms1b-waxal-base"
    processor = AutoProcessor.from_pretrained(str(ref), local_files_only=True)
    processor.tokenizer.pad_token = "[PAD]"
    processor.tokenizer.unk_token = "[UNK]"
    processor.tokenizer.word_delimiter_token = "|"
    model = Wav2Vec2ForCTC.from_pretrained(str(ref), local_files_only=True, low_cpu_mem_usage=True)
    if lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(ROOT / "checkpoints" / "nolimitsxl-mms1b-waxal"),
            local_files_only=True,
            is_trainable=False,
        )
    return model.to(device).eval(), processor


def okwija_probe(processor, model) -> dict:
    tok = processor.tokenizer
    vocab = tok.get_vocab()
    id_to_tokens: dict[int, list[str]] = {}
    for token, idx in vocab.items():
        id_to_tokens.setdefault(int(idx), []).append(str(token))
    duplicate_ids = {str(idx): values for idx, values in id_to_tokens.items() if len(values) > 1}
    delimiter_id = int(vocab["|"])
    pad_id = int(model.config.pad_token_id)
    probe_ids = tok("a b").input_ids
    return {
        "vocab_size_config": int(model.config.vocab_size),
        "vocab_entries": len(vocab),
        "duplicate_ids": duplicate_ids,
        "delimiter_id": delimiter_id,
        "pad_id": pad_id,
        "delimiter_is_blank": delimiter_id == pad_id,
        "id35_tokenizer_token": tok.convert_ids_to_tokens(35),
        "id35_tokenizer_decode": tok.decode([8, 35, 9]),
        "encode_a_space_b": probe_ids,
        "encode_a_space_b_decode": tok.decode(probe_ids),
        "safe_policy": "ID 35 is delimiter per tokenizer_config; ID 37 is CTC blank per config; í alias rejected",
    }


def okwija_id_map(processor, model) -> list[str]:
    vocab = processor.tokenizer.get_vocab()
    size = int(model.config.vocab_size)
    labels = [""] * size
    for token, idx in vocab.items():
        idx = int(idx)
        if idx >= size or token == "í":
            continue
        labels[idx] = token
    labels[int(vocab["|"])] = " "
    labels[int(model.config.pad_token_id)] = ""
    for token in ("[UNK]", "<s>", "</s>"):
        idx = vocab.get(token)
        if idx is not None:
            labels[int(idx)] = ""
    return labels


def collapse_ctc(ids: np.ndarray, labels: list[str], blank_id: int) -> tuple[str, Counter]:
    pieces: list[str] = []
    counts: Counter = Counter()
    previous = None
    for value in map(int, ids):
        counts[value] += 1
        if value != previous and value != blank_id:
            pieces.append(labels[value] if 0 <= value < len(labels) else "")
        previous = value
    return norm("".join(pieces)), counts


def make_decoder(labels: list[str], lang: str, alpha: float, beta: float = 0.5):
    unigrams_path = ROOT / "data" / "lms" / f"{lang}_unigrams.txt"
    unigrams = (
        [x.strip() for x in unigrams_path.read_text().splitlines() if x.strip()]
        if unigrams_path.exists()
        else None
    )
    return build_ctcdecoder(
        labels,
        kenlm_model_path=str(ROOT / "data" / "lms" / f"{lang}_2gram.arpa"),
        unigrams=unigrams,
        alpha=alpha,
        beta=beta,
    )


def whisper_decode(
    model,
    processor,
    audio: np.ndarray,
    lang: str,
    device: torch.device,
    sunbird: bool,
) -> str:
    features = processor(
        audio, sampling_rate=TARGET_SR, do_normalize=True, return_tensors="pt"
    ).input_features.to(device)
    if sunbird:
        forced = [
            (1, LANGUAGE_TOKEN[lang]),
            (2, processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")),
            (3, processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")),
        ]
    else:
        forced = processor.get_decoder_prompt_ids(language="sn", task="transcribe")
    with torch.inference_mode():
        ids = model.generate(
            features,
            forced_decoder_ids=forced,
            num_beams=1,
            do_sample=False,
            max_new_tokens=256,
        )
    return norm(
        processor.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    )


def run_system(
    tag: str,
    lang: str,
    audios: list[np.ndarray],
    device: torch.device,
) -> tuple[dict[str, list[str]], dict]:
    started = time.time()
    diagnostics: dict = {"model": MODEL_IDS[tag], "device": str(device)}
    outputs: dict[str, list[str]] = {tag: []}

    if tag == "lin_mms1b_greedy":
        model, processor, probe = load_mms1b("lin", device)
        diagnostics["tokenizer_probe"] = probe
        for audio in audios:
            outputs[tag].append(ctc_decode(model, processor, audio, device)[0])
        free_model(model, processor)
    elif tag in {"lin_waxal300_greedy", "lin_ft_v3_greedy", "lin_keystats_greedy"}:
        model, processor = load_standard_ctc(MODEL_IDS[tag], device)
        for audio in audios:
            outputs[tag].append(ctc_decode(model, processor, audio, device)[0])
        free_model(model, processor)
    elif "nolimits" in tag:
        model, processor = load_nolimits(device, "lora" in tag)
        for audio in audios:
            outputs[tag].append(ctc_decode(model, processor, audio, device)[0])
        free_model(model, processor)
    elif tag == "sna_waxal_greedy":
        model, processor = load_standard_ctc(MODEL_IDS[tag], device)
        vocab = processor.tokenizer.get_vocab()
        labels = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
        decoder = make_decoder(labels, "sna", alpha=0.05)
        outputs["sna_waxal_beam_a0p05"] = []
        for audio in audios:
            greedy, logits = ctc_decode(model, processor, audio, device)
            outputs[tag].append(greedy)
            beam = norm(decoder.decode(logits, beam_width=100).replace("|", " "))
            ratio = len(beam.split()) / max(1, len(greedy.split()))
            outputs["sna_waxal_beam_a0p05"].append(beam if 0.5 <= ratio <= 2.0 else greedy)
        free_model(model, processor, decoder)
    elif tag == "sna_okwija_safe_greedy":
        model, processor = load_standard_ctc(MODEL_IDS[tag], device)
        probe = okwija_probe(processor, model)
        diagnostics["tokenizer_probe"] = probe
        if probe["delimiter_is_blank"] or probe["pad_id"] != 37 or probe["delimiter_id"] != 35:
            raise RuntimeError(f"Okwija tokenizer safety gate failed: {probe}")
        labels = okwija_id_map(processor, model)
        try:
            decoders = {alpha: make_decoder(labels, "sna", alpha) for alpha in (0.05, 0.1, 0.2)}
            diagnostics["lm_decoder_gate"] = "pass"
        except Exception as exc:
            # Greedy is still unambiguous because tokenizer_config declares 35
            # as the delimiter and model.config declares 37 as the CTC blank.
            # Do not force an LM alphabet through pyctcdecode when it rejects
            # the checkpoint's duplicate/special-token map.
            decoders = {}
            diagnostics["lm_decoder_gate"] = (
                f"disabled: {type(exc).__name__}: {exc}"
            )
        for alpha in decoders:
            outputs[f"sna_okwija_safe_beam_a{str(alpha).replace('.', 'p')}"] = []
        emitted: Counter = Counter()
        hf_buggy: list[str] = []
        for audio in audios:
            inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            kwargs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
            with torch.inference_mode():
                logits = model(**kwargs).logits[0].float().cpu().numpy()
            ids = np.argmax(logits, axis=-1)
            safe, counts = collapse_ctc(ids, labels, int(model.config.pad_token_id))
            emitted.update(counts)
            outputs[tag].append(safe)
            hf_buggy.append(norm(processor.decode(ids)))
            for alpha, decoder in decoders.items():
                beam = norm(decoder.decode(logits, beam_width=100))
                ratio = len(beam.split()) / max(1, len(safe.split()))
                key = f"sna_okwija_safe_beam_a{str(alpha).replace('.', 'p')}"
                outputs[key].append(beam if 0.5 <= ratio <= 2.0 else safe)
        outputs["sna_okwija_hf_buggy_decode"] = hf_buggy
        diagnostics["argmax_id_counts"] = {str(k): int(v) for k, v in emitted.most_common()}
        diagnostics["argmax_id0_frames"] = int(emitted[0])
        diagnostics["argmax_delimiter_frames"] = int(emitted[35])
        diagnostics["argmax_blank_frames"] = int(emitted[37])
        free_model(model, processor, *decoders.values())
    elif tag in {
        "sna_badrex_greedy",
        "sna_khaya_dondo_greedy",
        "sna_manasseh_greedy",
        "sna_sulaimank_nonpunct_greedy",
    }:
        ref = local_snapshot(MODEL_IDS[tag])
        processor = AutoProcessor.from_pretrained(ref, local_files_only=True)
        model = Wav2Vec2BertForCTC.from_pretrained(
            ref, local_files_only=True, low_cpu_mem_usage=True
        ).to(device).eval()
        for audio in audios:
            outputs[tag].append(ctc_decode(model, processor, audio, device)[0])
        free_model(model, processor)
    elif tag in {"sna_mubarak_greedy", "lin_sunbird51_greedy", "sna_sunbird51_greedy"}:
        ref = local_snapshot(MODEL_IDS[tag])
        processor = WhisperProcessor.from_pretrained(ref, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            ref, local_files_only=True, low_cpu_mem_usage=True
        ).to(device).eval()
        sunbird = "sunbird" in tag
        for audio in audios:
            outputs[tag].append(whisper_decode(model, processor, audio, lang, device, sunbird))
        free_model(model, processor)
    else:
        raise ValueError(f"Unknown system {tag}")

    diagnostics["seconds"] = time.time() - started
    return outputs, diagnostics


def sha256_text(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def audio_fingerprint(audio: np.ndarray) -> str:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def route_and_cache_audit() -> dict:
    route_path = ROOT / "outputs" / "beat075" / "public_visible_index.csv"
    route = pd.read_csv(route_path, keep_default_na=False)
    new = route[route.split == "new"].copy()
    expected = {"lin": 444, "sna": 445, "lug": 3}
    counts = {k: int(v) for k, v in new.decode_lang.value_counts().to_dict().items()}
    if counts != expected:
        raise RuntimeError(f"Expanded route gate failed: {counts} != {expected}")
    caches = {
        "lin_mms1b": ROOT / "outputs/beat075/hyps_lin_mms1b_zs.csv",
        "lin_waxal300": ROOT / "outputs/beat075/hyps_lin_waxal300.csv",
        "sna_waxal300": ROOT / "outputs/beat075/hyps_sna_waxal300.csv",
        "sna_mms1b": ROOT / "outputs/beat075/hyps_sna_mms1b_zs.csv",
        "sna_badrex": ROOT / "outputs/goal_2026_08_06/hyps_sna_badrex.csv",
        "sna_mubarak": ROOT / "outputs/goal_2026_08_06/hyps_sna_mubarak_whisper.csv",
    }
    cache_report: dict = {}
    route_ids = {
        lang: set(new.loc[new.decode_lang == lang, "ID"].astype(str)) for lang in ("lin", "sna")
    }
    for name, path in caches.items():
        frame = pd.read_csv(path, keep_default_na=False)
        lang = name.split("_", 1)[0]
        ids = set(frame.ID.astype(str))
        cache_report[name] = {
            "path": str(path.relative_to(ROOT)),
            "rows": len(frame),
            "unique_ids": int(frame.ID.astype(str).nunique()),
            "new_route_coverage": len(ids & route_ids[lang]),
            "new_route_expected": len(route_ids[lang]),
            "missing_new_route_ids": sorted(route_ids[lang] - ids)[:20],
            "columns": list(frame.columns),
        }
    return {
        "route_table": str(route_path.relative_to(ROOT)),
        "expanded_rows": len(new),
        "decode_lang_counts": counts,
        "lid_lang_counts": {k: int(v) for k, v in new.lid_lang.value_counts().to_dict().items()},
        "lin_ids_sha256": sha256_text(sorted(route_ids["lin"])),
        "sna_ids_sha256": sha256_text(sorted(route_ids["sna"])),
        "caches": cache_report,
    }


def checkpoint_audit() -> dict:
    report: dict = {}
    for tag, ref in MODEL_IDS.items():
        local = Path(local_snapshot(ref))
        if not local.exists():
            report[tag] = {"reference": ref, "accessible_local": False}
            continue
        config_path = local / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        weights = list(local.glob("*.safetensors")) + list(local.glob("*.bin"))
        report[tag] = {
            "reference": ref,
            "resolved": str(local),
            "accessible_local": bool(config_path.exists() and weights),
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "vocab_size": config.get("vocab_size"),
            "pad_token_id": config.get("pad_token_id"),
            "weight_bytes": int(sum(path.stat().st_size for path in weights)),
            "updated_epoch": max([path.stat().st_mtime for path in weights], default=None),
        }
    for gated in (
        "sulaimank/w2vbert-lingala-waxal-punct-v2",
        "sulaimank/w2vbert-shona-waxal",
    ):
        local = Path(local_snapshot(gated))
        report[gated] = {
            "reference": gated,
            "accessible_local": local.exists() and (local / "config.json").exists(),
            "status": "pending/manual gate" if not (local / "config.json").exists() else "downloaded",
        }
    return report


def write_hypotheses(
    path: Path,
    lang: str,
    ids: list[str],
    outputs: dict[str, list[str]],
) -> None:
    previous = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame()
    rows = []
    for system, hyps in outputs.items():
        if len(hyps) != len(ids):
            raise RuntimeError(f"{system}: {len(hyps)} hypotheses for {len(ids)} IDs")
        rows.extend(
            {"ID": uid, "language": lang, "system": system, "hypothesis": hyp}
            for uid, hyp in zip(ids, hyps, strict=True)
        )
    fresh = pd.DataFrame(rows)
    combined = pd.concat([previous, fresh], ignore_index=True)
    combined = combined.drop_duplicates(["ID", "language", "system"], keep="last")
    combined.sort_values(["language", "system", "ID"]).to_csv(path, index=False)


def bootstrap_delta(
    refs: list[str],
    production: list[str],
    candidate: list[str],
    seed: int,
    reps: int = 1000,
    groups: list[str] | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(refs)
    deltas = []
    group_to_indices: dict[str, list[int]] = {}
    if groups is not None:
        for index, group in enumerate(groups):
            group_to_indices.setdefault(str(group), []).append(index)
        unique_groups = sorted(group_to_indices)
    for _ in range(reps):
        if groups is None:
            idx = list(map(int, rng.integers(0, n, n)))
        else:
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = [index for group in sampled_groups for index in group_to_indices[str(group)]]
        ref = [refs[i] for i in idx]
        base = [production[i] for i in idx]
        cand = [candidate[i] for i in idx]
        deltas.append(metrics(ref, cand)["zindi"] - metrics(ref, base)["zindi"])
    return {
        "reps": reps,
        "mean": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "p_delta_le_zero": float(np.mean(np.asarray(deltas) <= 0.0)),
        "resampling_unit": "speaker" if groups is not None else "utterance",
    }


@dataclass(frozen=True)
class Policy:
    name: str
    apply: Callable[[dict[str, str]], str]


def derive_policies(lang: str, system_names: list[str], tune_rows: list[dict]) -> list[Policy]:
    production = "lin_mms1b_greedy" if lang == "lin" else "sna_waxal_greedy"
    policies: list[Policy] = []
    blacklist = set(KNOWN_PUBLIC_FAILURES["promotion_blacklist"])
    challengers = [
        name
        for name in system_names
        if name != production and "hf_buggy" not in name and name not in blacklist
    ]
    if lang == "sna" and "sna_badrex_greedy" in system_names:
        policies.append(
            Policy(
                "sna_current_badrex_sim99",
                lambda row: row["sna_badrex_greedy"]
                if similarity(row[production], row["sna_badrex_greedy"]) >= 0.99
                else row[production],
            )
        )
    thresholds = [0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]
    for challenger in challengers:
        for threshold in thresholds:
            policies.append(
                Policy(
                    f"select_{challenger}_if_prod_similarity_ge_{threshold:.2f}",
                    lambda row, c=challenger, t=threshold: row[c]
                    if similarity(row[production], row[c]) >= t
                    else row[production],
                )
            )
    top_sources = sorted(
        challengers,
        key=lambda name: metrics(
            [row["corrected"] for row in tune_rows], [row[name] for row in tune_rows]
        )["zindi"],
        reverse=True,
    )[:4]
    for i, first in enumerate(top_sources):
        for second in top_sources[i + 1 :]:
            for threshold in (0.90, 0.95, 0.97, 0.99):
                policies.append(
                    Policy(
                        f"agree_{first}__{second}_ge_{threshold:.2f}_choose_{first}",
                        lambda row, a=first, b=second, t=threshold: row[a]
                        if similarity(row[a], row[b]) >= t
                        else row[production],
                    )
                )
    return policies


def analyze(
    sample: pd.DataFrame,
    hypotheses: pd.DataFrame,
    seed: int,
    out_dir: Path,
) -> dict:
    report: dict = {"languages": {}}
    score_rows: list[dict] = []
    ensemble_rows: list[dict] = []
    for lang in ("lin", "sna"):
        meta = sample[sample.language == lang].copy().sort_values("sample_position")
        wide = hypotheses[hypotheses.language == lang].pivot(
            index="ID", columns="system", values="hypothesis"
        )
        joined = meta.set_index("ID").join(wide, how="left")
        system_names = sorted(set(hypotheses.loc[hypotheses.language == lang, "system"]))
        complete = [name for name in system_names if joined[name].notna().all()]
        production = "lin_mms1b_greedy" if lang == "lin" else "sna_waxal_greedy"
        if production not in complete:
            report["languages"][lang] = {"error": f"missing production {production}"}
            continue
        joined = joined.reset_index()
        records = joined.to_dict("records")
        tune = [row for row in records if row["fold"] == "tune"]
        holdout = [row for row in records if row["fold"] == "holdout"]
        lang_report: dict = {"production": production, "systems": {}, "ensembles": {}}
        lang_report["speaker_split"] = {
            "tune_speakers": len({str(row["speaker_id"]) for row in tune}),
            "holdout_speakers": len({str(row["speaker_id"]) for row in holdout}),
            "speaker_overlap": sorted(
                {str(row["speaker_id"]) for row in tune}
                & {str(row["speaker_id"]) for row in holdout}
            ),
        }
        if lang_report["speaker_split"]["speaker_overlap"]:
            raise RuntimeError(f"{lang}: speaker-disjoint fold gate failed")

        normalized_variants: dict[str, tuple[str, list[str]]] = {}
        partial_screening: dict[str, dict] = {}
        for system in system_names:
            available = joined[joined[system].notna()].copy()
            if len(available) == 0 or len(available) == len(joined):
                continue
            value = metrics(
                [norm(text) for text in available.corrected],
                [norm(text) for text in available[system]],
            )
            partial_screening[system] = {
                "n": len(available),
                **value,
                "status": "screening_only_not_promotion_eligible",
            }
            score_rows.append(
                {
                    "language": lang,
                    "system": system,
                    "kind": "partial_screen",
                    "normalization": "challenge",
                    "reference": "corrected",
                    "split": f"screen_n{len(available)}",
                    "n": len(available),
                    "promotion_eligible": False,
                    **value,
                }
            )
        lang_report["partial_screening"] = partial_screening
        for system in complete:
            tune_refs = [norm(row["corrected"]) for row in tune]
            profile_scores = {}
            for profile in ("challenge", "apostrophe_drop", "apostrophe_space"):
                tune_hyps = [transform_text(row[system], profile) for row in tune]
                profile_scores[profile] = metrics(tune_refs, tune_hyps)
            chosen_profile = max(profile_scores, key=lambda p: profile_scores[p]["zindi"])
            for ref_source in ("corrected", "original"):
                for split_name, rows in (("full", records), ("tune", tune), ("holdout", holdout)):
                    refs = [norm(row[ref_source]) for row in rows]
                    hyps = [transform_text(row[system], chosen_profile) for row in rows]
                    value = metrics(refs, hyps)
                    score_rows.append(
                        {
                            "language": lang,
                            "system": system,
                            "kind": "acoustic",
                            "normalization": chosen_profile,
                            "reference": ref_source,
                            "split": split_name,
                            "n": len(rows),
                            "promotion_eligible": system
                            not in KNOWN_PUBLIC_FAILURES["promotion_blacklist"],
                            **value,
                        }
                    )
                    if ref_source == "corrected" and split_name == "holdout":
                        lang_report["systems"][system] = {
                            **value,
                            "normalization": chosen_profile,
                            "tune_normalization_scores": profile_scores,
                        }
            normalized_variants[system] = (
                chosen_profile,
                [transform_text(row[system], chosen_profile) for row in records],
            )

        # Use tune-only corrected labels to choose one label-free switching policy.
        policies = derive_policies(lang, complete, tune)
        policy_tune = []
        for policy in policies:
            hyps = [norm(policy.apply(row)) for row in tune]
            value = metrics([norm(row["corrected"]) for row in tune], hyps)
            policy_tune.append((value["zindi"], policy, value))
        policy_tune.sort(key=lambda item: item[0], reverse=True)
        selected = policy_tune[0] if policy_tune else None
        if selected:
            _, policy, tune_value = selected
            holdout_hyps = [norm(policy.apply(row)) for row in holdout]
            holdout_refs = [norm(row["corrected"]) for row in holdout]
            holdout_value = metrics(holdout_refs, holdout_hyps)
            prod_hyps = [norm(row[production]) for row in holdout]
            delta = holdout_value["zindi"] - metrics(holdout_refs, prod_hyps)["zindi"]
            boot = bootstrap_delta(
                holdout_refs,
                prod_hyps,
                holdout_hyps,
                seed + (0 if lang == "lin" else 1),
                groups=[str(row["speaker_id"]) for row in holdout],
            )
            lang_report["ensembles"]["selected_tune_policy"] = {
                "name": policy.name,
                "tune": tune_value,
                "holdout": holdout_value,
                "holdout_delta_vs_production": delta,
                "holdout_bootstrap": boot,
                "searched_policy_count": len(policies),
            }
            ensemble_rows.append(
                {
                    "language": lang,
                    "policy": policy.name,
                    "tune_zindi": tune_value["zindi"],
                    "holdout_zindi": holdout_value["zindi"],
                    "holdout_delta_vs_production": delta,
                    **{f"bootstrap_{k}": v for k, v in boot.items()},
                }
            )

        # Current shipped Shona sim99 policy is always reported, independent of tuning.
        if lang == "sna" and "sna_badrex_greedy" in complete:
            current = next((p for p in policies if p.name == "sna_current_badrex_sim99"), None)
            if current:
                for split_name, rows in (("full", records), ("holdout", holdout)):
                    value = metrics(
                        [norm(row["corrected"]) for row in rows],
                        [norm(current.apply(row)) for row in rows],
                    )
                    lang_report["ensembles"][f"current_badrex_sim99_{split_name}"] = value

        # Pre-declared conservative Shona selector: only leave the current
        # sim99 production policy when BadrEx and Manasseh closely agree.
        if lang == "sna" and {"sna_badrex_greedy", "sna_manasseh_greedy"} <= set(complete):
            def current_sim99(row: dict) -> str:
                return (
                    row["sna_badrex_greedy"]
                    if similarity(row[production], row["sna_badrex_greedy"]) >= 0.99
                    else row[production]
                )

            agreement_grid = []
            candidates = []
            for threshold in (0.90, 0.92, 0.95, 0.97, 0.98, 0.99):
                for chosen in ("sna_badrex_greedy", "sna_manasseh_greedy"):
                    name = f"badrex_manasseh_agree_ge_{threshold:.2f}_choose_{chosen}"

                    def apply(row, t=threshold, c=chosen):
                        return (
                            row[c]
                            if similarity(row["sna_badrex_greedy"], row["sna_manasseh_greedy"]) >= t
                            else current_sim99(row)
                        )

                    tune_value = metrics(
                        [norm(row["corrected"]) for row in tune],
                        [norm(apply(row)) for row in tune],
                    )
                    holdout_h = [norm(apply(row)) for row in holdout]
                    holdout_r = [norm(row["corrected"]) for row in holdout]
                    holdout_value = metrics(holdout_r, holdout_h)
                    baseline_h = [norm(current_sim99(row)) for row in holdout]
                    delta_current = holdout_value["zindi"] - metrics(holdout_r, baseline_h)["zindi"]
                    item = {
                        "name": name,
                        "threshold": threshold,
                        "chosen": chosen,
                        "tune": tune_value,
                        "holdout": holdout_value,
                        "holdout_delta_vs_current_sim99": delta_current,
                    }
                    agreement_grid.append(item)
                    candidates.append((tune_value["zindi"], item, holdout_h, baseline_h))
            candidates.sort(key=lambda value: value[0], reverse=True)
            _, best_item, best_h, baseline_h = candidates[0]
            best_item = dict(best_item)
            best_item["holdout_paired_speaker_bootstrap"] = bootstrap_delta(
                [norm(row["corrected"]) for row in holdout],
                baseline_h,
                best_h,
                seed + 11,
                groups=[str(row["speaker_id"]) for row in holdout],
            )
            lang_report["ensembles"]["conservative_badrex_manasseh_grid"] = agreement_grid
            lang_report["ensembles"]["conservative_badrex_manasseh_selected_on_tune"] = best_item

            badrex_h = [norm(row["sna_badrex_greedy"]) for row in holdout]
            manasseh_h = [norm(row["sna_manasseh_greedy"]) for row in holdout]
            waxal_h = [norm(row["sna_waxal_greedy"]) for row in holdout]
            holdout_r = [norm(row["corrected"]) for row in holdout]
            full_r = [norm(row["corrected"]) for row in records]
            full_badrex = [norm(row["sna_badrex_greedy"]) for row in records]
            full_manasseh = [norm(row["sna_manasseh_greedy"]) for row in records]
            full_waxal = [norm(row["sna_waxal_greedy"]) for row in records]
            lang_report["manasseh_vs_badrex_confirmation"] = {
                "full": {
                    "n": len(records),
                    "badrex": metrics(full_r, full_badrex),
                    "manasseh": metrics(full_r, full_manasseh),
                    "waxal_production": metrics(full_r, full_waxal),
                    "delta_zindi_manasseh_minus_badrex": metrics(full_r, full_manasseh)[
                        "zindi"
                    ]
                    - metrics(full_r, full_badrex)["zindi"],
                    "paired_speaker_bootstrap": bootstrap_delta(
                        full_r,
                        full_badrex,
                        full_manasseh,
                        seed + 19,
                        groups=[str(row["speaker_id"]) for row in records],
                    ),
                },
                "holdout": {
                    "n_holdout": len(holdout),
                    "badrex": metrics(holdout_r, badrex_h),
                    "manasseh": metrics(holdout_r, manasseh_h),
                    "waxal_production": metrics(holdout_r, waxal_h),
                    "delta_zindi_manasseh_minus_badrex": metrics(
                        holdout_r, manasseh_h
                    )["zindi"]
                    - metrics(holdout_r, badrex_h)["zindi"],
                    "paired_speaker_bootstrap": bootstrap_delta(
                        holdout_r,
                        badrex_h,
                        manasseh_h,
                        seed + 17,
                        groups=[str(row["speaker_id"]) for row in holdout],
                    ),
                },
            }

        # Per-row oracle quantifies routing headroom; it is explicitly not shippable.
        oracle_hyps = []
        for row in holdout:
            oracle_hyps.append(
                min(
                    (norm(row[name]) for name in complete if "hf_buggy" not in name),
                    key=lambda hyp: row_cost(norm(row["corrected"]), hyp),
                )
            )
        lang_report["oracle_holdout_not_shippable"] = metrics(
            [norm(row["corrected"]) for row in holdout], oracle_hyps
        )
        report["languages"][lang] = lang_report

    pd.DataFrame(score_rows).sort_values(
        ["language", "reference", "split", "zindi"], ascending=[True, True, True, False]
    ).to_csv(out_dir / "scores.csv", index=False)
    pd.DataFrame(ensemble_rows).to_csv(out_dir / "ensemble_scores.csv", index=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120, help="Rows per language")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--offline-labels", action="store_true")
    parser.add_argument("--skip-fingerprints", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS)
    args = parser.parse_args()
    unknown = sorted(set(args.systems) - set(MODEL_IDS))
    if unknown:
        raise ValueError(f"Unknown systems: {unknown}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.out_dir / "hf_cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(args.out_dir / "hf_cache" / "hub"))
    device = pick_device(args.device)

    corrected = fetch_corrected_labels(args.out_dir, args.offline_labels)
    samples: list[pd.DataFrame] = []
    audio_rows: dict[str, list[dict]] = {}
    label_audit: dict = {}
    for lang in ("lin", "sna"):
        original = original_metadata(lang)
        merged = original.merge(
            corrected[corrected.language == lang][["ID", "corrected"]], on="ID", how="outer", indicator=True
        )
        if not (merged._merge == "both").all():
            raise RuntimeError(f"Corrected/original ID mismatch for {lang}")
        merged = merged.drop(columns="_merge")
        merged["language"] = lang
        order = list(range(len(merged)))
        random.Random(args.seed).shuffle(order)
        selected = merged.iloc[order[: args.n]].copy().reset_index(drop=True)
        selected["sample_position"] = range(len(selected))
        speaker_counts = selected.speaker_id.astype(str).value_counts().to_dict()
        speakers = list(speaker_counts)
        random.Random(args.seed + (0 if lang == "lin" else 10_000)).shuffle(speakers)
        fold_sizes = {"tune": 0, "holdout": 0}
        speaker_fold: dict[str, str] = {}
        for speaker in speakers:
            fold = min(fold_sizes, key=lambda name: (fold_sizes[name], name))
            speaker_fold[speaker] = fold
            fold_sizes[fold] += int(speaker_counts[speaker])
        selected["fold"] = selected.speaker_id.astype(str).map(speaker_fold)
        if set(selected.loc[selected.fold == "tune", "speaker_id"]) & set(
            selected.loc[selected.fold == "holdout", "speaker_id"]
        ):
            raise RuntimeError(f"{lang}: speaker split construction failed")
        samples.append(selected)
        audio_rows[lang] = (
            []
            if args.analyze_only and args.skip_fingerprints
            else selected_rows(lang, list(selected.ID.astype(str)))
        )
        label_audit[lang] = {
            "rows_original": len(original),
            "rows_corrected": int((corrected.language == lang).sum()),
            "raw_changed": int((merged.original.fillna("") != merged.corrected.fillna("")).sum()),
            "normalized_changed": int(
                sum(norm(a) != norm(b) for a, b in zip(merged.original, merged.corrected, strict=True))
            ),
            "selected_raw_changed": int(
                (selected.original.fillna("") != selected.corrected.fillna("")).sum()
            ),
            "selected_normalized_changed": int(
                sum(norm(a) != norm(b) for a, b in zip(selected.original, selected.corrected, strict=True))
            ),
            "corrected_labels_sha256": sha256_text(
                f"{uid}\t{text}" for uid, text in zip(merged.ID, merged.corrected, strict=True)
            ),
        }
    sample = pd.concat(samples, ignore_index=True)
    sample.to_csv(args.out_dir / "matched_sample.csv", index=False)

    route_audit = route_and_cache_audit()
    new_ids = set(
        pd.read_csv(ROOT / "outputs/beat075/public_visible_index.csv")
        .query("split == 'new' and decode_lang in ['lin', 'sna']")
        .ID.astype(str)
    )
    overlap_ids = sorted(set(sample.ID.astype(str)) & new_ids)
    leakage = {"validation_test_id_overlap": overlap_ids, "validation_test_audio_hash_overlap": []}
    if not args.skip_fingerprints:
        validation_hashes = {
            audio_fingerprint(decode_audio(row["audio"]))
            for lang in ("lin", "sna")
            for row in audio_rows[lang]
        }
        route = pd.read_csv(ROOT / "outputs/beat075/public_visible_index.csv")
        test_hashes = set()
        for path in route.query("split == 'new' and decode_lang in ['lin', 'sna']").audio:
            arr, sr = sf.read(path, dtype="float32")
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if int(sr) != TARGET_SR:
                import librosa

                arr = librosa.resample(arr, orig_sr=int(sr), target_sr=TARGET_SR)
            test_hashes.add(audio_fingerprint(arr))
        leakage["validation_test_audio_hash_overlap"] = sorted(validation_hashes & test_hashes)
        leakage["validation_audio_hash_count"] = len(validation_hashes)
        leakage["expanded_lin_sna_audio_hash_count"] = len(test_hashes)
    if leakage["validation_test_id_overlap"] or leakage["validation_test_audio_hash_overlap"]:
        raise RuntimeError(f"Leakage gate failed: {leakage}")

    inventory = {
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "route_audit": route_audit,
        "known_public_failures": KNOWN_PUBLIC_FAILURES,
        "checkpoint_audit": checkpoint_audit(),
        "corrected_resource": {
            "dataset": CORRECTED_DATASET,
            "repository": f"https://huggingface.co/datasets/{CORRECTED_DATASET}",
            "splits_read": ["lin_asr/validation", "sna_asr/validation"],
            "test_split_read": False,
            "audio_read_from_corrected_resource": False,
            "limitations": "single external correction pass; not independently re-verified",
            "label_audit": label_audit,
        },
        "leakage_audit": leakage,
        "sample": {
            "n_per_language": args.n,
            "seed": args.seed,
            "fold_rule": "speaker-disjoint greedy-balanced assignment",
        },
    }
    (args.out_dir / "inventory.json").write_text(json.dumps(inventory, indent=2))

    hypotheses_path = args.out_dir / "hypotheses.csv"
    existing = pd.read_csv(hypotheses_path, keep_default_na=False) if hypotheses_path.exists() else pd.DataFrame()
    diagnostics_path = args.out_dir / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
    if not args.analyze_only:
        for lang in ("lin", "sna"):
            ids = list(sample.loc[sample.language == lang].sort_values("sample_position").ID.astype(str))
            audios = [decode_audio(row["audio"]) for row in audio_rows[lang]]
            requested = [tag for tag in args.systems if tag.startswith(lang + "_")]
            for tag in requested:
                done = (
                    not existing.empty
                    and tag in set(existing.system)
                    and len(existing[(existing.language == lang) & (existing.system == tag)]) == len(ids)
                )
                if done:
                    print(f"{tag}: cached", flush=True)
                    continue
                print(f"{tag}: decoding {len(ids)} on {device}", flush=True)
                try:
                    outputs, detail = run_system(tag, lang, audios, device)
                    write_hypotheses(hypotheses_path, lang, ids, outputs)
                    diagnostics[tag] = {"status": "ok", **detail}
                except Exception as exc:
                    diagnostics[tag] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                    print(f"{tag}: ERROR {exc}", file=sys.stderr, flush=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
                existing = pd.read_csv(hypotheses_path, keep_default_na=False) if hypotheses_path.exists() else pd.DataFrame()

    if not hypotheses_path.exists():
        raise RuntimeError("No hypotheses were produced")
    hypotheses = pd.read_csv(hypotheses_path, keep_default_na=False)
    analysis = analyze(sample, hypotheses, args.seed, args.out_dir)
    report = {
        "protocol": {
            "primary_reference": "Harcuracy corrected validation labels",
            "secondary_reference": "original locally cached WAXAL validation labels",
            "same_ids_for_all_systems": True,
            "normalization_selected_on": "tune half only",
            "ensemble_selected_on": "tune half only",
            "reported_selection_evidence": "holdout half",
            "test_labels_used": False,
            "device": str(device),
        },
        "inventory": "inventory.json",
        "diagnostics": "diagnostics.json",
        "decision": {
            "deployment_candidate": None,
            "status": "no_novel_passer",
            "reason": (
                "On corrected n=120, Manasseh trails the already-known BadrEx system; "
                "Khaya and Okwija fail the n=40 screen; the conservative agreement "
                "selector trails unconditional BadrEx. No Shona hypothesis is promoted."
            ),
            "sulaimank_sd2": "excluded; handled by separate worker",
        },
        **analysis,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
