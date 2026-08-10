"""Inference: language-conditioned (Phase 1) and audio-only (Phase 2)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset, load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import (
    HF_CONFIGS,
    HF_DATASET,
    LANGUAGES,
    MAX_LABEL_LENGTH,
    MAX_AUDIO_SECONDS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SEED,
    TARGET_SR,
)
from src.dataset import load_hf_asr_split
from src.text_norm import normalize_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("infer")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    checkpoint: str | Path,
    device: torch.device | None = None,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, torch.device]:
    device = device or pick_device()
    checkpoint = str(checkpoint)
    processor = WhisperProcessor.from_pretrained(checkpoint)
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.to(device)
    model.eval()
    return model, processor, device


def _prep_audio(array: np.ndarray, sr: int, max_seconds: float = MAX_AUDIO_SECONDS) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if sr != TARGET_SR:
        # datasets Audio cast usually handles this; simple resample fallback
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    max_len = int(max_seconds * TARGET_SR)
    if array.shape[0] > max_len:
        array = array[:max_len]
    return array


@torch.inference_mode()
def transcribe_array(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    array: np.ndarray,
    sr: int = TARGET_SR,
    device: torch.device | None = None,
    language: str | None = None,
    num_beams: int = 5,
) -> str:
    """Transcribe one waveform.

    language=None → audio-only / Phase-2 mode (no language or speaker metadata).
    language set → optional language hint when Whisper supports it; for lin/sna/lug
    we still decode without forced language tokens because Whisper lacks them —
    the fine-tuned weights carry the languages.
    """
    device = device or next(model.parameters()).device
    array = _prep_audio(array, sr)
    inputs = processor.feature_extractor(
        array, sampling_rate=TARGET_SR, return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)

    # Scale the generation budget with audio duration. A fixed 64-token cap
    # truncates valid 20–30 second transcripts.
    duration_s = float(len(array)) / TARGET_SR
    gen_kwargs = {
        "num_beams": max(1, int(num_beams)),
        "max_new_tokens": min(MAX_LABEL_LENGTH, max(64, int(duration_s * 8.0) + 32)),
        "do_sample": False,
        "use_cache": True,
    }
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = gen_kwargs["max_new_tokens"]
    if language:
        # Whisper accepts ISO-639-1 language hints. WAXAL uses ISO-639-3
        # codes, so map only the supported challenge languages. If a model
        # tokenizer lacks a token, keep the robust unconstrained decode.
        hint = {"lin": "ln", "sna": "sn", "lug": "lg"}.get(language, language)
        try:
            gen_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                language=hint, task="transcribe"
            )
        except (KeyError, ValueError, TypeError):
            logger.debug("No Whisper language prompt available for %s", language)
    predicted_ids = model.generate(input_features, **gen_kwargs)
    text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return normalize_text(text)


def transcribe_batch_from_hf(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    languages: Iterable[str],
    split: str,
    device: torch.device,
    max_samples: int | None = None,
    num_beams: int = 5,
    audio_only: bool = False,
) -> pd.DataFrame:
    """Run inference over HF split(s); return ID, language, prediction [, reference]."""
    rows = []
    for lang in languages:
        # Reuse shared loader (streams when max_samples is small)
        ds = load_hf_asr_split(
            lang,
            split,
            max_samples=max_samples,
            allow_test=(split == "test"),
        )
        n = len(ds)
        logger.info("Infer %s/%s n=%d audio_only=%s", lang, split, n, audio_only)
        for i in tqdm(range(n), desc=f"{lang}-{split}"):
            ex = ds[i]
            array = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"]["sampling_rate"])
            # Phase-2 must be audio-only; Phase-1 diagnostics may use the
            # supplied language hint explicitly.
            decode_language = None if audio_only else (ex.get("language") or lang)
            pred = transcribe_array(
                model,
                processor,
                array,
                sr,
                device=device,
                language=decode_language,
                num_beams=num_beams,
            )
            row = {
                "ID": ex["id"],
                "language": lang,
                "prediction": pred if pred else " ",
            }
            if "transcription" in ex and ex["transcription"] is not None:
                row["reference"] = normalize_text(ex["transcription"])
            rows.append(row)
    return pd.DataFrame(rows)


def run_predict_test(
    checkpoint: str | Path,
    out_csv: Path | None = None,
    max_per_lang: int | None = None,
    audio_only: bool = False,
    num_beams: int = 5,
) -> Path:
    """Generate predictions for the Phase-1 test split → submission-ready frame."""
    model, processor, device = load_model(checkpoint)
    df = transcribe_batch_from_hf(
        model,
        processor,
        languages=LANGUAGES,
        split="test",
        device=device,
        max_samples=max_per_lang,
        num_beams=num_beams,
        audio_only=audio_only,
    )
    out_csv = Path(out_csv or (OUTPUT_DIR / "test_predictions.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_csv, len(df))
    return out_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WAXAL ASR inference")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-per-lang", type=int, default=None)
    p.add_argument("--audio-only", action="store_true", help="Phase-2 mode: ignore language metadata")
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    args = parse_args(argv)
    model, processor, device = load_model(args.checkpoint)
    df = transcribe_batch_from_hf(
        model,
        processor,
        languages=tuple(args.languages),
        split=args.split,
        device=device,
        max_samples=args.max_per_lang,
        num_beams=args.num_beams,
        audio_only=args.audio_only,
    )
    out = args.out or (OUTPUT_DIR / f"preds_{args.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
