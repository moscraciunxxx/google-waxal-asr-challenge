"""Meta MMS (facebook/mms-1b-all) language-adapter inference for lin/sna/lug.

Phase 1: use language id from Test.csv / HF metadata.
Phase 2: run all three adapters and pick by CTC confidence (max mean logprob).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LANGUAGES, OUTPUT_DIR, PROJECT_ROOT, SAMPLE_SUBMISSION_CSV, TARGET_SR, TEST_CSV
from src.dataset import load_hf_asr_split
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mms_infer")

MMS_MODEL_ID = "facebook/mms-1b-all"
# ISO 639-3 codes used by MMS. Challenge langs + Phase-2 open-set extras.
# Unknown keys fall through to the raw code in set_lang (MMS has 1000+ adapters).
MMS_LANG = {
    "lin": "lin",
    "sna": "sna",
    "lug": "lug",
    "luo": "luo",
    "ach": "ach",
    "nyn": "nyn",
    "kam": "kam",
    "umb": "umb",
    "nya": "nya",
    "swh": "swh",
    "nso": "nso",
    "wol": "wol",
}


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_mms(model_id: str = MMS_MODEL_ID, device: torch.device | None = None):
    device = device or pick_device()
    logger.info("Loading %s on %s", model_id, device)
    # Prefer local cache when offline / already downloaded
    try:
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(model_id, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, processor, device


def set_lang(model, processor, lang: str) -> None:
    code = MMS_LANG.get(lang, lang)
    # Fine-tuned full checkpoints already bake in the adapter; loading again can fail.
    try:
        processor.tokenizer.set_target_lang(code)
    except Exception as e:
        logger.warning("set_target_lang(%s) failed: %s", code, e)
    # MMS quirk: after set_target_lang, convert_tokens_to_ids("|") can return id for 'a'.
    # Scan vocab for the real '|' token id and force word_delimiter.
    tok = processor.tokenizer
    try:
        pipe_id = None
        for i in range(len(tok)):
            if tok.convert_ids_to_tokens(i) == "|":
                pipe_id = i
                break
        if pipe_id is not None:
            tok.word_delimiter_token = "|"
            tok.word_delimiter_token_id = int(pipe_id)
            if hasattr(tok, "_word_delimiter_token"):
                tok._word_delimiter_token = "|"
        else:
            logger.warning("word_delimiter: no '|' in vocab for lang=%s", code)
    except Exception as e:
        logger.warning("word_delimiter fix failed: %s", e)
    try:
        # Prefer local cache — avoids HF HEAD on every language switch (10x slowdown).
        try:
            model.load_adapter(code, local_files_only=True)
        except TypeError:
            model.load_adapter(code)
        except Exception:
            model.load_adapter(code)
    except Exception as e:
        logger.info("load_adapter(%s) skipped/failed (ok for FT ckpt): %s", code, e)


@torch.inference_mode()
def transcribe_waveform(
    model,
    processor,
    array: np.ndarray,
    sr: int = TARGET_SR,
    device: torch.device | None = None,
    return_confidence: bool = False,
):
    device = device or next(model.parameters()).device
    array = np.asarray(array, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
    # light peak normalize
    peak = float(np.max(np.abs(array)) + 1e-9)
    if peak > 0:
        array = array / peak

    inputs = processor(array, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    logits = model(input_values).logits  # [1, T, V]
    pred_ids = torch.argmax(logits, dim=-1)[0]
    text = processor.decode(pred_ids)
    text = normalize_text(text) or "."

    if not return_confidence:
        return text

    # Mean log-prob of argmax path (higher is better)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0]
    conf = float(log_probs.max(dim=-1).values.mean().item())
    return text, conf


def predict_split_lang(
    model,
    processor,
    device,
    lang: str,
    split: str = "test",
    max_samples: int | None = None,
) -> pd.DataFrame:
    set_lang(model, processor, lang)
    ds = load_hf_asr_split(
        lang, split, max_samples=max_samples, allow_test=(split == "test")
    )
    rows = []
    for i in tqdm(range(len(ds)), desc=f"mms-{lang}-{split}"):
        ex = ds[i]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        hyp = transcribe_waveform(model, processor, arr, sr, device=device)
        row = {"ID": ex["id"], "language": lang, "prediction": hyp}
        if ex.get("transcription") is not None:
            row["reference"] = normalize_text(ex["transcription"])
        rows.append(row)
    return pd.DataFrame(rows)


def predict_phase2_audio_only(
    model,
    processor,
    device,
    lang_hint_order: tuple[str, ...] = LANGUAGES,
    split: str = "validation",
    max_samples: int | None = 8,
) -> pd.DataFrame:
    """Decode with each adapter; pick highest CTC confidence."""
    # Load audio once without adapter assumption
    # Use first language only to fetch clips; ids shared across configs by language files.
    # For multi-lang mixed set we'd need a different index; for smoke use per-lang.
    frames = []
    for lang in lang_hint_order:
        ds = load_hf_asr_split(
            lang, split, max_samples=max_samples, allow_test=(split == "test")
        )
        for i in range(len(ds)):
            ex = ds[i]
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"]["sampling_rate"])
            best_text, best_conf, best_lang = ".", -1e9, lang
            for cand in lang_hint_order:
                set_lang(model, processor, cand)
                text, conf = transcribe_waveform(
                    model, processor, arr, sr, device=device, return_confidence=True
                )
                if conf > best_conf:
                    best_text, best_conf, best_lang = text, conf, cand
            frames.append(
                {
                    "ID": ex["id"],
                    "language": lang,
                    "prediction": best_text,
                    "chosen_lang": best_lang,
                    "confidence": best_conf,
                }
            )
    return pd.DataFrame(frames)


def run_test_submission(
    out_csv: Path | None = None,
    languages: tuple[str, ...] = LANGUAGES,
    max_per_lang: int | None = None,
    model_id: str = MMS_MODEL_ID,
) -> Path:
    model, processor, device = load_mms(model_id)
    frames = []
    for lang in languages:
        frames.append(
            predict_split_lang(
                model, processor, device, lang, split="test", max_samples=max_per_lang
            )
        )
    preds = pd.concat(frames, ignore_index=True)
    out_preds = OUTPUT_DIR / "mms_test_predictions.csv"
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_preds, index=False)

    out_csv = Path(out_csv or (PROJECT_ROOT / "submission.csv"))
    build_submission(preds, sample_path=SAMPLE_SUBMISSION_CSV, out_path=out_csv)
    report = check_submission(out_csv, SAMPLE_SUBMISSION_CSV)
    logger.info("submission check: %s", report)
    if not report["ok"]:
        raise RuntimeError(report)
    return out_csv


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="MMS ASR inference for WAXAL")
    p.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    p.add_argument("--split", default="test")
    p.add_argument("--max-per-lang", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--phase2-smoke", action="store_true")
    p.add_argument("--model-id", default=MMS_MODEL_ID)
    args = p.parse_args(argv)

    if args.phase2_smoke:
        model, processor, device = load_mms(args.model_id)
        df = predict_phase2_audio_only(
            model, processor, device, max_samples=args.max_per_lang or 4
        )
        out = args.out or (OUTPUT_DIR / "mms_phase2_smoke.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(df)
        return

    if args.split == "test":
        run_test_submission(
            out_csv=args.out,
            languages=tuple(args.languages),
            max_per_lang=args.max_per_lang,
            model_id=args.model_id,
        )
        return

    model, processor, device = load_mms(args.model_id)
    frames = [
        predict_split_lang(
            model, processor, device, lang, split=args.split, max_samples=args.max_per_lang
        )
        for lang in args.languages
    ]
    df = pd.concat(frames, ignore_index=True)
    out = args.out or (OUTPUT_DIR / f"mms_{args.split}_preds.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if "reference" in df.columns:
        from src.metrics import score_by_language

        print(score_by_language(df.reference.tolist(), df.prediction.tolist(), df.language.tolist()))


if __name__ == "__main__":
    main()
