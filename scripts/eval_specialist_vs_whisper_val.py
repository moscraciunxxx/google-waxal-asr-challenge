#!/usr/bin/env python3
"""Same-protocol val WER/CER: WAXALNet MMS-300m specialist vs Whisper checkpoint.

Never loads HF test gold for thr-tuning. Train is not used here — validation only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORBIDDEN_TRAIN_SPLITS, LANGUAGES, OUTPUT_DIR, PROJECT_ROOT, SEED, TARGET_SR
from src.dataset import load_hf_asr_split
from src.legit_fusion import beats_baseline, mean_error
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_specialist_vs_whisper")

WAXAL300 = {
    "lin": "waxal-benchmarking/mms-300m-waxal-lin",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
}


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.inference_mode()
def decode_mms(model, processor, array, sr, device):
    return transcribe_waveform(model, processor, array, sr, device=device, return_confidence=False)


@torch.inference_mode()
def decode_whisper(model, processor, array, sr, device, lang_hint: str | None = None):
    if sr != TARGET_SR:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
    inputs = processor(array, sampling_rate=TARGET_SR, return_tensors="pt")
    feats = inputs.input_features.to(device)
    gen_kwargs = {"do_sample": False, "num_beams": 1}
    name_map = {"lin": "lingala", "sna": "shona", "lug": "swahili"}
    if lang_hint in name_map and "whisper" in getattr(model.config, "_name_or_path", "").lower() or True:
        try:
            if hasattr(processor, "get_decoder_prompt_ids") and lang_hint in name_map:
                gen_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                    language=name_map[lang_hint], task="transcribe"
                )
        except Exception:
            pass
    ids = model.generate(feats, **gen_kwargs)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return normalize_text(text) or "."


def eval_lang(lang: str, max_samples: int | None, whisper_ckpt: str, device: torch.device) -> dict:
    if "test" in FORBIDDEN_TRAIN_SPLITS:
        pass  # validation only below
    ds = load_hf_asr_split(lang, "validation", max_samples=max_samples)
    mid = WAXAL300[lang]
    try:
        mms_proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
        mms_model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
    except Exception:
        mms_proc = AutoProcessor.from_pretrained(mid)
        mms_model = Wav2Vec2ForCTC.from_pretrained(mid)
    mms_model.to(device).eval()

    try:
        wh_proc = WhisperProcessor.from_pretrained(whisper_ckpt, local_files_only=True)
        wh_model = WhisperForConditionalGeneration.from_pretrained(whisper_ckpt, local_files_only=True)
    except Exception:
        wh_proc = WhisperProcessor.from_pretrained(whisper_ckpt)
        wh_model = WhisperForConditionalGeneration.from_pretrained(whisper_ckpt)
    wh_model.to(device).eval()
    wh_model.config.forced_decoder_ids = None
    if hasattr(wh_model, "generation_config"):
        wh_model.generation_config.forced_decoder_ids = None

    refs, mms_hyps, wh_hyps = [], [], []
    n = len(ds)
    for i in range(n):
        ex = ds[i]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        ref = normalize_text(ex.get("transcription") or "")
        refs.append(ref)
        mms_hyps.append(decode_mms(mms_model, mms_proc, arr, sr, device))
        wh_hyps.append(decode_whisper(wh_model, wh_proc, arr, sr, device, lang_hint=lang))
        if (i + 1) % 20 == 0:
            logger.info("%s %d/%d", lang, i + 1, n)

    base = score_pairs(refs, mms_hyps)
    own = score_pairs(refs, wh_hyps)
    return {
        "lang": lang,
        "n": n,
        "split": "validation",
        "baseline_model": mid,
        "own_model": whisper_ckpt,
        "baseline": base,
        "own": own,
        "beats": beats_baseline(own, base),
        "mean_error_baseline": mean_error(base["wer"], base["cer"]),
        "mean_error_own": mean_error(own["wer"], own["cer"]),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--whisper-ckpt", default="checkpoints/whisper-waxal-legit-p2/best")
    p.add_argument("--max-per-lang", type=int, default=80)
    p.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    p.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR / "multi_agent_push" / "t1_val_report.json",
    )
    args = p.parse_args(argv)
    set_seed(SEED)
    device = pick_device()
    logger.info("device=%s whisper=%s max_per_lang=%s", device, args.whisper_ckpt, args.max_per_lang)

    results = []
    for lang in args.languages:
        logger.info("=== eval %s ===", lang)
        results.append(eval_lang(lang, args.max_per_lang, args.whisper_ckpt, device))

    all_beat = all(r["beats"] for r in results)
    report = {
        "seed": SEED,
        "protocol": "WAXAL validation only; same clips; normalize_text; jiwer; never test gold",
        "forbidden_train_splits": list(FORBIDDEN_TRAIN_SPLITS),
        "whisper_ckpt": args.whisper_ckpt,
        "max_per_lang": args.max_per_lang,
        "per_language": results,
        "all_languages_beat_baseline": all_beat,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s all_beat=%s", args.out, all_beat)
    for r in results:
        logger.info(
            "%s n=%d base_me=%.4f own_me=%.4f beats=%s base_wer=%.4f own_wer=%.4f",
            r["lang"],
            r["n"],
            r["mean_error_baseline"],
            r["mean_error_own"],
            r["beats"],
            r["baseline"]["wer"],
            r["own"]["wer"],
        )
    return 0 if all_beat else 2


if __name__ == "__main__":
    raise SystemExit(main())
