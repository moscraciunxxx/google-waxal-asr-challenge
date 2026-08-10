#!/usr/bin/env python3
"""Same-protocol val: own checkpoint vs WAXALNet MMS-300m specialists (criterion-1 baseline).

Never loads test gold.
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

from src.config import FORBIDDEN_TRAIN_SPLITS, OUTPUT_DIR, SEED, TARGET_SR
from src.dataset import load_hf_asr_split
from src.legit_fusion import beats_baseline, mean_error
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text
from src.train import set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_vs_mms300")

WAXAL300 = {
    "lin": "waxal-benchmarking/mms-300m-waxal-lin",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
}


def load_own(path: str, device: torch.device):
    path_l = path.lower()
    if "whisper" in path_l or Path(path).name.startswith("whisper") or "whisper" in str(Path(path).parts):
        try:
            proc = WhisperProcessor.from_pretrained(path, local_files_only=True)
            model = WhisperForConditionalGeneration.from_pretrained(path, local_files_only=True)
        except Exception:
            proc = WhisperProcessor.from_pretrained(path)
            model = WhisperForConditionalGeneration.from_pretrained(path)
        model.to(device).eval()
        model.config.forced_decoder_ids = None
        if hasattr(model, "generation_config"):
            model.generation_config.forced_decoder_ids = None
        return "whisper", model, proc
    try:
        proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=True)
    except Exception:
        proc = AutoProcessor.from_pretrained(path)
        model = Wav2Vec2ForCTC.from_pretrained(path)
    model.to(device).eval()
    return "ctc", model, proc


@torch.inference_mode()
def decode_own(kind, model, proc, arr, sr, device, lang: str) -> str:
    if kind == "ctc":
        return transcribe_waveform(model, proc, arr, sr, device=device)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt")
    ids = model.generate(inputs.input_features.to(device), do_sample=False, num_beams=1)
    return normalize_text(proc.batch_decode(ids, skip_special_tokens=True)[0]) or "."


def eval_lang(lang: str, own_path: str, max_n: int | None, device: torch.device) -> dict:
    mid = WAXAL300[lang]
    ds = load_hf_asr_split(lang, "validation", max_samples=max_n)
    try:
        b_proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
        b_model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
    except Exception:
        b_proc = AutoProcessor.from_pretrained(mid)
        b_model = Wav2Vec2ForCTC.from_pretrained(mid)
    b_model.to(device).eval()
    kind, o_model, o_proc = load_own(own_path, device)

    refs, base_hyps, own_hyps = [], [], []
    for i in range(len(ds)):
        ex = ds[i]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        refs.append(normalize_text(ex.get("transcription") or ""))
        base_hyps.append(transcribe_waveform(b_model, b_proc, arr, sr, device=device))
        own_hyps.append(decode_own(kind, o_model, o_proc, arr, sr, device, lang))
        if (i + 1) % 20 == 0:
            logger.info("%s %d/%d", lang, i + 1, len(ds))

    base = score_pairs(refs, base_hyps)
    own = score_pairs(refs, own_hyps)
    return {
        "lang": lang,
        "n": len(refs),
        "split": "validation",
        "baseline_model": mid,
        "own_model": own_path,
        "own_kind": kind,
        "baseline": base,
        "own": own,
        "beats": beats_baseline(own, base),
        "mean_error_baseline": mean_error(base["wer"], base["cer"]),
        "mean_error_own": mean_error(own["wer"], own["cer"]),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-per-lang", type=int, default=50)
    p.add_argument(
        "--own-map",
        type=str,
        default="",
        help="JSON map lang->ckpt path; default whisper-per-lang-legit",
    )
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "multi_agent_push" / "t1_val_report.json")
    args = p.parse_args(argv)
    set_all_seeds(SEED)
    device = pick_device()
    assert "test" in FORBIDDEN_TRAIN_SPLITS

    if args.own_map:
        own_map = json.loads(args.own_map)
    else:
        own_map = {
            l: f"checkpoints/whisper-per-lang-legit/{l}/best" for l in ("lin", "sna", "lug")
        }

    results = []
    for lang in ("lin", "sna", "lug"):
        logger.info("=== %s own=%s ===", lang, own_map[lang])
        results.append(eval_lang(lang, own_map[lang], args.max_per_lang, device))

    all_beat = all(r["beats"] for r in results)
    report = {
        "seed": SEED,
        "protocol": "validation only; same clips; normalize; jiwer; never test gold",
        "forbidden_train_splits": list(FORBIDDEN_TRAIN_SPLITS),
        "baseline_definition": "WAXALNet MMS-300m language specialist waxal-benchmarking/mms-300m-waxal-{lang}",
        "max_per_lang": args.max_per_lang,
        "per_language": results,
        "all_languages_beat_baseline": all_beat,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s all_beat=%s", args.out, all_beat)
    for r in results:
        logger.info(
            "%s base_me=%.4f own_me=%.4f beats=%s base_model=%s",
            r["lang"],
            r["mean_error_baseline"],
            r["mean_error_own"],
            r["beats"],
            r["baseline_model"],
        )
    return 0 if all_beat else 2


if __name__ == "__main__":
    raise SystemExit(main())
