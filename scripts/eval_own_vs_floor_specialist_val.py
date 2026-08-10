#!/usr/bin/env python3
"""Same-protocol val: floor-era / local specialist vs own Whisper-class checkpoint.

Baseline priority per lang (floor-related specialists):
  lug → checkpoints/mms-lug-ft-v3 if present else waxal mms-300m
  lin → checkpoints/mms-lin-ft-v2 if present else waxal mms-300m
  sna → checkpoints/mms-sna-ft-v2 if present else waxal mms-300m

Own: per-lang Whisper under checkpoints/whisper-per-lang-legit/{lang}/best
  or continued multi-lang, or WAXALNet whisper-small after local continued FT.

Never test gold.
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

from src.config import CHECKPOINT_DIR, FORBIDDEN_TRAIN_SPLITS, OUTPUT_DIR, SEED, TARGET_SR
from src.dataset import load_hf_asr_split
from src.legit_fusion import beats_baseline, mean_error
from src.metrics import score_pairs
from src.mms_infer import pick_device, set_lang, transcribe_waveform
from src.text_norm import normalize_text
from src.train import set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_own_vs_floor")

WAXAL300 = {
    "lin": "waxal-benchmarking/mms-300m-waxal-lin",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
}
FLOOR_SPEC = {
    "lin": CHECKPOINT_DIR / "mms-lin-ft-v2",
    "sna": CHECKPOINT_DIR / "mms-sna-ft-v2",
    "lug": CHECKPOINT_DIR / "mms-lug-ft-v3",
}


def resolve_baseline(lang: str) -> tuple[str, str]:
    """Return (kind, model_id_or_path). Prefer floor local FT when present."""
    local = FLOOR_SPEC[lang]
    if local.exists() and (local / "config.json").exists():
        return "floor_local_mms_ft", str(local)
    return "waxal_mms300", WAXAL300[lang]


def resolve_own(lang: str, ckpt_root: Path, allow_waxal_wh: bool) -> str:
    own = ckpt_root / lang / "best"
    if own.exists():
        return str(own)
    multi = CHECKPOINT_DIR / "whisper-waxal-legit-p2-serious" / "best"
    if multi.exists():
        return str(multi)
    multi2 = CHECKPOINT_DIR / "whisper-waxal-legit-p2" / "best"
    if multi2.exists():
        return str(multi2)
    if allow_waxal_wh:
        return f"waxal-benchmarking/whisper-small-waxal-{lang}"
    raise FileNotFoundError(f"No own whisper ckpt for {lang}")


@torch.inference_mode()
def load_ctc(path: str, lang: str, device):
    try:
        proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=True)
    except Exception:
        proc = AutoProcessor.from_pretrained(path)
        model = Wav2Vec2ForCTC.from_pretrained(path)
    model.to(device).eval()
    # local FT may already bake adapter; try set_lang for mms-1b style
    try:
        set_lang(model, proc, lang)
    except Exception:
        pass
    return model, proc


@torch.inference_mode()
def load_whisper(path: str, device):
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
    return model, proc


@torch.inference_mode()
def decode_wh(model, proc, arr, sr, device, lang: str):
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt")
    feats = inputs.input_features.to(device)
    ids = model.generate(feats, do_sample=False, num_beams=1)
    return normalize_text(proc.batch_decode(ids, skip_special_tokens=True)[0]) or "."


def eval_lang(lang: str, max_n: int | None, own_path: str, device) -> dict:
    kind, base_path = resolve_baseline(lang)
    ds = load_hf_asr_split(lang, "validation", max_samples=max_n)
    b_model, b_proc = load_ctc(base_path, lang, device)
    o_model, o_proc = load_whisper(own_path, device)

    refs, base_hyps, own_hyps = [], [], []
    for i in range(len(ds)):
        ex = ds[i]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        refs.append(normalize_text(ex.get("transcription") or ""))
        base_hyps.append(transcribe_waveform(b_model, b_proc, arr, sr, device=device))
        own_hyps.append(decode_wh(o_model, o_proc, arr, sr, device, lang))
        if (i + 1) % 15 == 0:
            logger.info("%s %d/%d", lang, i + 1, len(ds))

    base = score_pairs(refs, base_hyps)
    own = score_pairs(refs, own_hyps)
    return {
        "lang": lang,
        "n": len(refs),
        "split": "validation",
        "baseline_kind": kind,
        "baseline_model": base_path,
        "own_model": own_path,
        "baseline": base,
        "own": own,
        "beats": beats_baseline(own, base),
        "mean_error_baseline": mean_error(base["wer"], base["cer"]),
        "mean_error_own": mean_error(own["wer"], own["cer"]),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-per-lang", type=int, default=50)
    p.add_argument("--ckpt-root", type=Path, default=CHECKPOINT_DIR / "whisper-per-lang-legit")
    p.add_argument("--allow-waxal-whisper-fallback", action="store_true")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "multi_agent_push" / "t1_val_report.json")
    args = p.parse_args(argv)
    set_all_seeds(SEED)
    device = pick_device()
    assert "test" in FORBIDDEN_TRAIN_SPLITS

    results = []
    for lang in ("lin", "sna", "lug"):
        own = resolve_own(lang, args.ckpt_root, args.allow_waxal_whisper_fallback)
        logger.info("=== %s own=%s ===", lang, own)
        results.append(eval_lang(lang, args.max_per_lang, own, device))

    all_beat = all(r["beats"] for r in results)
    report = {
        "seed": SEED,
        "protocol": "validation only; floor specialist or WAXALNet MMS; never test gold",
        "forbidden_train_splits": list(FORBIDDEN_TRAIN_SPLITS),
        "max_per_lang": args.max_per_lang,
        "per_language": results,
        "all_languages_beat_baseline": all_beat,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s all_beat=%s", args.out, all_beat)
    for r in results:
        logger.info(
            "%s base=%s me_b=%.4f me_o=%.4f beats=%s",
            r["lang"],
            r["baseline_kind"],
            r["mean_error_baseline"],
            r["mean_error_own"],
            r["beats"],
        )
    return 0 if all_beat else 2


if __name__ == "__main__":
    raise SystemExit(main())
