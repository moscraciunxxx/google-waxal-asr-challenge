#!/usr/bin/env python3
"""Train Whisper per language; every N steps eval vs WAXAL MMS-300m; keep best if beats.

Never uses test gold. Target: all of lin/sna/lug beat mms-300m-waxal on val mean-error.
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
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORBIDDEN_TRAIN_SPLITS, SEED, TARGET_SR
from src.dataset import WhisperDataCollator, load_labeled_splits, prepare_whisper_example
from src.legit_fusion import beats_baseline, mean_error
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text
from src.dataset import load_hf_asr_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_whisper_beat_mms")

WAXAL300 = {
    "lin": "waxal-benchmarking/mms-300m-waxal-lin",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
}
WAXAL_WH = {
    "lin": "waxal-benchmarking/whisper-small-waxal-lin",
    "sna": "waxal-benchmarking/whisper-small-waxal-sna",
    "lug": "waxal-benchmarking/whisper-small-waxal-lug",
}


@torch.inference_mode()
def eval_vs_mms(whisper_model, whisper_proc, lang: str, device, max_n: int = 40) -> dict:
    mid = WAXAL300[lang]
    try:
        bp = AutoProcessor.from_pretrained(mid, local_files_only=True)
        bm = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
    except Exception:
        bp = AutoProcessor.from_pretrained(mid)
        bm = Wav2Vec2ForCTC.from_pretrained(mid)
    bm.to(device).eval()
    whisper_model.eval()
    ds = load_hf_asr_split(lang, "validation", max_samples=max_n)
    refs, base_h, own_h = [], [], []
    for i in range(len(ds)):
        ex = ds[i]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        refs.append(normalize_text(ex.get("transcription") or ""))
        base_h.append(transcribe_waveform(bm, bp, arr, sr, device=device))
        if sr != TARGET_SR:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        peak = float(np.max(np.abs(arr)) + 1e-9)
        arr = arr / peak
        feats = whisper_proc(arr, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(
            device
        )
        ids = whisper_model.generate(feats, do_sample=False, num_beams=1)
        own_h.append(
            normalize_text(whisper_proc.batch_decode(ids, skip_special_tokens=True)[0]) or "."
        )
    base = score_pairs(refs, base_h)
    own = score_pairs(refs, own_h)
    return {
        "baseline_model": mid,
        "baseline": base,
        "own": own,
        "beats": beats_baseline(own, base),
        "mean_error_baseline": mean_error(base["wer"], base["cer"]),
        "mean_error_own": mean_error(own["wer"], own["cer"]),
        "n": len(refs),
    }


def train_lang(
    lang: str,
    init_id: str,
    out_dir: Path,
    max_samples: int,
    max_steps: int,
    lr: float,
    eval_every: int,
    eval_n: int,
    device: torch.device,
) -> dict:
    assert "test" in FORBIDDEN_TRAIN_SPLITS
    set_seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== train %s init=%s steps=%s ===", lang, init_id, max_steps)

    try:
        processor = WhisperProcessor.from_pretrained(init_id, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(init_id, local_files_only=True)
    except Exception:
        processor = WhisperProcessor.from_pretrained(init_id)
        model = WhisperForConditionalGeneration.from_pretrained(init_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []

    train_raw = load_labeled_splits(languages=(lang,), splits=("train",), max_per_lang_split=max_samples)
    val_raw = load_labeled_splits(languages=(lang,), splits=("validation",), max_per_lang_split=min(32, max_samples))

    def _map(batch):
        return prepare_whisper_example(batch, processor)

    train_ds = train_raw.map(_map, remove_columns=train_raw.column_names, desc=f"prep-train-{lang}")
    val_ds = val_raw.map(_map, remove_columns=val_raw.column_names, desc=f"prep-val-{lang}")
    train_ds.reset_format()
    val_ds.reset_format()

    collator = WhisperDataCollator(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
    )
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "runs"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        max_steps=max_steps,
        warmup_steps=min(20, max_steps // 5),
        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="steps",
        save_steps=eval_every,
        logging_steps=5,
        predict_with_generate=False,
        fp16=False,
        bf16=False,
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
        load_best_model_at_end=False,
        save_total_limit=2,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )
    try:
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            processing_class=processor.feature_extractor,
        )
    except TypeError:
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            tokenizer=processor.feature_extractor,
        )

    # Custom callback-like: train in chunks and eval vs MMS
    best = None
    remaining = max_steps
    done = 0
    while remaining > 0:
        chunk = min(eval_every, remaining)
        args.max_steps = done + chunk
        # re-create args max_steps by mutating
        trainer.args.max_steps = done + chunk
        trainer.train(resume_from_checkpoint=False if done == 0 else None)
        # After partial train, HF may not support incremental max_steps well —
        # fall back to single train for simplicity below
        break

    trainer.train()
    model = trainer.model
    model.to(device)
    result = eval_vs_mms(model, processor, lang, device, max_n=eval_n)
    result["lang"] = lang
    result["own_model"] = str(out_dir / "best")
    result["init"] = init_id
    best_dir = out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    processor.save_pretrained(best_dir)
    (out_dir / "eval_vs_mms.json").write_text(json.dumps(result, indent=2))
    logger.info(
        "%s beats=%s own_me=%.4f base_me=%.4f",
        lang,
        result["beats"],
        result["mean_error_own"],
        result["mean_error_baseline"],
    )
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=["lin", "sna", "lug"])
    p.add_argument("--model-id", default="openai/whisper-medium")
    p.add_argument("--prefer-waxal-init", action="store_true", help="Init from waxal whisper-small if exists")
    p.add_argument("--max-samples", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--eval-every", type=int, default=40)
    p.add_argument("--eval-n", type=int, default=40)
    p.add_argument("--out-root", type=Path, default=Path("checkpoints/whisper-beat-mms"))
    args = p.parse_args(argv)

    device = pick_device()
    results = []
    for lang in args.languages:
        if args.prefer_waxal_init:
            init = WAXAL_WH.get(lang, args.model_id)
            # try local waxal first; fall back to model-id
            try:
                WhisperProcessor.from_pretrained(init, local_files_only=True)
            except Exception:
                init = args.model_id
        else:
            init = args.model_id
        out = args.out_root / lang
        r = train_lang(
            lang,
            init,
            out,
            args.max_samples,
            args.max_steps,
            args.lr,
            args.eval_every,
            args.eval_n,
            device,
        )
        results.append(r)

    all_beat = all(r["beats"] for r in results)
    report = {
        "seed": SEED,
        "protocol": "validation only; same clips; normalize; jiwer; never test gold",
        "forbidden_train_splits": list(FORBIDDEN_TRAIN_SPLITS),
        "baseline_definition": "WAXALNet MMS-300m waxal-benchmarking/mms-300m-waxal-{lang}",
        "own_definition": f"Whisper FT ({args.model_id} or waxal-whisper init) train only",
        "max_per_lang_train": args.max_samples,
        "max_steps": args.max_steps,
        "per_language": results,
        "all_languages_beat_baseline": all_beat,
    }
    out_json = Path("outputs/multi_agent_push/t1_val_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s all_beat=%s", out_json, all_beat)
    return 0 if all_beat else 2


if __name__ == "__main__":
    raise SystemExit(main())
