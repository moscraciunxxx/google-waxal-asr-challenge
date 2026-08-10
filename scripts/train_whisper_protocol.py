#!/usr/bin/env python3
"""Whisper-class FT under criterion1 protocol (select/report split).

Train on train only. Checkpoint schedule pick uses SELECT val[50:90] only.
CRITERION1 scoreboard: ONE report val[0:50] eval after selection.
Never test gold. Never baseline soup.
"""
from __future__ import annotations
import argparse, json, logging, os, random, sys
from pathlib import Path
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
from transformers import (
    AutoProcessor, Wav2Vec2ForCTC,
    WhisperForConditionalGeneration, WhisperProcessor,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed,
)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import FORBIDDEN_TRAIN_SPLITS, SEED, TARGET_SR
from src.criterion1_protocol import (
    REPORT_SLICE_LABEL, SELECT_SLICE_LABEL, WAXAL300,
    split_val_protocol, CheckpointCandidate, select_checkpoint_by_slice,
)
from src.dataset import WhisperDataCollator, load_labeled_splits, prepare_whisper_example, load_hf_asr_split
from src.legit_fusion import beats_baseline, mean_error
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("whisper_proto")

WAXAL_WH = {
    "lin": "waxal-benchmarking/whisper-small-waxal-lin",
    "sna": "waxal-benchmarking/whisper-small-waxal-sna",
    "lug": "waxal-benchmarking/whisper-small-waxal-lug",
}


@torch.inference_mode()
def eval_indices_whisper(w_model, w_proc, b_model, b_proc, ds, indices, device):
    w_model.eval()
    refs, bh, oh = [], [], []
    for i in indices:
        ex = ds[int(i)]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        refs.append(normalize_text(ex.get("transcription") or ""))
        bh.append(transcribe_waveform(b_model, b_proc, arr, sr, device=device))
        if sr != TARGET_SR:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        peak = float(np.max(np.abs(arr)) + 1e-9)
        arr = arr / peak
        feats = w_proc(arr, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(device)
        ids = w_model.generate(feats, do_sample=False, num_beams=1, max_new_tokens=128)
        oh.append(normalize_text(w_proc.batch_decode(ids, skip_special_tokens=True)[0]) or ".")
    base = score_pairs(refs, bh)
    own = score_pairs(refs, oh)
    return base, own, beats_baseline(own, base)


def train_lang(lang: str, init_id: str, out_dir: Path, max_samples: int, max_steps: int, lr: float, device) -> dict:
    assert "test" in FORBIDDEN_TRAIN_SPLITS
    set_seed(SEED)
    proto = split_val_protocol()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== Whisper FT %s init=%s steps=%s select=%s ===", lang, init_id, max_steps, proto.select_slice)

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
    def _map(batch):
        return prepare_whisper_example(batch, processor)
    train_ds = train_raw.map(_map, remove_columns=train_raw.column_names, desc=f"prep-{lang}")
    train_ds.reset_format()
    collator = WhisperDataCollator(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)

    # FIXED final training — single train run; no intermediate report eval
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "runs"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        max_steps=max_steps,
        warmup_steps=min(10, max_steps // 5),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=10,
        predict_with_generate=False,
        fp16=False, bf16=False,
        report_to=[],
        seed=SEED, data_seed=SEED,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )
    try:
        trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, data_collator=collator, processing_class=processor.feature_extractor)
    except TypeError:
        trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, data_collator=collator, tokenizer=processor.feature_extractor)
    trainer.train()
    model = trainer.model
    model.to(device)

    # SELECT-only score of FINAL weights (fixed schedule, no step hunt)
    mid = WAXAL300[lang]
    try:
        bp = AutoProcessor.from_pretrained(mid, local_files_only=True)
        bm = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
    except Exception:
        bp = AutoProcessor.from_pretrained(mid)
        bm = Wav2Vec2ForCTC.from_pretrained(mid)
    bm.to(device).eval()
    need = max(proto.report_indices[-1], proto.select_indices[-1]) + 1
    val = load_hf_asr_split(lang, "validation", max_samples=need)
    b_s, o_s, beat_s = eval_indices_whisper(model, processor, bm, bp, val, proto.select_indices, device)
    me_s = mean_error(o_s["wer"], o_s["cer"])
    log.info("%s SELECT_me=%.4f base=%.4f beats_select=%s", lang, me_s, mean_error(b_s["wer"], b_s["cer"]), beat_s)

    # ONE report eval
    b_r, o_r, beat_r = eval_indices_whisper(model, processor, bm, bp, val, proto.report_indices, device)
    me_r = mean_error(o_r["wer"], o_r["cer"])
    me_b = mean_error(b_r["wer"], b_r["cer"])
    log.info("%s REPORT_me=%.4f base=%.4f beats_report=%s", lang, me_r, me_b, beat_r)

    best = out_dir / "best"
    best.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best)
    processor.save_pretrained(best)
    meta = {
        "lang": lang,
        "init": init_id,
        "max_steps": max_steps,
        "lr": lr,
        "max_train": max_samples,
        "early_stop_slice": SELECT_SLICE_LABEL,
        "report_slice": REPORT_SLICE_LABEL,
        "fixed_final_only": True,
        "no_intermediate_step_pick": True,
        "pure_own_checkpoint": True,
        "no_baseline_blend": True,
        "select_mean_error": me_s,
        "select_beats": beat_s,
        "report_mean_error_own": me_r,
        "report_mean_error_base": me_b,
        "report_beats": beat_r,
        "own": o_r,
        "baseline": b_r,
        "baseline_model": mid,
        "own_model": str(best),
        "own_kind": "whisper_ft_protocol",
        "forbidden_train_splits": list(FORBIDDEN_TRAIN_SPLITS),
        "seed": SEED,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=["lin", "sna", "lug"])
    p.add_argument("--prefer-waxal-init", action="store_true", default=True)
    p.add_argument("--model-id", default="openai/whisper-small")
    p.add_argument("--max-samples", type=int, default=800)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--out-root", type=Path, default=Path("checkpoints/whisper-protocol-beat-mms"))
    args = p.parse_args()
    device = pick_device()
    results = []
    for lang in args.languages:
        init = WAXAL_WH.get(lang, args.model_id) if args.prefer_waxal_init else args.model_id
        try:
            WhisperProcessor.from_pretrained(init, local_files_only=True)
        except Exception:
            init = args.model_id
        r = train_lang(lang, init, args.out_root / lang, args.max_samples, args.max_steps, args.lr, device)
        results.append(r)
    all_beat = all(r["report_beats"] for r in results)
    report = {
        "seed": SEED,
        "protocol": "Whisper FT train-only; fixed-final; select val[50:90] logged; report val[0:50] once",
        "baseline_definition": "waxal-benchmarking/mms-300m-waxal-{lang}",
        "own_definition": "Whisper FT (waxal-whisper-small init preferred)",
        "per_language": results,
        "all_languages_beat_baseline": all_beat,
        "forbidden_train_splits": list(FORBIDDEN_TRAIN_SPLITS),
    }
    out = Path("outputs/multi_agent_push/t1_whisper_val_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s all_beat=%s", out, all_beat)
    # Also print comparison
    for r in results:
        print(f"WHISPER {r['lang']} report_me={r['report_mean_error_own']:.4f} base={r['report_mean_error_base']:.4f} beats={r['report_beats']}")
    print("WHISPER_ALL_BEAT", all_beat)
    return 0 if all_beat else 2

if __name__ == "__main__":
    raise SystemExit(main())
