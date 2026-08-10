#!/usr/bin/env python3
"""Phase-2: MMS-LID language ID → per-lang MMS FT/ZS decode.

Validated on HF val: facebook/mms-lid-126 restricted to {lin,sna,lug} → ~100% LID.
This replaces broken CTC-conf routing (portal WER 0.82).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# LID model may need first download; allow hub unless offline forced by caller
# ASR FT prefers offline after load

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2ForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, LANGUAGES, OUTPUT_DIR, PROJECT_ROOT, TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.mms_infer import load_mms, pick_device, set_lang, transcribe_waveform
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase2_lid")

LID_MODEL = "facebook/mms-lid-126"
PHASE2_AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
PHASE2_SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    if peak > 0:
        arr = arr / peak
    return arr, int(sr)


class MmsLid:
    def __init__(self, device: torch.device, model_id: str = LID_MODEL):
        self.device = device
        self.feat = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        self.model.to(device).eval()
        self.label2id = {
            v: int(k)
            for k, v in self.model.config.id2label.items()
            if v in LANGUAGES
        }
        missing = [l for l in LANGUAGES if l not in self.label2id]
        if missing:
            raise RuntimeError(f"LID model missing labels {missing}")
        self.langs = list(LANGUAGES)
        self.idxs = [self.label2id[l] for l in self.langs]
        logger.info("LID labels %s", self.label2id)

    @torch.inference_mode()
    def predict(self, array: np.ndarray, sr: int = TARGET_SR) -> tuple[str, dict[str, float]]:
        if sr != TARGET_SR:
            import librosa

            array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
        inputs = self.feat(
            array, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        )
        logits = self.model(inputs.input_values.to(self.device)).logits[0]
        sub = logits[self.idxs]
        probs = torch.softmax(sub, dim=-1)
        best = int(torch.argmax(probs).item())
        lang = self.langs[best]
        pmap = {l: float(probs[i]) for i, l in enumerate(self.langs)}
        return lang, pmap


def load_ft(lang: str, device: torch.device, suffix: str = "ft-v2"):
    for name in (f"mms-{lang}-{suffix}", f"mms-{lang}-ft-v2", f"mms-{lang}-ft"):
        ckpt = CHECKPOINT_DIR / name
        if (ckpt / "model.safetensors").exists() or (ckpt / "pytorch_model.bin").exists():
            logger.info("FT %s <- %s", lang, ckpt)
            processor = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
            model = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
            try:
                from scripts.mms_adapter_ft import fix_mms_tokenizer

                fix_mms_tokenizer(processor, lang)
            except Exception as e:
                logger.warning("fix_mms_tokenizer: %s", e)
            model.to(device).eval()
            return model, processor
    return None, None


def run_val(args) -> dict:
    device = torch.device(args.device) if args.device else pick_device()
    lid = MmsLid(device)
    # ASR: ZS with adapters + optional FT per lang
    zs_model, zs_proc, device = load_mms(device=device)
    for lang in LANGUAGES:
        set_lang(zs_model, zs_proc, lang)
    ft = {}
    if not args.no_ft:
        for lang in LANGUAGES:
            m, p = load_ft(lang, device, args.ckpt_suffix)
            if m is not None:
                ft[lang] = (m, p)

    rows = []
    for lang in LANGUAGES:
        n = args.max_per_lang
        ds = load_hf_asr_split(lang, "validation", max_samples=n)
        logger.info("val %s n=%d", lang, len(ds))
        for i in tqdm(range(len(ds)), desc=f"lid-val-{lang}"):
            ex = ds[i]
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"]["sampling_rate"])
            ref = normalize_text(ex["transcription"])
            pred_lang, probs = lid.predict(arr, sr)
            # decode with predicted lang
            if pred_lang in ft:
                hyp = transcribe_waveform(ft[pred_lang][0], ft[pred_lang][1], arr, sr, device)
                src = "ft"
            else:
                set_lang(zs_model, zs_proc, pred_lang)
                hyp = transcribe_waveform(zs_model, zs_proc, arr, sr, device)
                src = "zs"
            # oracle decode for ceiling
            if lang in ft:
                oracle = transcribe_waveform(ft[lang][0], ft[lang][1], arr, sr, device)
            else:
                set_lang(zs_model, zs_proc, lang)
                oracle = transcribe_waveform(zs_model, zs_proc, arr, sr, device)
            rows.append(
                {
                    "true": lang,
                    "pred_lang": pred_lang,
                    "ref": ref,
                    "hyp": hyp,
                    "oracle": oracle,
                    "src": src,
                    **{f"p_{k}": v for k, v in probs.items()},
                }
            )

    refs = [r["ref"] for r in rows]
    hyps = [r["hyp"] for r in rows]
    oracles = [r["oracle"] for r in rows]
    lid_acc = sum(r["true"] == r["pred_lang"] for r in rows) / max(len(rows), 1)
    m = score_pairs(refs, hyps)
    mo = score_pairs(refs, oracles)
    out = {
        "n": len(rows),
        "lid_acc": lid_acc,
        "lid_ft": {
            "wer": m["wer"],
            "cer": m["cer"],
            "error": m["score"],
            "zindi": 1.0 - m["score"],
        },
        "oracle_lang": {
            "wer": mo["wer"],
            "cer": mo["cer"],
            "error": mo["score"],
            "zindi": 1.0 - mo["score"],
        },
        "confusion": {},
    }
    from collections import Counter, defaultdict

    cm = defaultdict(Counter)
    for r in rows:
        cm[r["true"]][r["pred_lang"]] += 1
    out["confusion"] = {k: dict(v) for k, v in cm.items()}
    path = OUTPUT_DIR / "phase2_lid_val_metrics.json"
    path.write_text(json.dumps(out, indent=2))
    logger.info("VAL %s", json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def run_phase2(args) -> Path:
    device = torch.device(args.device) if args.device else pick_device()
    lid = MmsLid(device)
    zs_model, zs_proc, device = load_mms(device=device)
    for lang in LANGUAGES:
        set_lang(zs_model, zs_proc, lang)

    # Prefer FT for all three; lin ZS was better on Phase1 but FT ok if present
    ft = {}
    if not args.no_ft:
        for lang in LANGUAGES:
            m, p = load_ft(lang, device, args.ckpt_suffix)
            if m is not None:
                ft[lang] = (m, p)
    # lin: optionally force ZS (Phase1 found ZS > FT for lin)
    if args.lin_zs and "lin" in ft:
        logger.info("Using ZS for lin (not FT)")
        del ft["lin"]

    sample = pd.read_csv(PHASE2_SAMPLE)
    want = [str(x) for x in sample.ID.tolist()]
    wavs = {p.stem: p for p in PHASE2_AUDIO.glob("*.wav")}
    missing = [i for i in want if i not in wavs]
    if missing:
        raise SystemExit(f"missing {len(missing)} audio e.g. {missing[:3]}")

    detail_path = OUTPUT_DIR / "phase2_lid_detail.csv"
    done = {}
    if args.resume and detail_path.exists():
        prev = pd.read_csv(detail_path)
        for _, r in prev.iterrows():
            done[str(r.ID)] = r.to_dict()
        logger.info("resume %d", len(done))

    rows = list(done.values())
    todo = [i for i in want if i not in done]
    t0 = time.time()
    for n, uid in enumerate(tqdm(todo, desc="phase2-lid"), start=1):
        arr, sr = load_wav(wavs[uid])
        lang, probs = lid.predict(arr, sr)
        if lang in ft:
            hyp = transcribe_waveform(ft[lang][0], ft[lang][1], arr, sr, device)
            src = f"ft_{args.ckpt_suffix}"
        else:
            set_lang(zs_model, zs_proc, lang)
            hyp = transcribe_waveform(zs_model, zs_proc, arr, sr, device)
            src = "zs"
        rows.append(
            {
                "ID": uid,
                "prediction": hyp,
                "chosen_lang": lang,
                "p_lin": probs["lin"],
                "p_sna": probs["sna"],
                "p_lug": probs["lug"],
                "source": src,
            }
        )
        if n % 50 == 0 or n == len(todo):
            pd.DataFrame(rows).to_csv(detail_path, index=False)
            rate = n / max(time.time() - t0, 1e-6)
            logger.info(
                "n=%d rate=%.2f/s eta=%.0fs langs=%s",
                n,
                rate,
                (len(todo) - n) / max(rate, 1e-9),
                pd.DataFrame(rows)["chosen_lang"].value_counts().to_dict(),
            )

    detail = pd.DataFrame(rows)
    detail.to_csv(detail_path, index=False)
    out = Path(args.out or (PROJECT_ROOT / "submission_phase2_lid.csv"))
    build_submission(detail[["ID", "prediction"]], sample_path=PHASE2_SAMPLE, out_path=out)
    report = check_submission(out, PHASE2_SAMPLE)
    report["lang_counts"] = detail["chosen_lang"].value_counts().to_dict()
    report["source_counts"] = detail["source"].value_counts().to_dict()
    (OUTPUT_DIR / "phase2_lid_check.json").write_text(json.dumps(report, indent=2))
    logger.info("CHECK %s", report)
    if not report.get("ok"):
        raise SystemExit(report)
    print("UPLOAD", out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["val", "phase2", "both"], default="both")
    p.add_argument("--max-per-lang", type=int, default=40)
    p.add_argument("--device", default=None)
    p.add_argument("--no-ft", action="store_true")
    p.add_argument("--lin-zs", action="store_true", help="Use ZS for lin after LID")
    p.add_argument("--ckpt-suffix", default="ft-v2")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    val = None
    if args.mode in ("val", "both"):
        val = run_val(args)
        # Gate: require LID > 90% and better than public floor path
        if val["lid_acc"] < 0.9:
            raise SystemExit(f"LID accuracy too low: {val['lid_acc']}")
        if args.mode == "both" and val["lid_ft"]["zindi"] < 0.55:
            logger.warning(
                "Val zindi %.3f still low; proceeding to phase2 but expect weak public score",
                val["lid_ft"]["zindi"],
            )

    if args.mode in ("phase2", "both"):
        run_phase2(args)


if __name__ == "__main__":
    main()
