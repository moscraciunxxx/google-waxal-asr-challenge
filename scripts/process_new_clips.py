#!/usr/bin/env python3
"""Process the NEW Phase-2 clips (2026-08 test-set extension) end-to-end.

Replicates the publicly-validated pipeline per clip:
  1. MMS-LID-126 -> lang1/p1
  2. openset multi-hyp routing (exact run_phase2_openset.py logic) -> decode_lang
  3. route upgrades: lug -> mms-lug-ft-v3 greedy · ach -> waxal-ach KenLM beam
     (a0.2 b0.5 w100 + guard) · nyn -> mms-nyn-ft-v1 + KenLM beam (a0.3, guard)
  4. Luo dual island: lid=luo & p1>=0.99 -> mms1b.luo ∩ CLEAR CER<=0.15 -> mms1b text

Output: outputs/next_iter/new_clips_table.csv (one row per clip, all fields).
PAZA decodes run separately via paza_decode.py --phase2 --audio-dir ... --lid-csv ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import jiwer
import numpy as np
import pandas as pd
import torch
from pyctcdecode import build_ctcdecoder
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    Wav2Vec2BertForCTC,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, OUTPUT_DIR, PROJECT_ROOT, TARGET_SR
from src.mms_infer import load_mms, pick_device, set_lang, transcribe_waveform
from src.text_norm import normalize_text
from scripts.run_phase2_openset import FALLBACK, WAXAL300, ModelCache, load_wav, resolve_model_lang

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("new_clips")

LID_ID = "facebook/mms-lid-126"
CLEAR_ID = "CLEAR-Global/w2v-bert-2.0-luo_19_77h"


def mps_free():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def stage_lid(wavs: list[Path], out_csv: Path, device) -> pd.DataFrame:
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        expected_ids = {p.stem for p in wavs}
        cached_ids = set(df.get("ID", pd.Series(dtype=str)).astype(str))
        if len(df) == len(wavs) and cached_ids == expected_ids:
            logger.info("LID cached (%d)", len(df))
            return df
    fe = AutoFeatureExtractor.from_pretrained(LID_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(LID_ID).to(device).eval()
    rows = []
    t0 = time.time()
    with torch.inference_mode():
        for k, p in enumerate(wavs):
            arr, sr = load_wav(p)
            if sr != TARGET_SR:
                import librosa

                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
            inputs = fe(arr, sampling_rate=TARGET_SR, return_tensors="pt")
            logits = model(inputs.input_values.to(device)).logits[0]
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, 3)
            langs = [model.config.id2label[int(i)] for i in top.indices]
            rows.append(
                {
                    "ID": p.stem,
                    "lang1": langs[0],
                    "p1": float(top.values[0]),
                    "lang2": langs[1],
                    "p2": float(top.values[1]),
                    "lang3": langs[2],
                    "p3": float(top.values[2]),
                }
            )
            if (k + 1) % 100 == 0:
                logger.info("lid %d/%d %.1fs", k + 1, len(wavs), time.time() - t0)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    del model
    mps_free()
    return df


def stage_openset(lid: pd.DataFrame, audio_dir: Path, out_csv: Path, device) -> pd.DataFrame:
    done = {}
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = {str(r.ID): r.to_dict() for _, r in prev.iterrows()}
        logger.info("openset resume %d", len(done))
    cache = ModelCache(device)
    rows = list(done.values())
    t0 = time.time()
    todo = lid[~lid.ID.astype(str).isin(done)]
    for n, (_, r) in enumerate(todo.iterrows(), start=1):
        uid = str(r.ID)
        lid_lang = str(r.lang1)
        arr, sr = load_wav(audio_dir / f"{uid}.wav")
        if lid_lang == "luo":
            candidates = ["ach", "lug", "sog"]
        elif lid_lang == "lug":
            candidates = ["lug", "nyn", "sog"]
        elif lid_lang in WAXAL300:
            candidates = [lid_lang]
        else:
            candidates = [resolve_model_lang(lid_lang), "lug", "ach"]
        best_hyp, best_lang, best_conf = ".", candidates[0], -1e9
        for use_lang in candidates:
            src, model, processor = cache.get(use_lang)
            if src.startswith("mms1b") and use_lang in ("lin", "sna", "lug"):
                set_lang(model, processor, use_lang)
            hyp, conf = transcribe_waveform(model, processor, arr, sr, device=device, return_confidence=True)
            if conf > best_conf:
                best_hyp, best_lang, best_conf = hyp, use_lang, conf
        rows.append(
            {"ID": uid, "lid_lang": lid_lang, "p1": float(r.p1), "decode_lang": best_lang,
             "openset_text": normalize_text(best_hyp) or ".", "confidence": best_conf}
        )
        if n % 25 == 0 or n == len(todo):
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            logger.info("openset %d/%d %.1fs", n, len(todo), time.time() - t0)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    cache.cache.clear()
    mps_free()
    return df


@torch.inference_mode()
def ctc_greedy(model, proc, arr, device) -> str:
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return normalize_text(proc.decode(ids)) or "."


@torch.inference_mode()
def ctc_logits(model, proc, arr, device) -> np.ndarray:
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    return model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()


def make_decoder(proc, lang: str, alpha: float):
    vocab = proc.tokenizer.get_vocab()
    labels = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    uni = ROOT / "data" / "lms" / f"{lang}_unigrams.txt"
    unigrams = [w.strip() for w in uni.read_text().splitlines() if w.strip()] if uni.exists() else None
    return build_ctcdecoder(
        labels, kenlm_model_path=str(ROOT / "data" / "lms" / f"{lang}_2gram.arpa"),
        unigrams=unigrams, alpha=alpha, beta=0.5,
    )


def beam_with_guard(model, proc, decoder, arr, device) -> str:
    lg = ctc_logits(model, proc, arr, device)
    g = normalize_text(proc.decode(torch.tensor(lg.argmax(-1)))) or "."
    b = normalize_text(decoder.decode(lg, beam_width=100).replace("|", " ")) or "."
    gw, bw = max(1, len(g.split())), max(1, len(b.split()))
    return b if (0.5 <= bw / gw <= 2.0 and b.strip()) else g


def stage_routes(df: pd.DataFrame, audio_dir: Path, out_csv: Path, device) -> pd.DataFrame:
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        if len(prev) == len(df) and set(prev.ID.astype(str)) == set(df.ID.astype(str)):
            logger.info("routes cached")
            return prev
    df = df.copy()
    df["route_text"] = df["openset_text"]
    # lug route
    ids = df[df.decode_lang == "lug"].ID.tolist()
    if ids:
        proc = AutoProcessor.from_pretrained(str(CHECKPOINT_DIR / "mms-lug-ft-v3"), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(CHECKPOINT_DIR / "mms-lug-ft-v3"), local_files_only=True).to(device).eval()
        t0 = time.time()
        for k, uid in enumerate(ids):
            arr, _ = load_wav(audio_dir / f"{uid}.wav")
            df.loc[df.ID == uid, "route_text"] = ctc_greedy(model, proc, arr, device)
            if (k + 1) % 50 == 0:
                logger.info("lug %d/%d %.1fs", k + 1, len(ids), time.time() - t0)
        del model
        mps_free()
    # ach route (waxal-ach beam a0.2)
    ids = df[df.decode_lang == "ach"].ID.tolist()
    if ids:
        proc = AutoProcessor.from_pretrained(WAXAL300["ach"])
        model = Wav2Vec2ForCTC.from_pretrained(WAXAL300["ach"]).to(device).eval()
        dec = make_decoder(proc, "ach", 0.2)
        t0 = time.time()
        for k, uid in enumerate(ids):
            arr, _ = load_wav(audio_dir / f"{uid}.wav")
            df.loc[df.ID == uid, "route_text"] = beam_with_guard(model, proc, dec, arr, device)
            if (k + 1) % 50 == 0:
                logger.info("ach %d/%d %.1fs", k + 1, len(ids), time.time() - t0)
        del model
        mps_free()
    # nyn route (publicly-validated mms-nyn-ft-v1 + beam a0.3)
    ids = df[df.decode_lang == "nyn"].ID.tolist()
    if ids:
        proc = AutoProcessor.from_pretrained(str(CHECKPOINT_DIR / "mms-nyn-ft-v1"), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(CHECKPOINT_DIR / "mms-nyn-ft-v1"), local_files_only=True).to(device).eval()
        dec = make_decoder(proc, "nyn", 0.3)
        t0 = time.time()
        for k, uid in enumerate(ids):
            arr, _ = load_wav(audio_dir / f"{uid}.wav")
            df.loc[df.ID == uid, "route_text"] = beam_with_guard(model, proc, dec, arr, device)
            if (k + 1) % 50 == 0:
                logger.info("nyn %d/%d %.1fs", k + 1, len(ids), time.time() - t0)
        del model
        mps_free()
    df.to_csv(out_csv, index=False)
    return df


def stage_dual(df: pd.DataFrame, audio_dir: Path, out_csv: Path, device) -> pd.DataFrame:
    df = df.copy()
    for col, default in (("mms1b_luo", ""), ("clear_luo", ""), ("cer_mc", np.nan), ("dual_accept", 0)):
        if col not in df.columns:
            df[col] = default
    pool = df[(df.lid_lang == "luo") & (df.p1 >= 0.99)].ID.tolist()
    logger.info("dual pool (lid=luo p1>=0.99): %d", len(pool))
    if pool:
        mproc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
        mmodel = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
        mproc.tokenizer.set_target_lang("luo")
        mmodel.load_adapter("luo")
        mmodel.to(device).eval()
        hyps_m = {}
        for k, uid in enumerate(pool):
            arr, _ = load_wav(audio_dir / f"{uid}.wav")
            hyps_m[uid] = ctc_greedy(mmodel, mproc, arr, device)
            if (k + 1) % 50 == 0:
                logger.info("mms1b-luo %d/%d", k + 1, len(pool))
        del mmodel
        mps_free()
        cproc = AutoProcessor.from_pretrained(CLEAR_ID)
        cmodel = Wav2Vec2BertForCTC.from_pretrained(CLEAR_ID).to(device).eval()
        for k, uid in enumerate(pool):
            arr, _ = load_wav(audio_dir / f"{uid}.wav")
            inputs = cproc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            inputs = {kk: v.to(device) for kk, v in inputs.items() if kk in ("input_features", "attention_mask")}
            logits = cmodel(**inputs).logits
            ids = torch.argmax(logits, dim=-1)[0]
            ctext = normalize_text(cproc.batch_decode(ids.unsqueeze(0))[0].replace("|", " ")) or "."
            m = hyps_m[uid]
            cer_mc = jiwer.cer(m or ".", ctext or ".")
            accept = int(cer_mc <= 0.15)
            df.loc[df.ID == uid, ["mms1b_luo", "clear_luo", "cer_mc", "dual_accept"]] = [m, ctext, cer_mc, accept]
            if (k + 1) % 50 == 0:
                logger.info("clear %d/%d", k + 1, len(pool))
        del cmodel
        mps_free()
    df["final_base"] = np.where(df.dual_accept == 1, df.mms1b_luo, df.route_text)
    df["final_base"] = df["final_base"].fillna(df["route_text"])
    df.to_csv(out_csv, index=False)
    logger.info("dual accepts: %d", int(df.dual_accept.sum()))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, default=PROJECT_ROOT / "newaudios")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device) if args.device else pick_device()

    out = OUTPUT_DIR / "next_iter"
    out.mkdir(parents=True, exist_ok=True)
    wavs = sorted(args.audio_dir.glob("*.wav"))
    logger.info("new clips: %d device=%s", len(wavs), device)

    lid = stage_lid(wavs, out / "new_lid.csv", device)
    logger.info("LID dist: %s", lid.lang1.value_counts().head(8).to_dict())
    oset = stage_openset(lid, args.audio_dir, out / "new_openset.csv", device)
    logger.info("decode_lang dist: %s", oset.decode_lang.value_counts().to_dict())
    routed = stage_routes(oset, args.audio_dir, out / "new_routes.csv", device)
    final = stage_dual(routed, args.audio_dir, out / "new_clips_table.csv", device)
    logger.info("DONE: %s", out / "new_clips_table.csv")


if __name__ == "__main__":
    main()
