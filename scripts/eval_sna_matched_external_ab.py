#!/usr/bin/env python3
"""Matched n=80 sna A/B: waxal floor vs gold-only FT vs badrex w2v-bert vs mubarak whisper.

Same-ID seed=42 validation set. Gate: zindi >= floor + 0.005 AND wer strictly better.
Uses Wav2Vec2BertForCTC for badrex (not Wav2Vec2ForCTC).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    Wav2Vec2BertForCTC,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sna_matched_ab")

WAXAL_SNA = "waxal-benchmarking/mms-300m-waxal-sna"
GOLDONLY = ROOT / "checkpoints" / "mms300-sna-goldonly-v1" / "sna" / "best"
BADREX = "badrex/w2v-bert-2.0-shona-asr"
MUBARAK = "Mubarak127/waxal-whisper-large-v3-sna_asr"
GATE_MARGIN = 0.005


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prep_audio(ex) -> np.ndarray:
    arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
    sr = int(ex["audio"].get("sampling_rate") or TARGET_SR)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak


@torch.inference_mode()
def greedy_mms(model, proc, arr, device) -> str:
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits
    return normalize_text(proc.decode(torch.argmax(logits, dim=-1)[0])) or "."


@torch.inference_mode()
def greedy_w2vbert(model, proc, arr, device) -> str:
    # Wav2Vec2-BERT / SeamlessM4T feature extractor → input_features
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    kwargs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    logits = model(**kwargs).logits
    pred = torch.argmax(logits, dim=-1)[0]
    return normalize_text(proc.decode(pred)) or "."


@torch.inference_mode()
def whisper_decode(model, proc, arr, device, language: str = "sn") -> str:
    # Whisper expects 16k float waveform via processor
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    # Prefer forced Shona when supported; fall back to free gen
    try:
        forced = proc.get_decoder_prompt_ids(language=language, task="transcribe")
        ids = model.generate(
            input_features,
            forced_decoder_ids=forced,
            max_new_tokens=128,
        )
    except Exception:
        ids = model.generate(input_features, max_new_tokens=128)
    text = proc.batch_decode(ids, skip_special_tokens=True)[0]
    return normalize_text(text) or "."


def free_model(*objs):
    for o in objs:
        del o
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "goal_2026_08_06" / "sna_matched_external_ab.json",
    )
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="System tags to skip (waxal_greedy goldonly_ft_v1 badrex_w2vbert mubarak_whisper)",
    )
    args = ap.parse_args()
    device = pick_device(args.device)
    logger.info("device=%s n=%d seed=%d", device, args.n, args.seed)

    val = load_hf_asr_split("sna", "validation")
    idx = list(range(len(val)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]
    sample_ids = []
    refs, auds = [], []
    for i in idx:
        ex = val[i]
        sample_ids.append(str(ex.get("id") or ex.get("ID") or i))
        refs.append(normalize_text(ex.get("transcription") or "") or ".")
        auds.append(prep_audio(ex))
    logger.info("matched sna val n=%d first_ids=%s", len(refs), sample_ids[:5])

    results: dict = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def record(name: str, hyps: list[str] | None = None, error: str | None = None):
        if error is not None:
            results[name] = {"error": error, "n": len(refs)}
            logger.error("%s ERROR %s", name, error)
        else:
            assert hyps is not None
            sc = score_pairs(refs, hyps)
            results[name] = {
                "n": len(refs),
                "wer": sc["wer"],
                "cer": sc["cer"],
                "zindi": 1.0 - sc["score"],
            }
            logger.info(
                "%s -> zindi=%.6f wer=%.6f cer=%.6f",
                name,
                results[name]["zindi"],
                results[name]["wer"],
                results[name]["cer"],
            )
        payload = build_payload(results, sample_ids, args)
        args.out.write_text(json.dumps(payload, indent=2))
        return payload

    # --- waxal floor ---
    if "waxal_greedy" not in args.skip:
        logger.info("=== waxal_greedy")
        t0 = time.time()
        proc = AutoProcessor.from_pretrained(WAXAL_SNA)
        model = Wav2Vec2ForCTC.from_pretrained(WAXAL_SNA).to(device).eval()
        hyps = [greedy_mms(model, proc, a, device) for a in auds]
        free_model(model, proc)
        record("waxal_greedy", hyps)
        logger.info("waxal done in %.1fs", time.time() - t0)

    # --- gold-only FT ---
    if "goldonly_ft_v1" not in args.skip:
        logger.info("=== goldonly_ft_v1 %s", GOLDONLY)
        t0 = time.time()
        if not GOLDONLY.exists():
            record("goldonly_ft_v1", error=f"missing ckpt {GOLDONLY}")
        else:
            # Prefer FT processor; fall back to waxal processor if incomplete
            try:
                proc = AutoProcessor.from_pretrained(str(GOLDONLY))
            except Exception:
                proc = AutoProcessor.from_pretrained(WAXAL_SNA)
            model = Wav2Vec2ForCTC.from_pretrained(str(GOLDONLY)).to(device).eval()
            hyps = [greedy_mms(model, proc, a, device) for a in auds]
            free_model(model, proc)
            record("goldonly_ft_v1", hyps)
        logger.info("goldonly done in %.1fs", time.time() - t0)

    # --- badrex Wav2Vec2-BERT (correct class) ---
    if "badrex_w2vbert" not in args.skip:
        logger.info("=== badrex_w2vbert (Wav2Vec2BertForCTC)")
        t0 = time.time()
        try:
            proc = AutoProcessor.from_pretrained(BADREX)
            model = Wav2Vec2BertForCTC.from_pretrained(BADREX).to(device).eval()
            hyps = []
            for k, a in enumerate(auds):
                hyps.append(greedy_w2vbert(model, proc, a, device))
                if (k + 1) % 20 == 0:
                    logger.info("badrex %d/%d %.1fs", k + 1, len(auds), time.time() - t0)
            free_model(model, proc)
            record("badrex_w2vbert", hyps)
        except Exception as e:
            record("badrex_w2vbert", error=f"{type(e).__name__}: {e}")
        logger.info("badrex done in %.1fs", time.time() - t0)

    # --- mubarak whisper large ---
    if "mubarak_whisper" not in args.skip:
        logger.info("=== mubarak_whisper")
        t0 = time.time()
        try:
            # Whisper large may OOM on MPS; fall back to CPU if needed
            wdev = device
            try:
                proc = WhisperProcessor.from_pretrained(MUBARAK)
                model = WhisperForConditionalGeneration.from_pretrained(MUBARAK).to(wdev).eval()
            except Exception as e:
                logger.warning("MPS load failed (%s); retry CPU", e)
                wdev = torch.device("cpu")
                proc = WhisperProcessor.from_pretrained(MUBARAK)
                model = WhisperForConditionalGeneration.from_pretrained(MUBARAK).to(wdev).eval()
            hyps = []
            for k, a in enumerate(auds):
                hyps.append(whisper_decode(model, proc, a, wdev))
                if (k + 1) % 10 == 0:
                    logger.info("mubarak %d/%d %.1fs", k + 1, len(auds), time.time() - t0)
            free_model(model, proc)
            record("mubarak_whisper", hyps)
        except Exception as e:
            record("mubarak_whisper", error=f"{type(e).__name__}: {e}")
        logger.info("mubarak done in %.1fs", time.time() - t0)

    payload = build_payload(results, sample_ids, args)
    args.out.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s gate_pass=%s best=%s", args.out, payload.get("gate_pass"), payload.get("best_tag"))
    print(json.dumps(payload, indent=2))


def build_payload(results: dict, sample_ids: list, args) -> dict:
    floor_tag = "waxal_greedy"
    floor = results.get(floor_tag) or {}
    floor_z = float(floor.get("zindi") or 0.0)
    floor_w = float(floor.get("wer") or 1.0)

    best_tag = None
    best_z = -1.0
    for tag, r in results.items():
        if "error" in r or "zindi" not in r:
            continue
        if r["zindi"] > best_z:
            best_z = r["zindi"]
            best_tag = tag

    candidates = {}
    gate_pass = False
    ship_tag = None
    for tag, r in results.items():
        if tag == floor_tag or "error" in r or "zindi" not in r:
            continue
        dz = float(r["zindi"]) - floor_z
        better_wer = float(r["wer"]) < floor_w
        pass_i = (dz >= GATE_MARGIN) and better_wer and floor.get("zindi") is not None
        candidates[tag] = {
            "delta_zindi_vs_floor": dz,
            "better_wer": better_wer,
            "gate_pass": pass_i,
            "margin_required": GATE_MARGIN,
        }
        if pass_i and (ship_tag is None or r["zindi"] > results[ship_tag]["zindi"]):
            gate_pass = True
            ship_tag = tag

    return {
        "seed": args.seed,
        "n": args.n,
        "floor": floor_tag,
        "sample_ids_head": sample_ids[:10],
        "results": results,
        "best_tag": best_tag,
        "best_zindi": best_z if best_tag else None,
        "delta_vs_floor": (best_z - floor_z) if best_tag and floor.get("zindi") is not None else None,
        "candidates": candidates,
        "gate_pass": gate_pass,
        "ship_tag": ship_tag,
        "gate_rule": f"zindi >= floor+{GATE_MARGIN} AND wer < floor_wer; same n={args.n} seed={args.seed}",
        "supervision": "external / gold-only FT; no Phase-2 self-pseudo primary supervision",
        "models": {
            "waxal_greedy": WAXAL_SNA,
            "goldonly_ft_v1": str(GOLDONLY),
            "badrex_w2vbert": BADREX,
            "mubarak_whisper": MUBARAK,
        },
    }


if __name__ == "__main__":
    main()
