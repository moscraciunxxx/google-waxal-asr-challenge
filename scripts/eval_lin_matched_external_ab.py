#!/usr/bin/env python3
"""Matched n=80 lin A/B: mms1b zs floor vs open external lin specialists.

Gate: zindi >= floor+0.005 AND wer strictly better. Same seed=42 IDs.
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
    AutoConfig,
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
from scripts.mms_adapter_ft import pick_device, fix_mms_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lin_matched_ab")

GATE_MARGIN = 0.005
MMS1B = "facebook/mms-1b-all"

# Open / attempt candidates (gated may 403 — recorded as error)
CANDIDATES = [
    ("noirlab_whisper_large", "noirlab/whisper-large-v3-lingala-asr", "whisper"),
    ("drewmens_mms_waxal_lin", "DrewMens/mms-waxal-lingala", "wav2vec2"),
    ("keystats_xlsr_waxal_lin", "keystats/lingala-xlsr-waxal-finetuned", "wav2vec2"),
    ("sulaimank_w2vbert", "sulaimank/w2vbert-lingala-waxal", "w2vbert"),
    ("sunbird_whisper51", "Sunbird/asr-whisper-51-african-languages", "whisper"),
    ("douyeszn_w2vbert", "douyeszn/w2vbert-lin-waxal-aug-ft", "w2vbert"),
]


def prep_audio(ex) -> np.ndarray:
    arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
    sr = int(ex["audio"].get("sampling_rate") or TARGET_SR)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak


@torch.inference_mode()
def greedy_w2v(model, proc, arr, device) -> str:
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    if hasattr(inputs, "input_values") and inputs.get("input_values") is not None:
        logits = model(inputs.input_values.to(device)).logits
    else:
        kwargs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
        logits = model(**kwargs).logits
    return normalize_text(proc.decode(torch.argmax(logits, dim=-1)[0])) or "."


@torch.inference_mode()
def whisper_decode(model, proc, arr, device, language: str = "ln") -> str:
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt")
    feats = inputs.input_features.to(device)
    try:
        forced = proc.get_decoder_prompt_ids(language=language, task="transcribe")
        ids = model.generate(feats, forced_decoder_ids=forced, max_new_tokens=128)
    except Exception:
        try:
            # Sunbird may use custom language token names
            forced = proc.get_decoder_prompt_ids(language="lin", task="transcribe")
            ids = model.generate(feats, forced_decoder_ids=forced, max_new_tokens=128)
        except Exception:
            ids = model.generate(feats, max_new_tokens=128)
    return normalize_text(proc.batch_decode(ids, skip_special_tokens=True)[0]) or "."


def free(*objs, device=None):
    for o in objs:
        del o
    if device is not None and getattr(device, "type", None) == "mps":
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
        default=ROOT / "outputs" / "goal_2026_08_06" / "lin_matched_external_ab.json",
    )
    args = ap.parse_args()
    device = pick_device(args.device)
    logger.info("device=%s n=%d seed=%d", device, args.n, args.seed)

    val = load_hf_asr_split("lin", "validation")
    idx = list(range(len(val)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]
    sample_ids, refs, auds = [], [], []
    for i in idx:
        ex = val[i]
        sample_ids.append(str(ex.get("id") or ex.get("ID") or i))
        refs.append(normalize_text(ex.get("transcription") or "") or ".")
        auds.append(prep_audio(ex))
    logger.info("matched lin val n=%d head=%s", len(refs), sample_ids[:5])

    results: dict = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def record(name, hyps=None, error=None, model_id=None):
        if error is not None:
            results[name] = {"error": error, "n": len(refs), "model_id": model_id}
            logger.error("%s ERROR %s", name, error[:200])
        else:
            sc = score_pairs(refs, hyps)
            results[name] = {
                "n": len(refs),
                "wer": sc["wer"],
                "cer": sc["cer"],
                "zindi": 1.0 - sc["score"],
                "model_id": model_id,
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

    # --- floor: mms1b lin zs ---
    logger.info("=== mms1b_lin_zeroshot")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MMS1B)
    model = Wav2Vec2ForCTC.from_pretrained(MMS1B)
    fix_mms_tokenizer(proc, "lin")
    try:
        model.load_adapter("lin", local_files_only=True)
    except Exception:
        model.load_adapter("lin")
    model.to(device).eval()
    hyps = [greedy_w2v(model, proc, a, device) for a in auds]
    free(model, proc, device=device)
    record("mms1b_lin_zeroshot", hyps, model_id=MMS1B)
    logger.info("floor done %.1fs", time.time() - t0)

    for tag, mid, kind in CANDIDATES:
        logger.info("=== %s (%s)", tag, mid)
        t0 = time.time()
        try:
            cfg = AutoConfig.from_pretrained(mid)
            mtype = getattr(cfg, "model_type", "") or ""
            if kind == "whisper" or mtype == "whisper":
                wdev = device
                try:
                    wproc = WhisperProcessor.from_pretrained(mid)
                    wmodel = WhisperForConditionalGeneration.from_pretrained(mid).to(wdev).eval()
                except Exception as e:
                    logger.warning("load on %s failed (%s); CPU", device, e)
                    wdev = torch.device("cpu")
                    wproc = WhisperProcessor.from_pretrained(mid)
                    wmodel = WhisperForConditionalGeneration.from_pretrained(mid).to(wdev).eval()
                hyps = []
                for k, a in enumerate(auds):
                    hyps.append(whisper_decode(wmodel, wproc, a, wdev))
                    if (k + 1) % 20 == 0:
                        logger.info("%s %d/%d %.1fs", tag, k + 1, len(auds), time.time() - t0)
                free(wmodel, wproc, device=wdev)
                record(tag, hyps, model_id=mid)
            elif kind == "w2vbert" or mtype in ("wav2vec2-bert", "seamless_m4t"):
                bproc = AutoProcessor.from_pretrained(mid)
                bmodel = Wav2Vec2BertForCTC.from_pretrained(mid).to(device).eval()
                hyps = [greedy_w2v(bmodel, bproc, a, device) for a in auds]
                free(bmodel, bproc, device=device)
                record(tag, hyps, model_id=mid)
            else:
                vproc = AutoProcessor.from_pretrained(mid)
                vmodel = Wav2Vec2ForCTC.from_pretrained(mid).to(device).eval()
                hyps = [greedy_w2v(vmodel, vproc, a, device) for a in auds]
                free(vmodel, vproc, device=device)
                record(tag, hyps, model_id=mid)
        except Exception as e:
            record(tag, error=f"{type(e).__name__}: {e}", model_id=mid)
        logger.info("%s wall %.1fs", tag, time.time() - t0)

    payload = build_payload(results, sample_ids, args)
    args.out.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s gate_pass=%s ship=%s", args.out, payload["gate_pass"], payload.get("ship_tag"))
    print(json.dumps(payload, indent=2))


def build_payload(results, sample_ids, args) -> dict:
    floor_tag = "mms1b_lin_zeroshot"
    floor = results.get(floor_tag) or {}
    floor_z = float(floor.get("zindi") or 0.0)
    floor_w = float(floor.get("wer") or 1.0)
    best_tag, best_z = None, -1.0
    candidates = {}
    gate_pass, ship_tag = False, None
    for tag, r in results.items():
        if "error" in r or "zindi" not in r:
            continue
        if r["zindi"] > best_z:
            best_z, best_tag = r["zindi"], tag
        if tag == floor_tag:
            continue
        dz = float(r["zindi"]) - floor_z
        better_wer = float(r["wer"]) < floor_w
        ok = (dz >= GATE_MARGIN) and better_wer and floor.get("zindi") is not None
        candidates[tag] = {
            "delta_zindi_vs_floor": dz,
            "better_wer": better_wer,
            "gate_pass": ok,
            "margin_required": GATE_MARGIN,
        }
        if ok and (ship_tag is None or r["zindi"] > results[ship_tag]["zindi"]):
            gate_pass, ship_tag = True, tag
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
        "gate_rule": f"zindi >= floor+{GATE_MARGIN} AND wer < floor_wer; n={args.n} seed={args.seed}",
        "hf_access_note": (
            "Sunbird/salt dataset accepted but has NO lingala. "
            "sulaimank / Sunbird-asr-51 / douyeszn still 403 until model gates accepted."
        ),
        "supervision": "external eval only; no Phase-2 self-pseudo",
    }


if __name__ == "__main__":
    main()
