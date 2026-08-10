#!/usr/bin/env python3
"""A/B: full-utterance CTC vs Silero-VAD chunked stitch on WAXAL val (same IDs).

Gate: zindi_est improvement ≥ +0.01 on n=80 seed 42 before phase-2 redecode.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mms_adapter_ft import fix_mms_tokenizer, pick_device
from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.mms_infer import set_lang, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vad_ab")


def get_vad():
    from silero_vad import get_speech_timestamps, load_silero_vad

    return load_silero_vad(), get_speech_timestamps


@torch.inference_mode()
def decode_full(model, proc, arr, sr, device) -> str:
    return normalize_text(transcribe_waveform(model, proc, arr, sr, device=device)) or "."


@torch.inference_mode()
def decode_vad(model, proc, arr, sr, device, max_seg_s: float = 12.0) -> str:
    arr = np.asarray(arr, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    peak = float(np.max(np.abs(arr)) + 1e-9)
    arr = arr / peak
    vad_model, get_ts = get_vad()
    wav = torch.from_numpy(arr)
    ts = get_ts(wav, vad_model, sampling_rate=TARGET_SR, threshold=0.4, min_speech_duration_ms=200)
    if not ts:
        return decode_full(model, proc, arr, sr, device)
    pieces = []
    max_n = int(max_seg_s * TARGET_SR)
    for seg in ts:
        s, e = int(seg["start"]), int(seg["end"])
        chunk = arr[s:e]
        # further split long speech
        for i in range(0, len(chunk), max_n):
            sub = chunk[i : i + max_n]
            if len(sub) < int(0.3 * TARGET_SR):
                continue
            pieces.append(decode_full(model, proc, sub, TARGET_SR, device))
    text = normalize_text(" ".join(pieces)) or "."
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="lug", choices=["lug", "nyn", "ach", "lin", "sna"])
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--ckpt",
        default=None,
        help="default: mms-{lang}-ft-v3/v1 or waxal for ach",
    )
    args = ap.parse_args()
    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)

    if args.ckpt:
        ckpt = args.ckpt
        is_ft = Path(ckpt).exists()
    elif args.lang == "lug":
        ckpt = str(ROOT / "checkpoints" / "mms-lug-ft-v3")
        is_ft = True
    elif args.lang == "nyn":
        ckpt = str(ROOT / "checkpoints" / "mms-nyn-ft-v1")
        is_ft = True
    else:
        ckpt = f"waxal-benchmarking/mms-300m-waxal-{args.lang}"
        is_ft = False

    logger.info("lang=%s ckpt=%s device=%s", args.lang, ckpt, device)
    try:
        proc = AutoProcessor.from_pretrained(ckpt, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(ckpt, local_files_only=True)
    except Exception:
        proc = AutoProcessor.from_pretrained(ckpt)
        model = Wav2Vec2ForCTC.from_pretrained(ckpt)
    if is_ft:
        fix_mms_tokenizer(proc, args.lang)
    model.to(device).eval()

    ds = load_hf_asr_split(args.lang, "validation", max_samples=None)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: args.n]

    refs, full_hyps, vad_hyps = [], [], []
    for k, i in enumerate(idxs):
        ex = ds[int(i)]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        ref = normalize_text(str(ex.get("transcription") or ex.get("text") or ""))
        refs.append(ref)
        full_hyps.append(decode_full(model, proc, arr, sr, device))
        vad_hyps.append(decode_vad(model, proc, arr, sr, device))
        if (k + 1) % 10 == 0:
            logger.info("done %d/%d", k + 1, len(idxs))

    def pack(hyps):
        s = score_pairs(refs, hyps)
        return {
            "n": int(s["n"]),
            "wer": float(s["wer"]),
            "cer": float(s["cer"]),
            "error": float(s["score"]),
            "zindi": float(1.0 - s["score"]),
        }

    full = pack(full_hyps)
    vad = pack(vad_hyps)
    delta = vad["zindi"] - full["zindi"]
    out = {
        "lang": args.lang,
        "ckpt": ckpt,
        "full": full,
        "vad_chunk": vad,
        "delta_zindi": delta,
        "gate_pass": delta >= 0.01,
    }
    out_path = ROOT / "outputs" / "beat075" / f"vad_ab_{args.lang}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
