#!/usr/bin/env python3
"""Clean PAZA (microsoft/paza-whisper-large-v3-turbo) Dholuo decoder.

Modes:
  --sanity N     : decode N FLEURS luo_ke val clips, score vs refs (gate)
  --phase2       : decode all lid=luo Phase-2 wavs -> outputs/next_iter/paza_luo_hyps.csv

Guards: greedy, temperature fallback off, repetition collapse detection
(long n-gram loops -> flagged), 30s chunks via processor default.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT, TARGET_SR
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("paza")

PAZA_ID = "microsoft/paza-whisper-large-v3-turbo"
AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
LID_CSV = OUTPUT_DIR / "phase2_lid126_full.csv"


def load_paza(device):
    proc = AutoProcessor.from_pretrained(PAZA_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(PAZA_ID, torch_dtype=torch.float32)
    model.to(device).eval()
    # <|luo|> = 51867, <|transcribe|> = 50360, <|notimestamps|> = 50364
    tok = proc.tokenizer
    luo_id = tok.convert_tokens_to_ids("<|luo|>")
    tr_id = tok.convert_tokens_to_ids("<|transcribe|>")
    nots_id = tok.convert_tokens_to_ids("<|notimestamps|>")
    logger.info("token ids: luo=%s transcribe=%s notimestamps=%s", luo_id, tr_id, nots_id)
    # Register luo in generation config so generate(language=...) also works.
    gc = model.generation_config
    if getattr(gc, "lang_to_id", None) is not None:
        gc.lang_to_id["<|luo|>"] = int(luo_id)
    gc.forced_decoder_ids = None
    return model, proc, (int(luo_id), int(tr_id), int(nots_id))


FOREIGN_CHARS = set("ĩũīūễафрикйця")


def clean_hyp(text: str, dur: float, wps_cap: float = 2.2) -> str:
    """Anti-hallucination cleanup for PAZA notimestamps output.

    1. Cut at first word containing non-Dholuo characters (Kikuyu diacritics etc.)
    2. Cut where a 1-3 gram starts repeating >=3 times consecutively
    3. Words-per-second cap as backstop
    """
    words = text.split()
    out = []
    for w in words:
        if any(c in FOREIGN_CHARS for c in w):
            break
        out.append(w)
        # detect trailing n-gram loop
        stop = False
        for n in (1, 2, 3):
            if len(out) >= 3 * n and out[-n:] == out[-2 * n : -n] == out[-3 * n : -2 * n]:
                out = out[: len(out) - 2 * n]  # keep a single copy
                stop = True
                break
        if stop:
            break
    if wps_cap:
        max_words = max(3, int(dur * wps_cap) + 2)
        out = out[:max_words]
    return " ".join(out) or "."


_VAD = {}


def get_vad():
    if "m" not in _VAD:
        from silero_vad import get_speech_timestamps, load_silero_vad

        _VAD["m"] = load_silero_vad()
        _VAD["fn"] = get_speech_timestamps
    return _VAD["m"], _VAD["fn"]


@torch.inference_mode()
def decode_vad(model, proc, ids3, arr: np.ndarray, device, max_seg_s: float = 28.0) -> str:
    """VAD-chunked PAZA decode: transcribe only detected speech spans.

    Removes the padded-silence hallucination mode entirely; each span is
    decoded independently (still with clean_hyp per span as belt+braces).
    """
    vad_model, get_ts = get_vad()
    arr = arr / (float(np.max(np.abs(arr))) + 1e-9)
    ts = get_ts(torch.from_numpy(arr), vad_model, sampling_rate=TARGET_SR,
                min_silence_duration_ms=500, speech_pad_ms=300)
    if not ts:
        return "."
    cov = sum(t["end"] - t["start"] for t in ts) / max(1, len(arr))
    logger.debug("vad coverage %.2f spans=%d", cov, len(ts))
    # Trim-only: keep continuous audio from first to last speech (with padding),
    # split at the largest internal silence gaps only when > max_seg_s.
    max_len = int(max_seg_s * TARGET_SR)
    pad = int(0.3 * TARGET_SR)
    s0 = max(0, ts[0]["start"] - pad)
    e0 = min(len(arr), ts[-1]["end"] + pad)
    windows: list[tuple[int, int]] = []
    if e0 - s0 <= max_len:
        windows = [(s0, e0)]
    else:
        # split at biggest gaps between consecutive speech spans
        cur_s = s0
        prev_end = ts[0]["end"]
        for t in ts[1:]:
            if t["end"] + pad - cur_s > max_len:
                windows.append((cur_s, min(len(arr), prev_end + pad)))
                cur_s = max(0, t["start"] - pad)
            prev_end = t["end"]
        windows.append((cur_s, e0))
    parts = []
    for s, e in windows:
        seg = arr[s:e]
        if len(seg) < TARGET_SR // 5:
            continue
        h = decode_one(model, proc, ids3, seg, device, wps_cap=0)
        h = clean_hyp(h, len(seg) / TARGET_SR, wps_cap=2.6)
        if h and h != ".":
            parts.append(h)
    return normalize_text(" ".join(parts)) or "."


def rep_collapse(text: str, n: int = 3, times: int = 4) -> bool:
    """True if text ends in an n-gram repeated >= times (whisper loop)."""
    w = text.split()
    if len(w) < n * times:
        return False
    tail = w[-n:]
    reps = 0
    i = len(w) - n
    while i >= 0 and w[i : i + n] == tail:
        reps += 1
        i -= n
    return reps >= times


@torch.inference_mode()
def decode_one(model, proc, ids3, arr: np.ndarray, device, wps_cap: float = 2.4) -> str:
    """PAZA decode with prompt-token strip + words-per-second truncation.

    PAZA (whisper-turbo FT) hallucinates past end-of-audio on padded silence;
    its authors recommend WPS truncation. FLEURS Dholuo runs ~1.6-2.0 words/s.
    """
    luo_id, tr_id, nots_id = ids3
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt")
    feats = inputs.input_features.to(device)
    prompt = torch.tensor(
        [[model.generation_config.decoder_start_token_id, luo_id, tr_id, nots_id]],
        device=device,
    )
    out = model.generate(
        feats,
        decoder_input_ids=prompt,
        max_new_tokens=220,
        num_beams=1,
        do_sample=False,
    )
    # Decode WITHOUT slicing (token-boundary slicing corrupted first words);
    # the added lang tokens (<|luo|> etc.) are not marked special, so strip the
    # leaked leading language word textually instead.
    txt = proc.batch_decode(out, skip_special_tokens=True)[0]
    txt = normalize_text(txt) or "."
    txt = __import__("re").sub(r"^(?:luo|kik|som|mas|kln)\s+", "", txt) or "."
    if wps_cap:
        dur = len(arr) / TARGET_SR
        max_words = max(3, int(dur * wps_cap) + 2)
        words = txt.split()
        if len(words) > max_words:
            txt = " ".join(words[:max_words])
    return txt or "."


def load_wav(path: Path) -> np.ndarray:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(np.asarray(arr, dtype=np.float32), orig_sr=sr, target_sr=TARGET_SR)
    return np.asarray(arr, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", type=int, default=0)
    ap.add_argument("--phase2", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, proc, ids3 = load_paza(device)

    if args.sanity:
        from datasets import Audio, load_dataset
        from src.metrics import score_pairs

        ds = load_dataset("google/fleurs", "luo_ke", split="validation")
        ds = ds.cast_column("audio", Audio(decode=False))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(ds), size=min(args.sanity, len(ds)), replace=False)
        refs, hyps_raw, durs = [], [], []
        t0 = time.time()
        for k, i in enumerate(idx):
            ex = ds[int(i)]
            aud = ex["audio"]
            src = io.BytesIO(aud["bytes"]) if isinstance(aud, dict) and aud.get("bytes") else str(aud.get("path"))
            arr, sr = sf.read(src, dtype="float32", always_2d=False)
            if sr != TARGET_SR:
                import librosa

                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
            h = decode_one(model, proc, ids3, arr, device, wps_cap=0)
            refs.append(normalize_text(ex.get("transcription") or "") or ".")
            hyps_raw.append(h)
            durs.append(len(arr) / TARGET_SR)
            logger.info("[%d/%d %.1fs] raw %d words", k + 1, len(idx), time.time() - t0, len(h.split()))
        pd.DataFrame({"ref": refs, "raw": hyps_raw, "dur": durs}).to_csv(
            ROOT / "outputs" / "next_iter" / "paza_sanity_raw.csv", index=False
        )
        res = {"n": len(refs)}
        for cap in (1.8, 2.0, 2.2, 2.4, 2.8, 0):
            hyps = [clean_hyp(h, d, wps_cap=cap) for h, d in zip(hyps_raw, durs)]
            sc = score_pairs(refs, hyps)
            res[f"clean_cap_{cap}"] = {**sc, "zindi": 1.0 - sc["score"]}
            logger.info("clean cap=%s -> zindi=%.4f wer=%.4f cer=%.4f", cap, 1 - sc["score"], sc["wer"], sc["cer"])
        (ROOT / "outputs" / "next_iter" / "paza_fleurs_sanity.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    if args.phase2:
        lid = pd.read_csv(LID_CSV)
        lid["ID"] = lid["ID"].astype(str)
        pool = lid[lid.lang1 == "luo"]["ID"].tolist()
        if args.limit:
            pool = pool[: args.limit]
        out_path = ROOT / "outputs" / "next_iter" / "paza_luo_hyps.csv"
        rows = []
        # resume support
        done = set()
        if out_path.exists():
            prev = pd.read_csv(out_path)
            rows = prev.to_dict("records")
            done = set(prev["ID"].astype(str))
            logger.info("resuming: %d already decoded", len(done))
        t0 = time.time()
        for k, uid in enumerate(pool):
            if uid in done:
                continue
            arr = load_wav(AUDIO / f"{uid}.wav")
            raw = decode_one(model, proc, ids3, arr, device, wps_cap=0)
            dur = len(arr) / TARGET_SR
            h = clean_hyp(raw, dur, wps_cap=1.6)
            rows.append({"ID": uid, "paza_luo": h, "paza_raw": raw, "dur": dur, "rep_flag": int(rep_collapse(raw))})
            if (len(rows) % 25) == 0:
                pd.DataFrame(rows).to_csv(out_path, index=False)
                el = time.time() - t0
                logger.info("%d/%d decoded (%.1fs elapsed, %.2fs/utt)", len(rows), len(pool), el, el / max(1, k + 1))
        pd.DataFrame(rows).to_csv(out_path, index=False)
        logger.info("wrote %s (%d rows)", out_path, len(rows))


if __name__ == "__main__":
    main()
