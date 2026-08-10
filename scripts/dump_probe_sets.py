#!/usr/bin/env python3
"""Stage 1 (main venv): dump probe eval sets as wav files + refs + baseline hyps.

Sets (all seed-42, identical sampling to prior gates):
  lug: WAXAL lug validation shuffle[:N]  — baseline = checkpoints/mms-lug-ft-v3 greedy
  nyn: WAXAL nyn validation shuffle[:N]  — baseline = waxal-nyn KenLM beam (floor recipe)
  ach: WAXAL ach validation (proxy ids)[:N] — baseline = waxal-ach KenLM beam a02 (floor recipe)
  luo: FLEURS luo_ke validation choice[:N]  — baseline = mms-1b-all adapter.luo greedy

Output: outputs/next_iter/probe/{lang}/NNN.wav + refs.csv (ID,ref,baseline_hyp)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import pick_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe_dump")

OUT = ROOT / "outputs" / "next_iter" / "probe"


def prep(arr, sr):
    arr = np.asarray(arr, dtype=np.float32)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak


@torch.inference_mode()
def greedy(model, proc, arr, device):
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits
    return normalize_text(proc.decode(torch.argmax(logits, dim=-1)[0])) or "."


@torch.inference_mode()
def beam(model, proc, decoder, arr, device, width=100):
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    lg = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
    g = normalize_text(proc.decode(torch.tensor(lg.argmax(-1)))) or "."
    b = normalize_text(decoder.decode(lg, beam_width=width).replace("|", " ")) or "."
    gw, bw = max(1, len(g.split())), max(1, len(b.split()))
    return b if (0.5 <= bw / gw <= 2.0 and b.strip()) else g


def make_decoder(proc, lang, alpha):
    vocab = proc.tokenizer.get_vocab()
    labels = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    uni = ROOT / "data" / "lms" / f"{lang}_unigrams.txt"
    unigrams = [w.strip() for w in uni.read_text().splitlines() if w.strip()] if uni.exists() else None
    return build_ctcdecoder(
        labels, kenlm_model_path=str(ROOT / "data" / "lms" / f"{lang}_2gram.arpa"),
        unigrams=unigrams, alpha=alpha, beta=0.5,
    )


def dump(lang, rows, base_fn):
    d = OUT / lang
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, (rid, ref, arr) in enumerate(rows):
        sf.write(str(d / f"{i:03d}.wav"), arr, TARGET_SR)
        recs.append({"idx": i, "ID": rid, "ref": ref, "baseline_hyp": base_fn(arr)})
        if (i + 1) % 10 == 0:
            logger.info("%s %d/%d", lang, i + 1, len(rows))
    pd.DataFrame(recs).to_csv(d / "refs.csv", index=False)
    logger.info("wrote %s (%d rows)", d, len(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--langs", nargs="+", default=["lug", "nyn", "ach", "luo"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    N = args.n

    if "lug" in args.langs:
        val = load_hf_asr_split("lug", "validation")
        idx = list(range(len(val)))
        random.Random(42).shuffle(idx)
        rows = []
        for i in idx[:N]:
            ex = val[i]
            rows.append((str(ex.get("id") or i), normalize_text(ex.get("transcription") or "") or ".",
                         prep(ex["audio"]["array"], int(ex["audio"].get("sampling_rate") or TARGET_SR))))
        proc = AutoProcessor.from_pretrained("checkpoints/mms-lug-ft-v3", local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained("checkpoints/mms-lug-ft-v3", local_files_only=True).to(device).eval()
        dump("lug", rows, lambda a: greedy(model, proc, a, device))
        del model
        torch.mps.empty_cache() if device.type == "mps" else None

    if "nyn" in args.langs:
        val = load_hf_asr_split("nyn", "validation")
        idx = list(range(len(val)))
        random.Random(42).shuffle(idx)
        rows = []
        for i in idx[:N]:
            ex = val[i]
            rows.append((str(ex.get("id") or i), normalize_text(ex.get("transcription") or "") or ".",
                         prep(ex["audio"]["array"], int(ex["audio"].get("sampling_rate") or TARGET_SR))))
        proc = AutoProcessor.from_pretrained("waxal-benchmarking/mms-300m-waxal-nyn")
        model = Wav2Vec2ForCTC.from_pretrained("waxal-benchmarking/mms-300m-waxal-nyn").to(device).eval()
        dec = make_decoder(proc, "nyn", 0.3)
        dump("nyn", rows, lambda a: beam(model, proc, dec, a, device))
        del model
        torch.mps.empty_cache() if device.type == "mps" else None

    if "ach" in args.langs:
        proxy = pd.read_csv(ROOT / "data" / "proxy_val_index.csv")
        ach_ids = set(proxy.loc[proxy.language == "ach", "id"].astype(str))
        ds = load_hf_asr_split("ach", "validation")
        rows = []
        for i in range(len(ds)):
            if len(rows) >= N:
                break
            ex = ds[i]
            eid = str(ex.get("id") or "")
            if ach_ids and eid not in ach_ids and len(ach_ids) > 5:
                continue
            rows.append((eid or str(i), normalize_text(ex.get("transcription") or "") or ".",
                         prep(ex["audio"]["array"], int(ex["audio"].get("sampling_rate") or TARGET_SR))))
        proc = AutoProcessor.from_pretrained("waxal-benchmarking/mms-300m-waxal-ach")
        model = Wav2Vec2ForCTC.from_pretrained("waxal-benchmarking/mms-300m-waxal-ach").to(device).eval()
        dec = make_decoder(proc, "ach", 0.2)
        dump("ach", rows, lambda a: beam(model, proc, dec, a, device))
        del model
        torch.mps.empty_cache() if device.type == "mps" else None

    if "luo" in args.langs:
        from datasets import Audio, load_dataset

        ds = load_dataset("google/fleurs", "luo_ke", split="validation")
        ds = ds.cast_column("audio", Audio(decode=False))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(ds), size=min(N, len(ds)), replace=False)
        rows = []
        for i in idx:
            ex = ds[int(i)]
            aud = ex["audio"]
            src = io.BytesIO(aud["bytes"]) if isinstance(aud, dict) and aud.get("bytes") else str(aud.get("path"))
            arr, sr = sf.read(src, dtype="float32", always_2d=False)
            rows.append((str(ex.get("id") or i), normalize_text(ex.get("transcription") or "") or ".",
                         prep(arr, sr)))
        proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
        proc.tokenizer.set_target_lang("luo")
        model.load_adapter("luo")
        model.to(device).eval()
        dump("luo", rows, lambda a: greedy(model, proc, a, device))
        del model

    print("DUMP_DONE", OUT)


if __name__ == "__main__":
    main()
