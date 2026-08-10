#!/usr/bin/env python3
"""A/B lin decoders on WAXAL lin validation (n=120 seed 42).

Systems: waxal-300m-lin greedy · waxal-lin + KenLM beam (a0.2/a0.3) ·
mms-1b-all lin adapter zero-shot · checkpoints/mms-lin-ft-v2 greedy.
Winner (>= +0.01 over waxal greedy) decodes the 444 new lin clips.
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
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import pick_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lin_ab")

WAXAL_LIN = "waxal-benchmarking/mms-300m-waxal-lin"


def prep_audio(ex) -> np.ndarray:
    arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
    sr = int(ex["audio"].get("sampling_rate") or TARGET_SR)
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
def logits_np(model, proc, arr, device):
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    return model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "next_iter" / "lin_ab.json")
    args = ap.parse_args()
    device = pick_device(args.device)

    val = load_hf_asr_split("lin", "validation")
    idx = list(range(len(val)))
    random.Random(42).shuffle(idx)
    idx = idx[: args.n]
    refs, auds = [], []
    for i in idx:
        ex = val[i]
        refs.append(normalize_text(ex.get("transcription") or "") or ".")
        auds.append(prep_audio(ex))
    logger.info("lin val n=%d device=%s", len(refs), device)

    results = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def record(name, hyps):
        sc = score_pairs(refs, hyps)
        results[name] = {"n": len(refs), **sc, "zindi": 1.0 - sc["score"]}
        logger.info("%s -> zindi=%.4f wer=%.4f cer=%.4f", name, 1 - sc["score"], sc["wer"], sc["cer"])
        args.out.write_text(json.dumps(results, indent=2))

    # waxal greedy + beams
    proc = AutoProcessor.from_pretrained(WAXAL_LIN)
    model = Wav2Vec2ForCTC.from_pretrained(WAXAL_LIN).to(device).eval()
    vocab = proc.tokenizer.get_vocab()
    labels = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    uni = ROOT / "data" / "lms" / "lin_unigrams.txt"
    unigrams = [w.strip() for w in uni.read_text().splitlines() if w.strip()] if uni.exists() else None
    decs = {a: build_ctcdecoder(labels, kenlm_model_path=str(ROOT / "data" / "lms" / "lin_2gram.arpa"),
                                unigrams=unigrams, alpha=a, beta=0.5) for a in (0.2, 0.3)}
    g_h, b_h = [], {a: [] for a in decs}
    t0 = time.time()
    for k, arr in enumerate(auds):
        lg = logits_np(model, proc, arr, device)
        g = normalize_text(proc.decode(torch.tensor(lg.argmax(-1)))) or "."
        g_h.append(g)
        for a, d in decs.items():
            b = normalize_text(d.decode(lg, beam_width=100).replace("|", " ")) or "."
            gw, bw = max(1, len(g.split())), max(1, len(b.split()))
            b_h[a].append(b if (0.5 <= bw / gw <= 2.0 and b.strip()) else g)
        if (k + 1) % 40 == 0:
            logger.info("waxal %d/%d %.1fs", k + 1, len(auds), time.time() - t0)
    record("waxal_lin_greedy", g_h)
    for a in decs:
        record(f"waxal_lin_beam_a{a}", b_h[a])
    del model
    torch.mps.empty_cache() if device.type == "mps" else None

    # mms1b zero-shot
    mproc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    mmodel = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
    mproc.tokenizer.set_target_lang("lin")
    mmodel.load_adapter("lin")
    mmodel.to(device).eval()
    record("mms1b_lin_zeroshot", [greedy(mmodel, mproc, a, device) for a in auds])
    del mmodel
    torch.mps.empty_cache() if device.type == "mps" else None

    # local FTs
    for suffix in ("ft-v2", "ft-v3"):
        ft = CHECKPOINT_DIR / f"mms-lin-{suffix}"
        if not (ft / "model.safetensors").exists():
            continue
        fproc = AutoProcessor.from_pretrained(str(ft), local_files_only=True)
        fmodel = Wav2Vec2ForCTC.from_pretrained(str(ft), local_files_only=True).to(device).eval()
        record(f"mms_lin_{suffix}", [greedy(fmodel, fproc, a, device) for a in auds])
        del fmodel
        torch.mps.empty_cache() if device.type == "mps" else None

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
