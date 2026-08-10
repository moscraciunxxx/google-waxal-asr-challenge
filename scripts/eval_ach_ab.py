#!/usr/bin/env python3
"""A/B ach decoders on WAXAL ach validation (n=120 seed 42), identical beam recipe.

Baseline: waxal-300m-waxal-ach + KenLM 2gram beam a0.2 b0.5 w100 + guard (floor recipe).
Candidates: --ckpts dirs evaluated with the SAME beam recipe (and greedy for reference).
Ship gate: candidate+beam >= baseline+beam + 0.02.
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

from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import pick_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ach_ab")

BASE = "waxal-benchmarking/mms-300m-waxal-ach"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "next_iter" / "ach_ab.json")
    args = ap.parse_args()
    device = pick_device(args.device)

    val = load_hf_asr_split("ach", "validation")
    idx = list(range(len(val)))
    random.Random(42).shuffle(idx)
    idx = idx[: args.n]
    refs, auds = [], []
    for i in idx:
        ex = val[i]
        a = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"].get("sampling_rate") or TARGET_SR)
        if sr != TARGET_SR:
            import librosa

            a = librosa.resample(a, orig_sr=sr, target_sr=TARGET_SR)
        auds.append(a / (float(np.max(np.abs(a)) + 1e-9)))
        refs.append(normalize_text(ex.get("transcription") or "") or ".")
    logger.info("ach val n=%d device=%s", len(refs), device)

    results = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    uni = ROOT / "data" / "lms" / "ach_unigrams.txt"
    unigrams = [w.strip() for w in uni.read_text().splitlines() if w.strip()] if uni.exists() else None

    def eval_model(name, path):
        proc = AutoProcessor.from_pretrained(path, local_files_only=os.path.isdir(str(path)))
        model = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=os.path.isdir(str(path))).to(device).eval()
        vocab = proc.tokenizer.get_vocab()
        labels = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        dec = build_ctcdecoder(labels, kenlm_model_path=str(ROOT / "data" / "lms" / "ach_2gram.arpa"),
                               unigrams=unigrams, alpha=0.2, beta=0.5)
        g_h, b_h = [], []
        t0 = time.time()
        with torch.inference_mode():
            for k, arr in enumerate(auds):
                lg = model(proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
                                ).input_values.to(device)).logits[0].float().cpu().numpy()
                g = normalize_text(proc.decode(torch.tensor(lg.argmax(-1)))) or "."
                b = normalize_text(dec.decode(lg, beam_width=100).replace("|", " ")) or "."
                gw, bw = max(1, len(g.split())), max(1, len(b.split()))
                g_h.append(g)
                b_h.append(b if (0.5 <= bw / gw <= 2.0 and b.strip()) else g)
                if (k + 1) % 40 == 0:
                    logger.info("%s %d/%d %.1fs", name, k + 1, len(auds), time.time() - t0)
        for tag, hyps in ((f"{name}_greedy", g_h), (f"{name}_beam", b_h)):
            sc = score_pairs(refs, hyps)
            results[tag] = {"n": len(refs), **sc, "zindi": 1.0 - sc["score"]}
            logger.info("%s -> zindi=%.4f wer=%.4f cer=%.4f", tag, 1 - sc["score"], sc["wer"], sc["cer"])
        args.out.write_text(json.dumps(results, indent=2))
        del model
        torch.mps.empty_cache() if device.type == "mps" else None

    eval_model("waxal_ach", BASE)
    for ck in args.ckpts:
        eval_model(Path(ck).name, ck)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
