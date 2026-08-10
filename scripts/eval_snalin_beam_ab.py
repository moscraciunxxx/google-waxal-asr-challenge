#!/usr/bin/env python3
"""Beam A/B for the two routes that were never LM-decoded (889 of 2392 test rows).

sna: production decoder is waxal-300m-sna GREEDY (0.8476). sna_2gram.arpa exists
     but no beam was ever run on this route.
lin: production decoder is mms-1b adapter.lin ZERO-SHOT greedy (0.7700). The lin
     beam numbers on record (0.7213/0.7155) were measured on waxal-300m-lin, the
     model that LOST the A/B — mms-1b + lin LM is untested.

Same recipe as every shipped route: build_ctcdecoder(beta=0.5), beam_width=100,
length guard 0.5 <= bw/gw <= 2.0 (fall back to greedy outside it). n=120 seed 42.
Ship gate: >= +0.01 zindi on the route.
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
logger = logging.getLogger("snalin_beam")

ALPHAS = (0.1, 0.2, 0.3, 0.4)


def prep_audio(ex) -> np.ndarray:
    arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
    sr = int(ex["audio"].get("sampling_rate") or TARGET_SR)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    return arr / float(np.max(np.abs(arr)) + 1e-9)


def load_val(lang: str, n: int):
    val = load_hf_asr_split(lang, "validation")
    idx = list(range(len(val)))
    random.Random(42).shuffle(idx)
    refs, auds = [], []
    for i in idx[:n]:
        ex = val[i]
        refs.append(normalize_text(ex.get("transcription") or "") or ".")
        auds.append(prep_audio(ex))
    return refs, auds


def labels_for(proc) -> list[str]:
    """Sorted label list for pyctcdecode; handles MMS per-language nested vocab."""
    vocab = proc.tokenizer.get_vocab()
    if vocab and isinstance(next(iter(vocab.values())), dict):
        # MMS tokenizer keeps {lang: {token: id}}; target lang was already set
        vocab = vocab[proc.tokenizer.target_lang]
    return [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]


def run_route(lang: str, model, proc, refs, auds, device, results, out_path):
    labels = labels_for(proc)
    uni = ROOT / "data" / "lms" / f"{lang}_unigrams.txt"
    unigrams = [w.strip() for w in uni.read_text().splitlines() if w.strip()] if uni.exists() else None
    arpa = str(ROOT / "data" / "lms" / f"{lang}_2gram.arpa")
    decs = {a: build_ctcdecoder(labels, kenlm_model_path=arpa, unigrams=unigrams, alpha=a, beta=0.5)
            for a in ALPHAS}

    g_h = []
    b_h = {a: [] for a in ALPHAS}
    b_raw = {a: [] for a in ALPHAS}  # no length guard, to see if the guard is helping or hurting
    t0 = time.time()
    with torch.inference_mode():
        for k, arr in enumerate(auds):
            inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            lg = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
            g = normalize_text(proc.decode(torch.tensor(lg.argmax(-1)))) or "."
            g_h.append(g)
            gw = max(1, len(g.split()))
            for a, d in decs.items():
                b = normalize_text(d.decode(lg, beam_width=100).replace("|", " ")) or "."
                bw = max(1, len(b.split()))
                b_raw[a].append(b)
                b_h[a].append(b if (0.5 <= bw / gw <= 2.0 and b.strip()) else g)
            if (k + 1) % 30 == 0:
                logger.info("%s %d/%d %.1fs", lang, k + 1, len(auds), time.time() - t0)

    def record(name, hyps):
        sc = score_pairs(refs, hyps)
        results[name] = {"n": len(refs), **sc, "zindi": 1.0 - sc["score"]}
        logger.info("%s -> zindi=%.4f wer=%.4f cer=%.4f", name, 1 - sc["score"], sc["wer"], sc["cer"])
        out_path.write_text(json.dumps(results, indent=2))

    record(f"{lang}_greedy", g_h)
    for a in ALPHAS:
        record(f"{lang}_beam_a{a}", b_h[a])
        record(f"{lang}_beam_a{a}_noguard", b_raw[a])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=["sna", "lin"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "next_iter" / "snalin_beam_ab.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    for lang in args.langs:
        refs, auds = load_val(lang, args.n)
        logger.info("%s val n=%d device=%s", lang, len(refs), device)
        if lang == "sna":
            # production sna decoder
            path = "waxal-benchmarking/mms-300m-waxal-sna"
            proc = AutoProcessor.from_pretrained(path)
            model = Wav2Vec2ForCTC.from_pretrained(path).to(device).eval()
        else:
            # production lin decoder: mms-1b-all with the lin adapter
            proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
            model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
            proc.tokenizer.set_target_lang(lang)
            model.load_adapter(lang)
            model = model.to(device).eval()
        run_route(lang, model, proc, refs, auds, device, results, args.out)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
