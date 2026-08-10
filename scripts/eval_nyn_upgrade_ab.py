#!/usr/bin/env python3
"""A/B eval on WAXAL nyn validation: floor nyn route vs MMS-1B nyn adapter.

Systems:
  waxal_nyn_greedy  — waxal-benchmarking/mms-300m-waxal-nyn, greedy CTC
  waxal_nyn_beam    — same + KenLM 2gram beam (alpha 0.3, beta 0.5, beam 100)
                      + length-guard vs greedy  (== floor nyn_beam recipe)
  mms1b_nyn         — facebook/mms-1b-all + adapter nyn, greedy (zero-shot)
  ft:<path>         — optional fine-tuned MMS-1B nyn checkpoint(s) via --ft

Data: google/WaxalNLP nyn_asr validation, seed-42 sample (never test gold).
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("nyn_ab")

WAXAL_NYN = "waxal-benchmarking/mms-300m-waxal-nyn"
MMS_ID = "facebook/mms-1b-all"


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prep_audio(ex) -> tuple[np.ndarray, int]:
    arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
    sr = int(ex["audio"].get("sampling_rate") or TARGET_SR)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, TARGET_SR


@torch.inference_mode()
def greedy_decode(model, processor, arr, device) -> str:
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return normalize_text(processor.decode(ids)) or "."


@torch.inference_mode()
def ctc_logits(model, processor, arr, device) -> np.ndarray:
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits
    return logits[0].float().cpu().numpy()


def build_nyn_decoder(processor, alpha: float = 0.3, beta: float = 0.5):
    vocab = processor.tokenizer.get_vocab()
    labels = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    lm = ROOT / "data" / "lms" / "nyn_2gram.arpa"
    uni_path = ROOT / "data" / "lms" / "nyn_unigrams.txt"
    unigrams = None
    if uni_path.exists():
        unigrams = [w.strip() for w in uni_path.read_text().splitlines() if w.strip()]
    return build_ctcdecoder(labels, kenlm_model_path=str(lm), unigrams=unigrams, alpha=alpha, beta=beta)


def length_guard(greedy: str, beamed: str) -> str:
    gw = max(1, len(greedy.split()))
    bw = max(1, len(beamed.split()))
    if 0.5 <= bw / gw <= 2.0 and beamed.strip():
        return beamed
    return greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ft", nargs="*", default=[], help="FT checkpoint dirs to also eval")
    ap.add_argument("--ft-beam", action="store_true", help="also KenLM-beam-decode FT systems")
    ap.add_argument("--skip-base", action="store_true", help="only eval --ft systems")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "next_iter" / "nyn_ab.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    logger.info("device=%s n=%d", device, args.n)

    val = load_hf_asr_split("nyn", "validation")
    idx = list(range(len(val)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]
    logger.info("nyn validation total=%d using n=%d", len(val), len(idx))

    refs, audios = [], []
    for i in idx:
        ex = val[i]
        refs.append(normalize_text(ex.get("transcription") or "") or ".")
        audios.append(prep_audio(ex)[0])

    results: dict[str, dict] = {}
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    def record(name: str, hyps: list[str]):
        sc = score_pairs(refs, hyps)
        results[name] = {"n": len(refs), **sc, "zindi": 1.0 - sc["score"]}
        logger.info("%s -> %s", name, results[name])
        out.write_text(
            json.dumps(
                {"seed": args.seed, "n": len(refs), "device": str(device), "results": results},
                indent=2,
            )
        )

    if not args.skip_base:
        logger.info("== waxal-nyn greedy + beam ==")
        proc = AutoProcessor.from_pretrained(WAXAL_NYN)
        model = Wav2Vec2ForCTC.from_pretrained(WAXAL_NYN).to(device).eval()
        decoder = build_nyn_decoder(proc)
        g_hyps, b_hyps = [], []
        t0 = time.time()
        for k, arr in enumerate(audios):
            lg = ctc_logits(model, proc, arr, device)
            ids = lg.argmax(-1)
            greedy = normalize_text(proc.decode(torch.tensor(ids))) or "."
            beamed = normalize_text(decoder.decode(lg, beam_width=100).replace("|", " ")) or "."
            g_hyps.append(greedy)
            b_hyps.append(length_guard(greedy, beamed))
            if (k + 1) % 20 == 0:
                logger.info("waxal %d/%d %.1fs", k + 1, len(audios), time.time() - t0)
        record("waxal_nyn_greedy", g_hyps)
        record("waxal_nyn_beam_a03b05w100", b_hyps)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

        logger.info("== mms-1b nyn zero-shot ==")
        mproc = AutoProcessor.from_pretrained(MMS_ID)
        mmodel = Wav2Vec2ForCTC.from_pretrained(MMS_ID)
        mproc.tokenizer.set_target_lang("nyn")
        mmodel.load_adapter("nyn")
        mmodel.to(device).eval()
        hyps = []
        t0 = time.time()
        for k, arr in enumerate(audios):
            hyps.append(greedy_decode(mmodel, mproc, arr, device))
            if (k + 1) % 20 == 0:
                logger.info("mms1b %d/%d %.1fs", k + 1, len(audios), time.time() - t0)
        record("mms1b_nyn_zeroshot", hyps)
        del mmodel
        if device.type == "mps":
            torch.mps.empty_cache()

    for ft in args.ft:
        ft_path = Path(ft)
        name = f"ft:{ft_path.name}"
        logger.info("== %s ==", name)
        fproc = AutoProcessor.from_pretrained(str(ft_path), local_files_only=True)
        fmodel = Wav2Vec2ForCTC.from_pretrained(str(ft_path), local_files_only=True).to(device).eval()
        g_hyps, b_hyps = [], []
        fdecoder = build_nyn_decoder(fproc) if args.ft_beam else None
        t0 = time.time()
        for k, arr in enumerate(audios):
            lg = ctc_logits(fmodel, fproc, arr, device)
            ids = lg.argmax(-1)
            g = normalize_text(fproc.decode(torch.tensor(ids))) or "."
            g_hyps.append(g)
            if fdecoder is not None:
                b = normalize_text(fdecoder.decode(lg, beam_width=100).replace("|", " ")) or "."
                b_hyps.append(length_guard(g, b))
            if (k + 1) % 20 == 0:
                logger.info("%s %d/%d %.1fs", name, k + 1, len(audios), time.time() - t0)
        record(name, g_hyps)
        if fdecoder is not None:
            record(f"{name}+beam", b_hyps)
        del fmodel
        if device.type == "mps":
            torch.mps.empty_cache()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
