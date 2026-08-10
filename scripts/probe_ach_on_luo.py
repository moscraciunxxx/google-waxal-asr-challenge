#!/usr/bin/env python3
"""Decode the true-Dholuo probe wavs with the INCUMBENT ach decoder.

Needed to finish the loss matrix for the luo-swap decision. We already know the
damage L of swapping a non-Dholuo row to the luo decoder (measured on the ach probe).
The mirror quantity G -- the gain when the row really is Dholuo -- requires the ach
decoder's output on true-Dholuo audio, which was never dumped.

Incumbent recipe, identical to production: waxal-300m-waxal-ach + ach_2gram KenLM
beam alpha 0.2 beta 0.5 width 100, length guard 0.5 <= bw/gw <= 2.0.
"""

from __future__ import annotations

import json
import logging
import os
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
sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from src.metrics import score_pairs
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ach_on_luo")

ACH = "waxal-benchmarking/mms-300m-waxal-ach"
PROBE = ROOT / "outputs" / "next_iter" / "probe" / "luo"


def main() -> None:
    device = torch.device("cpu")  # 40 short clips; keep MPS free for other jobs
    refs = pd.read_csv(PROBE / "refs.csv")
    proc = AutoProcessor.from_pretrained(ACH)
    model = Wav2Vec2ForCTC.from_pretrained(ACH).to(device).eval()
    vocab = proc.tokenizer.get_vocab()
    labels = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    uni = ROOT / "data" / "lms" / "ach_unigrams.txt"
    unigrams = [w.strip() for w in uni.read_text().splitlines() if w.strip()] if uni.exists() else None
    dec = build_ctcdecoder(labels, kenlm_model_path=str(ROOT / "data" / "lms" / "ach_2gram.arpa"),
                           unigrams=unigrams, alpha=0.2, beta=0.5)

    hyps = []
    with torch.inference_mode():
        for i in range(len(refs)):
            wav = PROBE / f"{i:03d}.wav"
            arr, sr = sf.read(wav, dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(-1)
            assert sr == TARGET_SR, f"{wav} sr={sr}"
            arr = arr / float(np.max(np.abs(arr)) + 1e-9)
            lg = model(proc(arr, sampling_rate=TARGET_SR, return_tensors="pt",
                            padding=True).input_values.to(device)).logits[0].float().cpu().numpy()
            g = normalize_text(proc.decode(torch.tensor(lg.argmax(-1)))) or "."
            b = normalize_text(dec.decode(lg, beam_width=100).replace("|", " ")) or "."
            gw, bw = max(1, len(g.split())), max(1, len(b.split()))
            hyps.append(b if (0.5 <= bw / gw <= 2.0 and b.strip()) else g)
            if (i + 1) % 10 == 0:
                logger.info("%d/%d", i + 1, len(refs))

    refs["ach_hyp"] = hyps
    refs.to_csv(PROBE / "ach_incumbent.csv", index=False)

    gold = refs.ref.astype(str).tolist()
    s_ach = score_pairs(gold, hyps)
    s_luo = score_pairs(gold, refs.baseline_hyp.astype(str).tolist())
    G = (1 - s_luo["score"]) - (1 - s_ach["score"])
    out = {
        "n": len(gold),
        "ach_incumbent_on_true_dholuo": 1 - s_ach["score"],
        "mms1b_luo_on_true_dholuo": 1 - s_luo["score"],
        "G_gain_per_correct_swap": G,
    }
    print(json.dumps(out, indent=2))
    (ROOT / "outputs" / "next_iter" / "luo_swap_gain.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
