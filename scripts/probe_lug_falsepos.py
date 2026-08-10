#!/usr/bin/env python3
"""Measure the luo-swap gate's false-positive rate on true LUGANDA audio.

Gap this closes: the PAZA-inter-MMS agreement gate was calibrated as TPR on FLEURS
Dholuo vs FPR on WAXAL Acholi. But of the 785 lid=luo phase-2 rows the gate is applied
to, 308 (39%) are routed to the LUGANDA decoder, and the gate's behaviour on true
Luganda was never measured. Every EV estimate for the hybrid swap silently assumes the
Acholi FPR of 0.025 also holds for those 308 rows.

Outputs, on WAXAL lug validation (n=40, seed 42):
  - FPR at each candidate threshold (rows a Luganda clip would be wrongly swapped at)
  - L_lug: zindi damage of swapping a true-Luganda row from its production decoder
    (mms-lug-ft-v3 + lug KenLM beam) to mms-1b adapter.luo

Together with the Acholi numbers this gives a composition-weighted loss matrix over
what the 785 rows actually are, instead of an Acholi-only one.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from jiwer import cer as jiwer_cer
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.text_norm import normalize_text
from scripts.paza_decode import load_paza, decode_one, clean_hyp
from scripts.mms_adapter_ft import pick_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lug_fp")

OUT = ROOT / "outputs" / "next_iter" / "lug_falsepos_probe.csv"
OUTJ = ROOT / "outputs" / "next_iter" / "lug_falsepos_probe.json"
THRS = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5]


def main() -> None:
    device = pick_device(None)
    val = load_hf_asr_split("lug", "validation")
    idx = list(range(len(val)))
    random.Random(42).shuffle(idx)
    idx = idx[:40]

    refs, auds = [], []
    for i in idx:
        ex = val[i]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"].get("sampling_rate") or TARGET_SR)
        if sr != TARGET_SR:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        auds.append(arr / float(np.max(np.abs(arr)) + 1e-9))
        refs.append(normalize_text(ex.get("transcription") or "") or ".")
    durs = [len(a) / TARGET_SR for a in auds]
    logger.info("lug val n=%d median dur %.1fs device=%s", len(refs), float(np.median(durs)), device)

    # --- 1. PAZA, Dholuo-forced (same call as the 785-row run) ---
    pmodel, pproc, ids3 = load_paza(device)
    paza_raw = []
    for k, a in enumerate(auds):
        paza_raw.append(decode_one(pmodel, pproc, ids3, a, device, wps_cap=0))
        if (k + 1) % 10 == 0:
            logger.info("paza %d/%d", k + 1, len(auds))
    del pmodel
    if device.type == "mps":
        torch.mps.empty_cache()
    paza_c16 = [normalize_text(clean_hyp(r, d, 1.6)) or "." for r, d in zip(paza_raw, durs)]

    # --- 2. mms-1b adapter.luo (the swap target) ---
    mproc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    mmodel = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
    mproc.tokenizer.set_target_lang("luo")
    mmodel.load_adapter("luo")
    mmodel = mmodel.to(device).eval()
    mms_luo = []
    with torch.inference_mode():
        for k, a in enumerate(auds):
            lg = mmodel(mproc(a, sampling_rate=TARGET_SR, return_tensors="pt",
                              padding=True).input_values.to(device)).logits
            mms_luo.append(normalize_text(mproc.decode(torch.argmax(lg, dim=-1)[0])) or ".")
            if (k + 1) % 10 == 0:
                logger.info("mms1b-luo %d/%d", k + 1, len(auds))
    del mmodel
    if device.type == "mps":
        torch.mps.empty_cache()

    # --- 3. incumbent Luganda decoder: mms-lug-ft-v3 + lug KenLM beam ---
    lug_ck = CHECKPOINT_DIR / "mms-lug-ft-v3"
    lproc = AutoProcessor.from_pretrained(str(lug_ck), local_files_only=True)
    lmodel = Wav2Vec2ForCTC.from_pretrained(str(lug_ck), local_files_only=True).to(device).eval()
    vocab = lproc.tokenizer.get_vocab()
    labels = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    uni = ROOT / "data" / "lms" / "lug_unigrams.txt"
    unigrams = [w.strip() for w in uni.read_text().splitlines() if w.strip()] if uni.exists() else None
    dec = build_ctcdecoder(labels, kenlm_model_path=str(ROOT / "data" / "lms" / "lug_2gram.arpa"),
                           unigrams=unigrams, alpha=0.2, beta=0.5)
    inc = []
    with torch.inference_mode():
        for k, a in enumerate(auds):
            lg = lmodel(lproc(a, sampling_rate=TARGET_SR, return_tensors="pt",
                              padding=True).input_values.to(device)).logits[0].float().cpu().numpy()
            g = normalize_text(lproc.decode(torch.tensor(lg.argmax(-1)))) or "."
            b = normalize_text(dec.decode(lg, beam_width=100).replace("|", " ")) or "."
            gw, bw = max(1, len(g.split())), max(1, len(b.split()))
            inc.append(b if (0.5 <= bw / gw <= 2.0 and b.strip()) else g)
            if (k + 1) % 10 == 0:
                logger.info("lug incumbent %d/%d", k + 1, len(auds))

    cer_pm = [jiwer_cer(c, m) if c.strip() else 999.0 for c, m in zip(paza_c16, mms_luo)]
    df = pd.DataFrame({"ref": refs, "dur": durs, "paza_raw": paza_raw, "paza_c16": paza_c16,
                       "mms_luo": mms_luo, "incumbent": inc, "cer_pm": cer_pm})
    df.to_csv(OUT, index=False)

    s_inc = score_pairs(refs, inc)
    s_luo = score_pairs(refs, mms_luo)
    res = {
        "n": len(refs),
        "median_dur": float(np.median(durs)),
        "incumbent_lug_zindi": 1 - s_inc["score"],
        "swapped_to_mms_luo_zindi": 1 - s_luo["score"],
        "L_lug_damage_per_swap": (1 - s_inc["score"]) - (1 - s_luo["score"]),
        "fpr": {str(t): float(np.mean(np.array(cer_pm) <= t)) for t in THRS},
        "median_cer_pm": float(np.median(cer_pm)),
        "paza_c16_words_q25_med_q75": [float(np.percentile([len(c.split()) for c in paza_c16], q)) for q in (25, 50, 75)],
    }
    print(json.dumps(res, indent=2))
    OUTJ.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
