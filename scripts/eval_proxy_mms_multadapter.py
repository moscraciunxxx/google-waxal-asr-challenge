#!/usr/bin/env python3
"""Offline gate: same-family MMS multi-adapter pick on labeled Luo + Acholi.

Measures zindi = 1-0.5*(WER+CER) for:
- force luo adapter
- force ach adapter
- multi-adapter max conf among {luo,ach,lug}
- oracle pick among adapters

Also builds a Phase-2-weighted composite projection when combined with
proxy non-luo spine zindi.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import io
from tqdm import tqdm
from datasets import Audio, load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_hf_asr_split
from src.metrics import compute_cer, compute_wer
from src.mms_infer import load_mms, set_lang, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_mms_multadapter")

ADAPTERS = ("luo", "ach", "lug")


def load_wav(path):
    if isinstance(path, dict):
        if path.get("bytes") is not None:
            src = io.BytesIO(path["bytes"])
        else:
            src = str(path.get("path") or path)
    else:
        src = str(path)
    arr, sr = sf.read(src, dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


def zindi(refs, hyps):
    w = float(compute_wer(refs, hyps))
    c = float(compute_cer(refs, hyps))
    return 1.0 - 0.5 * w - 0.5 * c, w, c


def score_systems(model, processor, device, items, max_n=80, seed=42):
    rng = np.random.default_rng(seed)
    if len(items) > max_n:
        idx = rng.choice(len(items), size=max_n, replace=False)
        items = [items[i] for i in idx]

    force = {L: [] for L in ADAPTERS}
    multi_hyps = []
    oracle_hyps = []
    refs = []
    pick_langs = []

    for arr, sr, ref in tqdm(items, desc="eval-items"):
        ref = normalize_text(ref)
        refs.append(ref)
        scores = {}
        for L in ADAPTERS:
            set_lang(model, processor, L)
            text, conf = transcribe_waveform(
                model, processor, arr, sr, device=device, return_confidence=True
            )
            text = normalize_text(text) or "."
            scores[L] = (text, float(conf))
            force[L].append(text)
        best_L = max(ADAPTERS, key=lambda L: scores[L][1])
        multi_hyps.append(scores[best_L][0])
        pick_langs.append(best_L)
        # oracle: min CER to ref
        best_o = min(ADAPTERS, key=lambda L: compute_cer([ref], [scores[L][0]]))
        oracle_hyps.append(scores[best_o][0])

    out = {}
    for L in ADAPTERS:
        z, w, c = zindi(refs, force[L])
        out[f"force_{L}"] = {"zindi": z, "wer": w, "cer": c, "n": len(refs)}
    z, w, c = zindi(refs, multi_hyps)
    out["multi_maxconf"] = {
        "zindi": z, "wer": w, "cer": c, "n": len(refs),
        "pick_mass": {L: pick_langs.count(L) for L in ADAPTERS},
    }
    z, w, c = zindi(refs, oracle_hyps)
    out["oracle_adapter"] = {"zindi": z, "wer": w, "cer": c, "n": len(refs)}
    return out


def main():
    model, processor, device = load_mms()
    for L in ADAPTERS:
        set_lang(model, processor, L)

    # FLEURS true Luo
    ds = load_dataset("google/fleurs", "luo_ke", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    luo_items = []
    for i in range(len(ds)):
        ex = ds[i]
        arr, sr = load_wav(ex["audio"])
        ref = ex.get("transcription") or ex.get("raw_transcription") or ""
        luo_items.append((arr, sr, ref))

    # WAXAL ach validation (false Luo)
    ach_ds = load_hf_asr_split("ach", "validation", max_samples=80)
    ach_items = []
    for i in range(len(ach_ds)):
        ex = ach_ds[i]
        aud = ex["audio"]
        if isinstance(aud, dict) and "array" in aud:
            arr = np.asarray(aud["array"], dtype=np.float32)
            sr = int(aud.get("sampling_rate", 16000))
            peak = float(np.max(np.abs(arr)) + 1e-9)
            arr = arr / peak
        else:
            arr, sr = load_wav(aud)
        ref = ex.get("transcription") or ""
        ach_items.append((arr, sr, ref))

    t0 = time.time()
    res_luo = score_systems(model, processor, device, luo_items, max_n=80)
    res_ach = score_systems(model, processor, device, ach_items, max_n=80)
    wall = time.time() - t0

    # Mixed set 50/50 as stressed LID=luo mass
    mixed_items = luo_items[:40] + ach_items[:40]
    res_mix = score_systems(model, processor, device, mixed_items, max_n=80, seed=0)

    # Phase-2 weighted projection
    # mass_luo=0.523; assume among lid=luo, p_true from calib ~0.5 mid
    # multi-adapter z on true luo and false ach:
    z_true = res_luo["multi_maxconf"]["zindi"]
    z_false = res_ach["multi_maxconf"]["zindi"]
    projections = {}
    for p_true in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for z_non in [0.60, 0.65, 0.68, 0.70, 0.74]:
            z_luo = p_true * z_true + (1 - p_true) * z_false
            z = 0.523333 * z_luo + (1 - 0.523333) * z_non
            projections[f"p{p_true}_non{z_non}"] = {
                "z_luo": z_luo, "z_non": z_non, "projected_public": z,
                "beats_k63": z > 0.720212909,
            }

    out = {
        "adapters": list(ADAPTERS),
        "fleurs_luo": res_luo,
        "waxal_ach": res_ach,
        "mixed_50_50": res_mix,
        "projections": projections,
        "wall_s": wall,
        "method": "same-family MMS multi-adapter max CTC conf",
    }
    path = ROOT / "outputs" / "beat_k63" / "mms_multadapter_offline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    best = max(projections.items(), key=lambda kv: kv[1]["projected_public"])
    print("BEST PROJECTION", best)


if __name__ == "__main__":
    main()
