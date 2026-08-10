#!/usr/bin/env python3
"""Gate a Luo FT checkpoint on FLEURS luo_ke validation (n=80, seed 42).

Reproduces the historical zero-shot baselines (mms1b_luo 0.8551) on the same
sample and compares the FT checkpoint. Also measures the false-fire side:
decode WAXAL ach validation (n=40, proxy ids) with the FT model + CLEAR and
report dual-agreement CER stats so the dual gate FPR can be re-checked.
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
from datasets import Audio, load_dataset
from transformers import AutoProcessor, Wav2Vec2BertForCTC, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import pick_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("luo_ft_gate")

MMS_ID = "facebook/mms-1b-all"
CLEAR_ID = "CLEAR-Global/w2v-bert-2.0-luo_19_77h"


def decode_audio(aud) -> tuple[np.ndarray, int]:
    if isinstance(aud, dict) and aud.get("bytes") is not None:
        src = io.BytesIO(aud["bytes"])
    else:
        src = str(aud.get("path") if isinstance(aud, dict) else aud)
    arr, sr = sf.read(src, dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


def resample(arr, sr):
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    return arr


@torch.inference_mode()
def greedy(model, processor, arr, device) -> str:
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return normalize_text(processor.decode(ids)) or "."


@torch.inference_mode()
def greedy_clear(model, processor, arr, device) -> str:
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_features", "attention_mask")}
    logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)[0]
    txt = processor.batch_decode(ids.unsqueeze(0))[0]
    return normalize_text(txt.replace("|", " ")) or "."


def char_cer(a: str, b: str) -> float:
    import jiwer

    a = a or "."
    b = b or "."
    return jiwer.cer(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", type=Path, required=True)
    ap.add_argument("--n-luo", type=int, default=80)
    ap.add_argument("--n-ach", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "next_iter" / "luo_ft_gate.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {"seed": args.seed, "device": str(device)}

    # --- FLEURS luo val sample (same protocol as historical gate: seed 42, n=80)
    ds = load_dataset("google/fleurs", "luo_ke", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n_luo, len(ds)), replace=False)
    refs, wavs = [], []
    for i in idx:
        ex = ds[int(i)]
        arr, sr = decode_audio(ex["audio"])
        wavs.append(resample(arr, sr))
        refs.append(normalize_text(ex.get("transcription") or ex.get("raw_transcription") or "") or ".")
    logger.info("fleurs luo val n=%d", len(refs))

    def record(name, hyps, refs_):
        sc = score_pairs(refs_, hyps)
        results[name] = {"n": len(refs_), **sc, "zindi": 1.0 - sc["score"]}
        logger.info("%s -> %s", name, results[name])
        args.out.write_text(json.dumps(results, indent=2))

    ft_hyps_luo: list[str] = []

    # FT model
    fproc = AutoProcessor.from_pretrained(str(args.ft), local_files_only=True)
    fmodel = Wav2Vec2ForCTC.from_pretrained(str(args.ft), local_files_only=True).to(device).eval()
    t0 = time.time()
    for k, arr in enumerate(wavs):
        ft_hyps_luo.append(greedy(fmodel, fproc, arr, device))
        if (k + 1) % 20 == 0:
            logger.info("ft fleurs %d/%d %.1fs", k + 1, len(wavs), time.time() - t0)
    record("ft_luo_on_fleurs", ft_hyps_luo, refs)

    zs_hyps_luo: list[str] = []
    if not args.skip_baselines:
        mproc = AutoProcessor.from_pretrained(MMS_ID)
        mmodel = Wav2Vec2ForCTC.from_pretrained(MMS_ID)
        mproc.tokenizer.set_target_lang("luo")
        mmodel.load_adapter("luo")
        mmodel.to(device).eval()
        for k, arr in enumerate(wavs):
            zs_hyps_luo.append(greedy(mmodel, mproc, arr, device))
        record("zeroshot_mms1b_luo_on_fleurs", zs_hyps_luo, refs)
        del mmodel
        if device.type == "mps":
            torch.mps.empty_cache()

    # --- False-fire side: WAXAL ach validation (true Acholi audio)
    proxy = pd.read_csv(ROOT / "data" / "proxy_val_index.csv")
    ach_ids = set(proxy.loc[proxy.language == "ach", "id"].astype(str))
    ds_ach = load_hf_asr_split("ach", "validation")
    ach_refs, ach_wavs = [], []
    for i in range(len(ds_ach)):
        if len(ach_wavs) >= args.n_ach:
            break
        ex = ds_ach[i]
        eid = str(ex.get("id") or "")
        if ach_ids and eid not in ach_ids and len(ach_ids) > 5:
            continue
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"].get("sampling_rate") or TARGET_SR)
        peak = float(np.max(np.abs(arr)) + 1e-9)
        ach_wavs.append(resample(arr / peak, sr))
        ach_refs.append(normalize_text(ex.get("transcription") or "") or ".")
    logger.info("waxal ach val n=%d", len(ach_wavs))

    ft_hyps_ach = [greedy(fmodel, fproc, a, device) for a in ach_wavs]
    record("ft_luo_on_ach_audio", ft_hyps_ach, ach_refs)
    del fmodel
    if device.type == "mps":
        torch.mps.empty_cache()

    # --- CLEAR hyps for dual-agreement stats (TPR on fleurs, FPR on ach)
    cproc = AutoProcessor.from_pretrained(CLEAR_ID)
    cmodel = Wav2Vec2BertForCTC.from_pretrained(CLEAR_ID).to(device).eval()
    clear_luo = [greedy_clear(cmodel, cproc, a, device) for a in wavs]
    clear_ach = [greedy_clear(cmodel, cproc, a, device) for a in ach_wavs]
    del cmodel
    record("clear_on_fleurs", clear_luo, refs)

    cer_luo = [char_cer(m, c) for m, c in zip(ft_hyps_luo, clear_luo)]
    cer_ach = [char_cer(m, c) for m, c in zip(ft_hyps_ach, clear_ach)]
    thr = 0.15
    results["dual_gate_ft_clear"] = {
        "thr": thr,
        "tpr_fleurs": float(np.mean([c <= thr for c in cer_luo])),
        "fpr_ach": float(np.mean([c <= thr for c in cer_ach])),
        "cer_luo_p50": float(np.percentile(cer_luo, 50)),
        "cer_ach_p10": float(np.percentile(cer_ach, 10)),
        "cer_ach_min": float(np.min(cer_ach)) if cer_ach else None,
    }
    if zs_hyps_luo:
        cer_luo_zs = [char_cer(m, c) for m, c in zip(zs_hyps_luo, clear_luo)]
        results["dual_gate_zeroshot_clear"] = {
            "thr": thr,
            "tpr_fleurs": float(np.mean([c <= thr for c in cer_luo_zs])),
        }
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("dual gate stats: %s", results["dual_gate_ft_clear"])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
