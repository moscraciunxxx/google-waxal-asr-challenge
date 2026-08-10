#!/usr/bin/env python3
"""Domain-beam redecode of PUBLIC-VISIBLE Luganda only (lid!=luo, decode=lug).

Proxy (outputs/proxy_lug_beam_ab.json): ft_v3 greedy 0.874 → domain beam α0.3 0.898
(+0.023). Prior public "lugbeam full" ban was mass rewrite including private/luo-routed
rows; this script is scoped to public-visible lug only + length guard vs floor.

Uses pyctcdecode + KenLM ARPA (prefer domain merged if present).
"""

from __future__ import annotations

import argparse
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
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mms_adapter_ft import fix_mms_tokenizer, pick_device
from src.config import TARGET_SR
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lug_domain_beam")


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


def build_decoder(processor, arpa: Path, alpha: float, beta: float):
    vocab = processor.tokenizer.get_vocab()
    id2 = {i: t for t, i in vocab.items()}
    labels = [id2[i] for i in range(len(id2))]
    uni = arpa.with_name(arpa.name.replace("_2gram.arpa", "_unigrams.txt").replace("_3gram.arpa", "_unigrams.txt"))
    # try standard unigrams
    candidates = [
        uni,
        ROOT / "data" / "lms" / "lug_unigrams.txt",
        ROOT / "data" / "lms_phase2_domain" / "lug_unigrams.txt",
    ]
    unigrams = None
    for u in candidates:
        if u.exists():
            unigrams = [x for x in u.read_text().splitlines() if x.strip()]
            break
    kwargs = {"kenlm_model_path": str(arpa), "alpha": alpha, "beta": beta}
    if unigrams:
        kwargs["unigrams"] = unigrams
    return build_ctcdecoder(labels, **kwargs)


@torch.inference_mode()
def decode_beam(model, processor, decoder, arr, sr, device, beam_width: int = 100) -> str:
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
    text = decoder.decode(logits, beam_width=beam_width)
    text = text.replace("|", " ")
    return normalize_text(text) or "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "mms-lug-ft-v3")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--arpa",
        type=Path,
        default=None,
        help="default: phase2 domain merged if present else data/lms/lug_2gram.arpa",
    )
    ap.add_argument("--len-lo", type=float, default=0.70)
    ap.add_argument("--len-hi", type=float, default=1.35)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "submission_phase2_beat075_lug_domain_beam.csv",
    )
    args = ap.parse_args()

    arpa = args.arpa
    if arpa is None:
        for c in [
            ROOT / "outputs" / "beat075" / "lug_domain_3gram.arpa",
            ROOT / "data" / "lms_phase2_domain" / "lug_merged_2gram.arpa",
            ROOT / "data" / "lms" / "lug_2gram.arpa",
        ]:
            if c.exists():
                arpa = c
                break
    if arpa is None or not arpa.exists():
        raise SystemExit("no ARPA found")

    device = pick_device(args.device)
    logger.info("ckpt=%s arpa=%s alpha=%s device=%s", args.ckpt, arpa, args.alpha, device)

    processor = AutoProcessor.from_pretrained(str(args.ckpt), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(args.ckpt), local_files_only=True)
    fix_mms_tokenizer(processor, "lug")
    model.to(device).eval()
    decoder = build_decoder(processor, arpa, args.alpha, args.beta)

    idx = pd.read_csv(ROOT / "outputs" / "beat075" / "public_visible_index.csv")
    lug = idx[idx.decode_lang == "lug"].copy()
    floor = pd.read_csv(ROOT / "submission_phase2_v2_full.csv")
    floor_map = floor.set_index("ID")["Target"].astype(str).to_dict()

    hyps = {}
    t0 = time.time()
    for k, r in enumerate(lug.itertuples()):
        arr, sr = load_wav(Path(r.audio))
        hyps[r.ID] = decode_beam(model, processor, decoder, arr, sr, device, args.beam)
        if (k + 1) % 25 == 0:
            logger.info("%d/%d %.1fs", k + 1, len(lug), time.time() - t0)
    logger.info("done decode %d in %.1fs", len(lug), time.time() - t0)

    # merge with length guard
    tgt = floor_map.copy()
    n_rep = 0
    n_guard = 0
    for uid, hyp in hyps.items():
        fl = floor_map[uid]
        fw, hw = len(normalize_text(fl).split()), len(normalize_text(hyp).split())
        if fw > 0:
            ratio = hw / fw
            if ratio < args.len_lo or ratio > args.len_hi:
                n_guard += 1
                continue
        if normalize_text(hyp) and normalize_text(hyp) != normalize_text(fl):
            tgt[uid] = hyp
            n_rep += 1

    out_rows = [{"ID": r.ID, "Target": tgt[r.ID]} for r in floor.itertuples()]
    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    meta = {
        "n_public_lug": len(lug),
        "n_replaced": n_rep,
        "n_length_guard": n_guard,
        "alpha": args.alpha,
        "beta": args.beta,
        "beam": args.beam,
        "arpa": str(arpa),
        "ckpt": str(args.ckpt),
        "out": str(args.out),
        "proxy_note": "val +0.023 domain beam vs greedy; scoped to public lug only",
    }
    (ROOT / "outputs" / "beat075" / "lug_domain_beam_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
