#!/usr/bin/env python3
"""Validate an Acholi-vs-Luganda discriminator on data where the label is known.

Motivation: the 3-component mixture fit on the 785 lid=luo rows implies roughly
619 Acholi / 155 Luganda, but the openset router sends 308 of them to the Luganda
decoder. If ~150 rows are decoded in the wrong language that costs far more than the
entire Dholuo-swap program. Before touching any row we must show a discriminator can
actually tell these two apart.

Signal deliberately independent of the router: the router chose decode_lang by CTC
confidence argmax inside the waxal-300m family. Here we decode each clip with BOTH
production decoders and ask which language's KenLM better explains its own decode
(per-character logprob), plus the CTC margin as a secondary feature.

Ship gate: this only becomes actionable if accuracy on held-out val is high AND the
rows it would flip are ones where switching actually improves the score. Both are
measured here; nothing is written to a submission by this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import kenlm
import numpy as np
import pandas as pd
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
logger = logging.getLogger("achlug")

ACH = "waxal-benchmarking/mms-300m-waxal-ach"
LUG_CKPT = CHECKPOINT_DIR / "mms-lug-ft-v3"
OUT = ROOT / "outputs" / "next_iter" / "achlug_discriminator.csv"
OUTJ = ROOT / "outputs" / "next_iter" / "achlug_discriminator.json"


def build(lang: str, path: str, device, alpha: float):
    proc = AutoProcessor.from_pretrained(path, local_files_only=os.path.isdir(str(path)))
    model = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=os.path.isdir(str(path))).to(device).eval()
    vocab = proc.tokenizer.get_vocab()
    labels = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    uni = ROOT / "data" / "lms" / f"{lang}_unigrams.txt"
    unigrams = [w.strip() for w in uni.read_text().splitlines() if w.strip()] if uni.exists() else None
    dec = build_ctcdecoder(labels, kenlm_model_path=str(ROOT / "data" / "lms" / f"{lang}_2gram.arpa"),
                           unigrams=unigrams, alpha=alpha, beta=0.5)
    return proc, model, dec


@torch.inference_mode()
def decode(proc, model, dec, arr, device):
    """Returns (hypothesis, mean per-frame CTC logprob of the greedy path)."""
    lg = model(proc(arr, sampling_rate=TARGET_SR, return_tensors="pt",
                    padding=True).input_values.to(device)).logits[0].float().cpu().numpy()
    lp = lg - np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lg.max(-1, keepdims=True)
    conf = float(lp.max(-1).mean())
    g = normalize_text(proc.decode(torch.tensor(lg.argmax(-1)))) or "."
    b = normalize_text(dec.decode(lg, beam_width=100).replace("|", " ")) or "."
    gw, bw = max(1, len(g.split())), max(1, len(b.split()))
    return (b if (0.5 <= bw / gw <= 2.0 and b.strip()) else g), conf


def lm_percharm(lm, text: str) -> float:
    t = text.strip()
    if not t:
        return -99.0
    return float(lm.score(t, bos=True, eos=True)) / max(1, len(t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)

    lm_ach = kenlm.Model(str(ROOT / "data" / "lms" / "ach_2gram.arpa"))
    lm_lug = kenlm.Model(str(ROOT / "data" / "lms" / "lug_2gram.arpa"))

    proc_a, model_a, dec_a = build("ach", ACH, device, 0.2)
    proc_l, model_l, dec_l = build("lug", str(LUG_CKPT), device, 0.2)

    rows = []
    for true_lang in ("ach", "lug"):
        val = load_hf_asr_split(true_lang, "validation")
        idx = list(range(len(val)))
        random.Random(42).shuffle(idx)
        for k, i in enumerate(idx[: args.n]):
            ex = val[i]
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"].get("sampling_rate") or TARGET_SR)
            if sr != TARGET_SR:
                import librosa

                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
            arr = arr / float(np.max(np.abs(arr)) + 1e-9)
            ref = normalize_text(ex.get("transcription") or "") or "."
            h_a, c_a = decode(proc_a, model_a, dec_a, arr, device)
            h_l, c_l = decode(proc_l, model_l, dec_l, arr, device)
            rows.append({
                "true_lang": true_lang, "ref": ref,
                "hyp_ach": h_a, "hyp_lug": h_l,
                "ctc_ach": c_a, "ctc_lug": c_l,
                "lm_ach": lm_percharm(lm_ach, h_a), "lm_lug": lm_percharm(lm_lug, h_l),
                "s_ach": score_pairs([ref], [h_a])["score"],
                "s_lug": score_pairs([ref], [h_l])["score"],
            })
            if (k + 1) % 25 == 0:
                logger.info("%s %d/%d", true_lang, k + 1, args.n)

    df = pd.DataFrame(rows)
    df["d_lm"] = df.lm_ach - df.lm_lug
    df["d_ctc"] = df.ctc_ach - df.ctc_lug
    df.to_csv(OUT, index=False)

    res = {"n_per_lang": args.n}
    print("\n=== oracle: how much is at stake per row ===")
    for tl in ("ach", "lug"):
        s = df[df.true_lang == tl]
        z_own = 1 - s[f"s_{tl}"].mean()
        z_oth = 1 - s[f"s_{'lug' if tl == 'ach' else 'ach'}"].mean()
        print(f"  true {tl}: own decoder zindi {z_own:.4f} | other decoder {z_oth:.4f} | cost of misroute {z_own - z_oth:+.4f}")
        res[f"cost_misroute_{tl}"] = float(z_own - z_oth)

    print("\n=== discriminator accuracy (predict ach when score > 0) ===")
    for feat in ("d_lm", "d_ctc"):
        pred = np.where(df[feat] > 0, "ach", "lug")
        acc = float((pred == df.true_lang).mean())
        acc_a = float((pred[df.true_lang == "ach"] == "ach").mean())
        acc_l = float((pred[df.true_lang == "lug"] == "lug").mean())
        print(f"  {feat:>6}: overall {acc:.3f}  (true-ach {acc_a:.3f}, true-lug {acc_l:.3f})")
        res[f"acc_{feat}"] = {"overall": acc, "ach": acc_a, "lug": acc_l}

    # best single threshold on d_lm
    best = max(((float((np.where(df.d_lm > t, "ach", "lug") == df.true_lang).mean()), t)
                for t in np.percentile(df.d_lm, np.arange(1, 100))), key=lambda x: x[0])
    print(f"  best d_lm threshold {best[1]:+.4f} -> accuracy {best[0]:.3f}")
    res["best_d_lm"] = {"thr": float(best[1]), "acc": float(best[0])}

    OUTJ.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT} and {OUTJ}")


if __name__ == "__main__":
    main()
