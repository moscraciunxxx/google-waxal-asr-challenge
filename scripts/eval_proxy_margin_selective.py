#!/usr/bin/env python3
"""Proxy A/B: margin-primary multi-hyp + selective upgrades (beam ach/nyn, FT-lug).

Deployable when LID maps into the same cand sets as openset; primary = cands[0]:
  if best_conf - second >= thr → max-conf lang, else primary.
Upgrades: lug→FT-lug, ach→KenLM beam+guard, nyn→beam+guard.

Baseline: pure max-conf multi-hyp (openset-equivalent).
Gate: candidate − baseline ≥ 0.01. No Phase-1 test gold.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, OUTPUT_DIR, PROJECT_ROOT, TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import fix_mms_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("proxy_margin_sel")

CANDS = {
    "ach": ["ach", "lug", "sog"],
    "lug": ["lug", "nyn", "sog"],
    "nyn": ["nyn", "lug", "sog"],
    "sog": ["sog", "lug"],
    "mas": ["mas", "lug"],
}
WAXAL = {L: f"waxal-benchmarking/mms-300m-waxal-{L}" for L in CANDS}
GATE = 0.01


def zindi(refs, hyps):
    s = score_pairs(refs, hyps)
    return {
        "n": int(s["n"]),
        "wer": float(s["wer"]),
        "cer": float(s["cer"]),
        "zindi_est": float(1.0 - s["score"]),
    }


def mk_dec(proc, lang: str, alpha: float, beta: float):
    vocab = proc.tokenizer.get_vocab()
    id2 = {i: t for t, i in vocab.items()}
    labels = [id2[i] for i in range(len(id2))]
    lm = ROOT / "data" / "lms" / f"{lang}_2gram.arpa"
    uni = ROOT / "data" / "lms" / f"{lang}_unigrams.txt"
    unigrams = [u for u in uni.read_text().splitlines() if u.strip()]
    return build_ctcdecoder(
        labels, kenlm_model_path=str(lm), unigrams=unigrams, alpha=alpha, beta=beta
    )


@torch.inference_mode()
def beam_decode(model, proc, dec, arr, sr, device, beam_width: int) -> str:
    arr = np.asarray(arr, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    arr = arr / (float(np.max(np.abs(arr)) + 1e-9))
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
    text = dec.decode(logits, beam_width=beam_width).replace("|", " ")
    return normalize_text(text) or "."


def guard(greedy: str, beamed: str) -> str:
    gw = max(1, len(greedy.split()))
    bw = max(1, len(beamed.split()))
    r = bw / gw
    return beamed if 0.5 <= r <= 2.0 else greedy


def pick(scored, cands, thr):
    scored = sorted(scored, key=lambda x: x[2], reverse=True)
    best_L, best_h, best_c = scored[0]
    second_c = scored[1][2] if len(scored) > 1 else -1e9
    margin = best_c - second_c
    if thr is None or margin >= thr:
        return best_L, best_h, margin, "maxconf" if thr is None else "margin_ok"
    for L, h, c in scored:
        if L == cands[0]:
            return L, h, margin, "primary_fb"
    return best_L, best_h, margin, "primary_missing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha-ach", type=float, default=0.2)
    ap.add_argument("--alpha-nyn", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "proxy_margin_selective.json")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    logger.info("device %s", device)
    proxy = pd.read_csv(ROOT / "data" / "proxy_val_index.csv")

    audio, refs, langs = {}, {}, {}
    for lang in ["ach", "lug", "nyn", "sog", "mas"]:
        sub = proxy[proxy.language == lang].head(40)
        ds = load_hf_asr_split(lang, "validation")
        want = {normalize_text(str(r.transcription)): str(r.id) for _, r in sub.iterrows()}
        for i in range(len(ds)):
            row = ds[i]
            t = normalize_text(str(row.get("transcription") or row.get("text") or ""))
            if t in want:
                uid = want[t]
                a = row["audio"]
                audio[uid] = (np.asarray(a["array"], dtype=np.float32), int(a["sampling_rate"]))
                refs[uid] = t
                langs[uid] = lang
                del want[t]
                if not want:
                    break
    ids = sorted(audio.keys())
    logger.info("matched %d", len(ids))

    cache = {}

    def get(lang):
        if lang not in cache:
            mid = WAXAL[lang]
            logger.info("load %s", mid)
            p = AutoProcessor.from_pretrained(mid)
            m = Wav2Vec2ForCTC.from_pretrained(mid).to(device).eval()
            cache[lang] = (m, p)
        return cache[lang]

    ft_p = AutoProcessor.from_pretrained(str(CHECKPOINT_DIR / "mms-lug-ft-v2"), local_files_only=True)
    ft_m = Wav2Vec2ForCTC.from_pretrained(str(CHECKPOINT_DIR / "mms-lug-ft-v2"), local_files_only=True)
    fix_mms_tokenizer(ft_p, "lug")
    ft_m.to(device).eval()

    ach_m, ach_p = get("ach")
    nyn_m, nyn_p = get("nyn")
    dec_ach = mk_dec(ach_p, "ach", args.alpha_ach, args.beta)
    dec_nyn = mk_dec(nyn_p, "nyn", args.alpha_nyn, args.beta)

    records = []
    for uid in tqdm(ids, desc="decode"):
        arr, sr = audio[uid]
        tlang = langs[uid]
        cands = CANDS[tlang]
        hyps = {}
        for L in set(cands):
            m, p = get(L)
            h, conf = transcribe_waveform(m, p, arr, sr, device=device, return_confidence=True)
            hyps[L] = (normalize_text(h) or ".", float(conf))
        hft, cft = transcribe_waveform(ft_m, ft_p, arr, sr, device=device, return_confidence=True)
        hyps["ft_lug"] = (normalize_text(hft) or ".", float(cft))
        g = hyps.get("ach", (".", 0.0))[0]
        b = beam_decode(ach_m, ach_p, dec_ach, arr, sr, device, args.beam)
        hyps["ach_beam"] = (guard(g if g != "." else b, b), 0.0)
        g = hyps.get("nyn", (".", 0.0))[0]
        b = beam_decode(nyn_m, nyn_p, dec_nyn, arr, sr, device, args.beam)
        hyps["nyn_beam"] = (guard(g if g != "." else b, b), 0.0)
        records.append({"id": uid, "lang": tlang, "ref": refs[uid], "hyps": hyps, "cands": cands})

    def apply(r, thr):
        scored = [(L, r["hyps"][L][0], r["hyps"][L][1]) for L in r["cands"] if L in r["hyps"]]
        return pick(scored, r["cands"], thr)

    def upgrade(r, L, h):
        if L == "lug":
            return r["hyps"]["ft_lug"][0], "ft_lug"
        if L == "ach":
            return r["hyps"]["ach_beam"][0], "ach_beam"
        if L == "nyn":
            return r["hyps"]["nyn_beam"][0], "nyn_beam"
        return h, L

    ref_list = [r["ref"] for r in records]
    recipes = {}

    hyps = []
    for r in records:
        L, h, _, _ = apply(r, None)
        hyps.append(h)
    recipes["multihyp"] = zindi(ref_list, hyps)

    hyps = []
    for r in records:
        L, h, _, _ = apply(r, None)
        h2, _ = upgrade(r, L, h)
        hyps.append(h2)
    recipes["multihyp_selective"] = zindi(ref_list, hyps)

    for thr in [0.005, 0.01, 0.02]:
        hyps = []
        acc = 0
        for r in records:
            L, h, _, _ = apply(r, thr)
            acc += int(L == r["lang"])
            hyps.append(h)
        m = zindi(ref_list, hyps)
        m["route_acc"] = acc / max(1, len(records))
        recipes[f"margin_{thr}"] = m

        hyps = []
        sources = {}
        for r in records:
            L, h, _, _ = apply(r, thr)
            h2, src = upgrade(r, L, h)
            hyps.append(h2)
            sources[src] = sources.get(src, 0) + 1
        m = zindi(ref_list, hyps)
        m["sources"] = sources
        recipes[f"margin_{thr}_selective"] = m

    base = recipes["multihyp"]["zindi_est"]
    prior = recipes["multihyp_selective"]["zindi_est"]
    for name, m in recipes.items():
        if name == "multihyp":
            continue
        m["delta_vs_multihyp"] = m["zindi_est"] - base
        m["delta_vs_selective"] = m["zindi_est"] - prior
        m["gate_pass"] = m["delta_vs_multihyp"] >= GATE

    best = max((n for n in recipes if n != "multihyp"), key=lambda n: recipes[n]["zindi_est"])
    out = {
        "n": len(records),
        "params": {
            "alpha_ach": args.alpha_ach,
            "alpha_nyn": args.alpha_nyn,
            "beta": args.beta,
            "beam": args.beam,
        },
        "baseline": "multihyp",
        "recipes": recipes,
        "best": best,
        "best_zindi": recipes[best]["zindi_est"],
        "gate_delta": GATE,
        "note": (
            "On proxy, cands[0]==true_lang so high margin thr ≈ oracle. "
            "Deploy thr=0.01 is the planned Phase-2 setting (primary=LID-mapped first cand)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    logger.info("WROTE %s best=%s %.4f", args.out, best, out["best_zindi"])
    summary = {
        k: {
            kk: vv
            for kk, vv in v.items()
            if kk in ("n", "wer", "cer", "zindi_est", "delta_vs_multihyp", "delta_vs_selective", "gate_pass", "route_acc")
        }
        for k, v in recipes.items()
    }
    print(json.dumps({"best": best, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
