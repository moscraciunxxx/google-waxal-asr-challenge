#!/usr/bin/env python3
"""Fixed offline proxy A/B gate: openset-equivalent vs candidate recipes.

Criterion (goal plan): candidate zindi_est - baseline zindi_est >= 0.01
on the same clips and same scoring code (src.metrics.score_pairs).

Baseline (openset-equivalent): multi-hyp CTC conf among waxal-300m using
true_lang-mapped candidate sets matching scripts/run_phase2_openset.py:

  ach (luo-proxy): ach|lug|sog
  lug: lug|nyn|sog
  nyn: nyn|lug|sog   # openset uses nyn when in WAXAL300 as single; we use multi for nyn like lug family
  sog: sog|lug
  mas: mas|lug

Candidate recipes (no cross-family conf-mix):
  A) hard_lang_plus_ft_lug: true-lang waxal; if true==lug use mms-lug-ft-v2
  B) hard_lang_plus_ft_lug_achft: true-lang waxal-ach-lmhead if true==ach else A
  C) openset_multihyp_then_upgrade: multihyp as baseline, but if pick lang==lug
     replace with FT; if pick lang==ach replace with ach-lmhead (same-family upgrades)

Never uses Phase-1 test gold for training.
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
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, OUTPUT_DIR, PROJECT_ROOT
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import fix_mms_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("proxy_ab_gate")

PROXY_CSV = PROJECT_ROOT / "data" / "proxy_val_index.csv"
GATE_DELTA = 0.01

CANDS = {
    "ach": ["ach", "lug", "sog"],
    "lug": ["lug", "nyn", "sog"],
    "nyn": ["nyn", "lug", "sog"],
    "sog": ["sog", "lug"],
    "mas": ["mas", "lug"],
}


def zindi_from_pairs(refs, hyps):
    s = score_pairs(refs, hyps)
    return {
        "n": int(s["n"]),
        "wer": float(s["wer"]),
        "cer": float(s["cer"]),
        "error": float(s["score"]),
        "zindi_est": float(1.0 - s["score"]),
    }


def load_waxal(lang: str, device: torch.device):
    mid = f"waxal-benchmarking/mms-300m-waxal-{lang}"
    p = AutoProcessor.from_pretrained(mid, local_files_only=True)
    m = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True).to(device).eval()
    return m, p


def load_ft_lug(device: torch.device):
    ckpt = CHECKPOINT_DIR / "mms-lug-ft-v2"
    p = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
    m = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
    fix_mms_tokenizer(p, "lug")
    m.to(device).eval()
    return m, p


def load_ach_lmhead(device: torch.device):
    ckpt = CHECKPOINT_DIR / "waxal-ach-lmhead-ft"
    if not ckpt.exists():
        return None
    p = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
    m = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True).to(device).eval()
    return m, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "proxy_ab_gate.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    logger.info("device=%s", device)
    idx = pd.read_csv(PROXY_CSV)
    idx["id"] = idx["id"].astype(str)
    langs = sorted(idx["language"].unique().tolist())

    waxal = {}
    for L in langs:
        for c in CANDS.get(L, [L]):
            if c not in waxal and c in {"ach", "nyn", "lug", "sog", "mas", "lin", "sna"}:
                logger.info("load waxal %s", c)
                waxal[c] = load_waxal(c, device)
    # ensure all cands
    for L in ["ach", "lug", "nyn", "sog", "mas"]:
        if L not in waxal:
            logger.info("load waxal %s", L)
            waxal[L] = load_waxal(L, device)

    logger.info("load ft lug")
    ft = load_ft_lug(device)
    ach_ft = load_ach_lmhead(device)
    logger.info("ach_lmhead=%s", ach_ft is not None)

    clips = []
    for lang in langs:
        want = set(idx[idx.language == lang].id)
        ds = load_hf_asr_split(lang, "validation", max_samples=None)
        found = 0
        for i in range(len(ds)):
            ex = ds[i]
            eid = str(ex["id"])
            if eid not in want:
                continue
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"]["sampling_rate"])
            ref = normalize_text(ex["transcription"])
            clips.append({"id": eid, "lang": lang, "arr": arr, "sr": sr, "ref": ref})
            found += 1
            if found >= len(want):
                break
        logger.info("%s found %d", lang, found)

    assert len(clips) == len(idx), (len(clips), len(idx))

    # Cache hyps: for each clip, waxal hyp+conf per needed lang, ft, ach_ft
    rows = []
    for c in clips:
        need = set(CANDS[c["lang"]])
        need.add(c["lang"])
        hyps = {}
        for L in need:
            h, conf = transcribe_waveform(
                *waxal[L], c["arr"], c["sr"], device=device, return_confidence=True
            )
            hyps[L] = (normalize_text(h) or ".", float(conf))
        h_ft, c_ft = transcribe_waveform(
            ft[0], ft[1], c["arr"], c["sr"], device=device, return_confidence=True
        )
        hyps["ft_lug"] = (normalize_text(h_ft) or ".", float(c_ft))
        if ach_ft is not None:
            ha, ca = transcribe_waveform(
                ach_ft[0], ach_ft[1], c["arr"], c["sr"], device=device, return_confidence=True
            )
            hyps["ach_lmhead"] = (normalize_text(ha) or ".", float(ca))
        rows.append({"id": c["id"], "lang": c["lang"], "ref": c["ref"], "hyps": hyps})

    def multihyp(hyps, true_lang):
        cands = CANDS[true_lang]
        best_l = max(cands, key=lambda L: hyps[L][1])
        return hyps[best_l][0], best_l

    def recipe_baseline(r):
        h, L = multihyp(r["hyps"], r["lang"])
        return h

    def recipe_A(r):
        if r["lang"] == "lug":
            return r["hyps"]["ft_lug"][0]
        return r["hyps"][r["lang"]][0]

    def recipe_B(r):
        if r["lang"] == "lug":
            return r["hyps"]["ft_lug"][0]
        if r["lang"] == "ach" and "ach_lmhead" in r["hyps"]:
            return r["hyps"]["ach_lmhead"][0]
        return r["hyps"][r["lang"]][0]

    def recipe_C(r):
        h, L = multihyp(r["hyps"], r["lang"])
        if L == "lug":
            return r["hyps"]["ft_lug"][0]
        if L == "ach" and "ach_lmhead" in r["hyps"]:
            return r["hyps"]["ach_lmhead"][0]
        return h

    def recipe_D_margin_primary(r):
        # margin: best conf among cands if margin>=0.01 else primary (first)
        cands = CANDS[r["lang"]]
        scored = [(L, r["hyps"][L][0], r["hyps"][L][1]) for L in cands]
        scored.sort(key=lambda x: x[2], reverse=True)
        best_L, best_h, best_c = scored[0]
        second_c = scored[1][2] if len(scored) > 1 else -1e9
        if best_c - second_c >= 0.01:
            L, h = best_L, best_h
        else:
            L, h = cands[0], r["hyps"][cands[0]][0]
        if L == "lug":
            return r["hyps"]["ft_lug"][0]
        if L == "ach" and "ach_lmhead" in r["hyps"]:
            return r["hyps"]["ach_lmhead"][0]
        return h

    recipes = {
        "openset_multihyp_conf": recipe_baseline,
        "A_hard_lang_ft_lug": recipe_A,
        "B_hard_lang_ft_lug_achft": recipe_B,
        "C_multihyp_upgrade_ft": recipe_C,
        "D_margin_primary_upgrade": recipe_D_margin_primary,
    }

    refs = [r["ref"] for r in rows]
    results = {}
    for name, fn in recipes.items():
        hyps = [fn(r) for r in rows]
        results[name] = zindi_from_pairs(refs, hyps)
        logger.info("%s %s", name, results[name])

    baseline = results["openset_multihyp_conf"]["zindi_est"]
    winners = []
    for name, m in results.items():
        if name == "openset_multihyp_conf":
            continue
        delta = m["zindi_est"] - baseline
        m["delta_vs_baseline"] = delta
        m["gate_pass"] = delta >= GATE_DELTA
        if m["gate_pass"]:
            winners.append((name, delta, m["zindi_est"]))

    winners.sort(key=lambda x: -x[1])
    out = {
        "proxy_csv": str(PROXY_CSV),
        "n": len(rows),
        "langs": langs,
        "gate_delta": GATE_DELTA,
        "baseline": "openset_multihyp_conf",
        "results": results,
        "winners": [{"name": n, "delta": d, "zindi_est": z} for n, d, z in winners],
        "best_winner": winners[0][0] if winners else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    logger.info("WROTE %s best=%s", args.out, out["best_winner"])
    print(json.dumps({"best": out["best_winner"], "winners": out["winners"], "baseline_zindi": baseline}, indent=2))


if __name__ == "__main__":
    main()
