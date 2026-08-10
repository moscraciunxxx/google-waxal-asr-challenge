#!/usr/bin/env python3
"""Validate the conservative train-lexicon Luganda split-join postprocess.

The candidate only joins adjacent tokens when the concatenation is present in
the train/validation Luganda unigram lexicon and the observed split bigram is
unseen.  This is evaluated against the production mms-lug-ft-v3 decoder on a
fixed WAXAL Luganda validation sample before any Phase-2 artifact is built.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.beat075_lug_domain_beam import load_wav  # noqa: F401  (shared path contract)
from scripts.eval_lug_ft_ab import prep_audio
from scripts.mms_adapter_ft import fix_mms_tokenizer, pick_device
from scripts.phase3_text_norm_ablations import feat_D_join_lug_splits
from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.text_norm import normalize_text


@torch.inference_mode()
def greedy(model, processor, arr: np.ndarray, device: torch.device) -> str:
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return normalize_text(processor.decode(ids)) or "."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "beat075" / "lug_splitjoin_val.json",
    )
    args = ap.parse_args()

    counts = json.loads((ROOT / "data" / "lms" / "lug_counts.json").read_text())
    uni = {
        str(w): int(c)
        for w, c in counts["uni"].items()
        if not str(w).startswith("<")
    }
    bi = {str(k): int(v) for k, v in counts["bi"].items()}

    val = load_hf_asr_split("lug", "validation")
    idx = list(range(len(val)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]

    refs: list[str] = []
    hyps: list[str] = []
    device = pick_device(args.device)
    ckpt = ROOT / "checkpoints" / "mms-lug-ft-v3"
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    processor = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
    fix_mms_tokenizer(processor, "lug")
    model = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
    model.to(device).eval()

    for i in idx:
        ex = val[int(i)]
        refs.append(normalize_text(str(ex.get("transcription") or "")) or ".")
        arr = prep_audio(ex)
        hyps.append(greedy(model, processor, arr, device))

    fixed = [feat_D_join_lug_splits(h, uni, bi) for h in hyps]
    base = score_pairs(refs, hyps)
    post = score_pairs(refs, fixed)
    changed = [
        {"index": int(i), "before": hyps[i], "after": fixed[i]}
        for i in range(len(hyps))
        if normalize_text(hyps[i]) != normalize_text(fixed[i])
    ]
    result = {
        "lang": "lug",
        "ckpt": str(ckpt),
        "n": len(refs),
        "seed": args.seed,
        "device": str(device),
        "baseline": {**base, "zindi": 1.0 - float(base["score"])},
        "splitjoin": {**post, "zindi": 1.0 - float(post["score"])},
        "delta_zindi": (1.0 - float(post["score"])) - (1.0 - float(base["score"])),
        "n_changed": len(changed),
        "changed": changed,
        "rule": "join a+b when train unigram(a+b)>=3 and train bigram(a,b)==0; otherwise join when unigram>=5 and unigram>=5*max(bigram,1) with bigram<=1",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["delta_zindi"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
