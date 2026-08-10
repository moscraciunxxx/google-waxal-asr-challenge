#!/usr/bin/env python3
"""Stage 2 (.venv-omni): transcribe probe sets with omniASR and score vs baselines.

Run with:
  DYLD_LIBRARY_PATH=$PWD/.venv/sndfile_shim .venv-omni/bin/python scripts/probe_omniasr.py \
      --model omniASR_LLM_1B_v2

Scores zindi = 1 - 0.5*(WER+CER) on identical rows as the recorded baselines.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
from jiwer import cer, wer

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "outputs" / "next_iter" / "probe"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe_omni")

LANG_CODE = {"lug": "lug_Latn", "nyn": "nyn_Latn", "ach": "ach_Latn", "luo": "luo_Latn"}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s']+", flags=re.UNICODE)
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def normalize_text(text: str) -> str:
    # Exact mirror of src/text_norm.py (kept dependency-free for the omni venv)
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _CTRL_RE.sub("", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    text = re.sub(r"\s+'\s+", " ", text)
    text = text.strip("' ")
    return text


def zindi(refs, hyps):
    w = wer(refs, hyps)
    c = cer(refs, hyps)
    return {"wer": w, "cer": c, "zindi": 1.0 - 0.5 * (w + c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="omniASR_LLM_1B_v2")
    ap.add_argument("--langs", nargs="+", default=["lug", "nyn", "ach", "luo"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "next_iter" / "omniasr_probe.json")
    args = ap.parse_args()

    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    device = args.device
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    logger.info("loading %s on %s (%s)", args.model, device, dtype)
    t0 = time.time()
    pipe = ASRInferencePipeline(args.model, device=device, dtype=dtype)
    logger.info("loaded in %.1fs", time.time() - t0)

    results = {"model": args.model, "device": device}
    for lang in args.langs:
        d = PROBE / lang
        refs_df = pd.read_csv(d / "refs.csv")
        paths = [str(d / f"{int(i):03d}.wav") for i in refs_df["idx"]]
        refs = [normalize_text(r) for r in refs_df["ref"].astype(str)]
        base = [normalize_text(h) for h in refs_df["baseline_hyp"].astype(str)]
        logger.info("== %s: %d files ==", lang, len(paths))
        t0 = time.time()
        hyps_raw = pipe.transcribe(paths, lang=[LANG_CODE[lang]] * len(paths), batch_size=args.batch)
        el = time.time() - t0
        hyps = [normalize_text(h) or "." for h in hyps_raw]
        res_o = zindi(refs, hyps)
        res_b = zindi(refs, base)
        results[lang] = {
            "n": len(refs),
            "omni": res_o,
            "baseline": res_b,
            "delta_zindi": res_o["zindi"] - res_b["zindi"],
            "sec_per_utt": el / max(1, len(paths)),
        }
        logger.info("%s omni=%.4f baseline=%.4f delta=%+.4f (%.2fs/utt)",
                    lang, res_o["zindi"], res_b["zindi"], results[lang]["delta_zindi"],
                    results[lang]["sec_per_utt"])
        args.out.write_text(json.dumps(results, indent=2))
        pd.DataFrame({"idx": refs_df["idx"], "ref": refs, "omni": hyps, "baseline": base}).to_csv(
            d / f"omni_{args.model}.csv", index=False
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
