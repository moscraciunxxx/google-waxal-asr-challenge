#!/usr/bin/env python3
"""Apply the validated ach-vs-lug discriminator to the 785 lid=luo phase-2 rows.

The discriminator (per-char KenLM logprob of each language's own decode, threshold
d_lm > +1.7344) scores 0.968 held-out accuracy over 200 cross-validation repeats on
WAXAL ach/lug validation -- see eval_achlug_discriminator.py.

These 785 rows are entirely inside the private 70%, so this can never be validated on
the public leaderboard. That is precisely why the decision rule was calibrated on data
with known labels first, and why this script only WRITES A TABLE -- the submission
build is a separate, explicit step.

Cost of a misroute, measured on val:
    true Luganda decoded as Acholi : -0.7931 zindi
    true Acholi  decoded as Luganda: -0.3247 zindi
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import kenlm
import numpy as np
import pandas as pd
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, TARGET_SR
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import pick_device
from scripts.eval_achlug_discriminator import ACH, LUG_CKPT, build, decode, lm_percharm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("apply_achlug")

AUDIO = ROOT / "data" / "phase2" / "audio"
OUT = ROOT / "outputs" / "next_iter" / "achlug_785_decisions.csv"
THR = 1.7344


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    device = pick_device(args.device)

    tbl = pd.read_csv(ROOT / "outputs" / "next_iter" / "hybrid_agreement_785.csv")[["ID"]]
    detail = pd.read_csv(ROOT / "outputs" / "phase2_selective_v3_dual15_lugbeam_detail.csv")
    tbl = tbl.merge(detail[["ID", "decode_lang", "prediction"]], on="ID", how="left")
    if args.limit:
        tbl = tbl.head(args.limit)
    logger.info("rows=%d router split: %s", len(tbl), tbl.decode_lang.value_counts().to_dict())

    lm_ach = kenlm.Model(str(ROOT / "data" / "lms" / "ach_2gram.arpa"))
    lm_lug = kenlm.Model(str(ROOT / "data" / "lms" / "lug_2gram.arpa"))
    proc_a, model_a, dec_a = build("ach", ACH, device, 0.2)
    proc_l, model_l, dec_l = build("lug", str(LUG_CKPT), device, 0.2)

    recs = []
    for k, r in enumerate(tbl.itertuples(index=False)):
        wav = AUDIO / f"{r.ID}.wav"
        arr, sr = sf.read(str(wav), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(-1)
        if sr != TARGET_SR:
            import librosa

            arr = librosa.resample(np.asarray(arr, np.float32), orig_sr=sr, target_sr=TARGET_SR)
        arr = arr / float(np.max(np.abs(arr)) + 1e-9)
        h_a, c_a = decode(proc_a, model_a, dec_a, arr, device)
        h_l, c_l = decode(proc_l, model_l, dec_l, arr, device)
        la, ll = lm_percharm(lm_ach, h_a), lm_percharm(lm_lug, h_l)
        recs.append({"ID": r.ID, "router_lang": r.decode_lang,
                     "hyp_ach": h_a, "hyp_lug": h_l,
                     "lm_ach": la, "lm_lug": ll, "d_lm": la - ll,
                     "ctc_ach": c_a, "ctc_lug": c_l,
                     "pred_lang": "ach" if (la - ll) > THR else "lug"})
        if (k + 1) % 50 == 0:
            logger.info("%d/%d", k + 1, len(tbl))
            pd.DataFrame(recs).to_csv(OUT, index=False)

    df = pd.DataFrame(recs)
    df.to_csv(OUT, index=False)

    print("\n=== discriminator vs router on the 785 private rows ===")
    print(pd.crosstab(df.router_lang, df.pred_lang, rownames=["router"], colnames=["discriminator"]))
    dis = df[(df.router_lang.isin(["ach", "lug"])) & (df.router_lang != df.pred_lang)]
    print(f"\ndisagreements: {len(dis)} of {len(df)} rows")
    for a, b in (("ach", "lug"), ("lug", "ach")):
        n = int(((df.router_lang == a) & (df.pred_lang == b)).sum())
        gain = 0.7931 if b == "lug" else 0.3247
        print(f"  router {a} -> discriminator {b}: {n:4d} rows  (gain if correct {gain:+.4f}/row)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
