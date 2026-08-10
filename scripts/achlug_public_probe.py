#!/usr/bin/env python3
"""Run the ach-vs-lug discriminator on the PUBLIC-ELIGIBLE lug-routed rows.

The 229 flips the discriminator proposes on the 785 lid=luo rows are worth ~+0.05 on
the private leaderboard, but those rows are 100% private -- unmeasurable. The public
ladder also records that every previous routing rewrite lost points, so this thesis
must be tested before it is trusted.

The test: the 430 old rows whose LID was 'lug' (not 'luo') sit in the public-eligible
pool. Applying the identical discriminator to them yields flips that DO move the
public score. One upload then measures whether the mechanism transfers, at no risk to
the private-set decision.

Writes a decisions table only; the submission build is a separate explicit step.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import kenlm
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from scripts.mms_adapter_ft import pick_device
from scripts.eval_achlug_discriminator import ACH, LUG_CKPT, build, decode, lm_percharm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("achlug_pub")

AUDIO = ROOT / "data" / "phase2" / "audio"
OUT = ROOT / "outputs" / "next_iter" / "achlug_public_decisions.csv"
THR = 1.7344


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)

    luo_ids = set(pd.read_csv(ROOT / "outputs" / "next_iter" / "hybrid_agreement_785.csv").ID)
    detail = pd.read_csv(ROOT / "outputs" / "phase2_selective_v3_dual15_lugbeam_detail.csv")
    pool = detail[(~detail.ID.isin(luo_ids)) & (detail.decode_lang == "lug")].copy()
    logger.info("public-eligible lug-routed rows: %d", len(pool))

    lm_ach = kenlm.Model(str(ROOT / "data" / "lms" / "ach_2gram.arpa"))
    lm_lug = kenlm.Model(str(ROOT / "data" / "lms" / "lug_2gram.arpa"))
    proc_a, model_a, dec_a = build("ach", ACH, device, 0.2)
    proc_l, model_l, dec_l = build("lug", str(LUG_CKPT), device, 0.2)

    recs = []
    for k, r in enumerate(pool.itertuples(index=False)):
        arr, sr = sf.read(str(AUDIO / f"{r.ID}.wav"), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(-1)
        if sr != TARGET_SR:
            import librosa

            arr = librosa.resample(np.asarray(arr, np.float32), orig_sr=sr, target_sr=TARGET_SR)
        arr = arr / float(np.max(np.abs(arr)) + 1e-9)
        h_a, c_a = decode(proc_a, model_a, dec_a, arr, device)
        h_l, c_l = decode(proc_l, model_l, dec_l, arr, device)
        la, ll = lm_percharm(lm_ach, h_a), lm_percharm(lm_lug, h_l)
        recs.append({"ID": r.ID, "router_lang": "lug", "hyp_ach": h_a, "hyp_lug": h_l,
                     "lm_ach": la, "lm_lug": ll, "d_lm": la - ll,
                     "pred_lang": "ach" if (la - ll) > THR else "lug",
                     "margin": abs((la - ll) - THR)})
        if (k + 1) % 50 == 0:
            logger.info("%d/%d", k + 1, len(pool))
            pd.DataFrame(recs).to_csv(OUT, index=False)

    df = pd.DataFrame(recs)
    df.to_csv(OUT, index=False)
    flips = df[df.pred_lang == "ach"]
    print(f"\n=== public-eligible probe: {len(df)} lug-routed rows ===")
    print(f"discriminator flips to ach: {len(flips)}")
    for lo in (0.0, 0.5, 1.0, 2.0, 3.0):
        n = int((flips.margin >= lo).sum())
        # ~45% of non-luo rows land in the public 30% of 2392
        print(f"  margin>={lo:.1f}: {n:3d} flips  (~{n*0.45:.0f} in public; "
              f"expected public delta if correct {n*0.45*0.3247/718:+.4f})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
