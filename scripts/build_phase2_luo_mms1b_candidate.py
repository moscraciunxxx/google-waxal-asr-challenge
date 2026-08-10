#!/usr/bin/env python3
"""Phase-2 candidate: native MMS-1B luo adapter for LID=luo mass.

Problem: ~785/1500 clips are LID-predicted luo; champion has no waxal-300m-luo
and multi-hyps among ach|lug|sog. facebook/mms-1b-all has a real **luo** adapter.

Recipe (no cross-family CTC conf-mix):
  - lid_lang == luo  → hard re-decode with mms-1b-all + adapter luo
  - optional: decode_lang == lug (non-luo lids) → mms-lug-ft-v2 hard upgrade
  - optional: decode_lang == ach (non-luo lids) → waxal-ach-lmhead-ft
  - else keep openset hyp

Open-source only; no Phase-1 test gold training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, OUTPUT_DIR, PROJECT_ROOT
from src.mms_infer import load_mms, pick_device, set_lang, transcribe_waveform
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import fix_mms_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("luo_mms1b")

AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
OPENSET = OUTPUT_DIR / "phase2_openset_detail.csv"


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upgrade-lug", action="store_true", default=True)
    ap.add_argument("--upgrade-ach", action="store_true", default=True)
    ap.add_argument("--no-upgrade-lug", action="store_true")
    ap.add_argument("--no-upgrade-ach", action="store_true")
    ap.add_argument("--min-p1-luo", type=float, default=0.0, help="Only luo upgrade if p1>=this")
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "submission_phase2_luo_mms1b_candidate.csv",
    )
    ap.add_argument(
        "--detail",
        type=Path,
        default=OUTPUT_DIR / "phase2_luo_mms1b_detail.csv",
    )
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.no_upgrade_lug:
        args.upgrade_lug = False
    if args.no_upgrade_ach:
        args.upgrade_ach = False

    device = torch.device(args.device) if args.device else pick_device()
    op = pd.read_csv(OPENSET)
    op["ID"] = op["ID"].astype(str)

    logger.info("load mms-1b-all + luo adapter")
    mms, mms_p, device = load_mms(device=device)
    set_lang(mms, mms_p, "luo")

    ft = None
    ach = None
    if args.upgrade_lug:
        ckpt = CHECKPOINT_DIR / "mms-lug-ft-v2"
        if ckpt.exists():
            logger.info("load %s", ckpt)
            p = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
            m = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
            fix_mms_tokenizer(p, "lug")
            m.to(device).eval()
            ft = (m, p)
    if args.upgrade_ach:
        ckpt = CHECKPOINT_DIR / "waxal-ach-lmhead-ft"
        if ckpt.exists():
            logger.info("load %s", ckpt)
            p = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
            m = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True).to(device).eval()
            ach = (m, p)

    rows = []
    counts = Counter()
    for _, r in tqdm(op.iterrows(), total=len(op), desc="luo-mms1b"):
        uid = str(r.ID)
        base = normalize_text(str(r.prediction)) or "."
        dlang = str(r.decode_lang)
        lid = str(r.lid_lang)
        p1 = float(r.p1) if pd.notna(r.p1) else 0.0
        hyp = base
        src = "openset_keep"

        if lid == "luo" and p1 >= args.min_p1_luo:
            arr, sr = load_wav(AUDIO / f"{uid}.wav")
            # luo adapter already set once after load; do not re-set_lang per clip
            # (load_adapter hits network HEAD and is very slow).
            hyp = normalize_text(transcribe_waveform(mms, mms_p, arr, sr, device=device)) or "."
            src = "mms1b_luo"
        elif args.upgrade_lug and dlang == "lug" and ft is not None and lid != "luo":
            arr, sr = load_wav(AUDIO / f"{uid}.wav")
            hyp = normalize_text(transcribe_waveform(ft[0], ft[1], arr, sr, device=device)) or "."
            src = "ft_lug"
        elif args.upgrade_ach and dlang == "ach" and ach is not None and lid != "luo":
            arr, sr = load_wav(AUDIO / f"{uid}.wav")
            hyp = normalize_text(transcribe_waveform(ach[0], ach[1], arr, sr, device=device)) or "."
            src = "ach_lmhead"

        counts[src] += 1
        rows.append(
            {
                "ID": uid,
                "prediction": hyp,
                "source": src,
                "lid_lang": lid,
                "p1": p1,
                "decode_lang": dlang,
                "openset_prediction": base,
                "changed": int(hyp != base),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.detail, index=False)
    build_submission(df[["ID", "prediction"]], sample_path=SAMPLE, out_path=args.out)
    rep = check_submission(args.out, SAMPLE)
    o = pd.read_csv(PROJECT_ROOT / "submission_phase2_openset.csv").set_index("ID")["Target"].astype(str)
    n = df.set_index("ID")["prediction"].astype(str)
    rep.update(
        {
            "source_counts": counts.most_common(),
            "n_changed": int((o.reindex(n.index) != n).sum()),
            "method": "openset + hard mms-1b-luo for lid=luo + optional lug/ach upgrades on non-luo",
            "min_p1_luo": args.min_p1_luo,
        }
    )
    (OUTPUT_DIR / "phase2_luo_mms1b_check.json").write_text(json.dumps(rep, indent=2, default=str))
    logger.info("%s", rep)
    print("UPLOAD", args.out)
    print("CHANGED", rep["n_changed"], dict(counts))


if __name__ == "__main__":
    main()
