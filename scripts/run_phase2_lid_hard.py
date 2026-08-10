#!/usr/bin/env python3
"""Phase-2 candidate: LID-hard routing to reduce multi-hyp conf sink.

Hypothesis (from proxy Phase-2 baselines):
- Expanding multi-hyp via CTC conf *hurts* (lug conf sink).
- Tight / oracle routing ≈ 0.74 on proxy; broad multi-hyp ≈ 0.68.
- Champion public 0.529 uses multi-hyp for luo and lug.

This script starts from champion openset detail and **only changes** rows where
we can apply a safer rule:

1. lid_lang == luo  → keep champion multi-hyp hyp (no luo waxal model).
2. lid_lang == lug and p1 >= --p1-thresh:
     - option hard_lug: force waxal-lug greedy (re-decode) OR keep champion if
       already decode_lang==lug
3. lid_lang with a waxal model and p1 >= thresh → single-lang waxal (re-decode)
4. else keep champion prediction.

Default mode ``tighten``:
- For lid=lug with p1 >= thresh AND champion decode_lang in {nyn,sog}:
  re-decode with waxal-lug only and keep if we want force-true-LID.
- Actually default is safer: for high-p1 lid that has waxal-300m, re-decode
  single-lang and replace champion only when new hyp differs (always replace
  for fair A/B).

Produces:
  submission_phase2_lid_hard.csv
  outputs/phase2_lid_hard_detail.csv

No KenLM. No cross-family conf. No mms-1b mix.
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

from src.config import OUTPUT_DIR, PROJECT_ROOT, TARGET_SR
from src.mms_infer import pick_device, transcribe_waveform
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase2_lid_hard")

PHASE2_AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
PHASE2_SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
OPENSET_DETAIL = OUTPUT_DIR / "phase2_openset_detail.csv"

WAXAL300 = {
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
    "lin": "waxal-benchmarking/mms-300m-waxal-lin",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "ach": "waxal-benchmarking/mms-300m-waxal-ach",
    "sog": "waxal-benchmarking/mms-300m-waxal-sog",
    "nyn": "waxal-benchmarking/mms-300m-waxal-nyn",
    "mas": "waxal-benchmarking/mms-300m-waxal-mas",
}


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    if peak > 0:
        arr = arr / peak
    return arr, int(sr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--openset-detail", type=Path, default=OPENSET_DETAIL)
    p.add_argument("--p1-thresh", type=float, default=0.95)
    p.add_argument(
        "--mode",
        choices=["lid_single", "keep_luo_multihyp", "ft_lug_hard"],
        default="keep_luo_multihyp",
        help=(
            "keep_luo_multihyp: luo keep champion; high-p1 other LID with waxal → single redecode. "
            "lid_single: always single if waxal exists else keep. "
            "ft_lug_hard: lid=lug high p1 → mms-lug-ft-v2; else champion."
        ),
    )
    p.add_argument("--device", default=None)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "submission_phase2_lid_hard.csv",
    )
    p.add_argument(
        "--detail",
        type=Path,
        default=OUTPUT_DIR / "phase2_lid_hard_detail.csv",
    )
    args = p.parse_args()

    if not args.openset_detail.exists():
        raise SystemExit(f"Need champion detail {args.openset_detail}")

    device = torch.device(args.device) if args.device else pick_device()
    logger.info("device=%s mode=%s p1_thresh=%.3f", device, args.mode, args.p1_thresh)

    op = pd.read_csv(args.openset_detail)
    op["ID"] = op["ID"].astype(str)
    if args.max_files:
        op = op.head(args.max_files)

    cache: dict[str, tuple] = {}

    def get_waxal(lang: str):
        if lang not in cache:
            mid = WAXAL300[lang]
            logger.info("Loading %s", mid)
            proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
            model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
            model.to(device).eval()
            cache[lang] = (model, proc)
        return cache[lang]

    ft = None
    if args.mode == "ft_lug_hard":
        from scripts.mms_adapter_ft import fix_mms_tokenizer
        from src.config import CHECKPOINT_DIR

        ckpt = CHECKPOINT_DIR / "mms-lug-ft-v2"
        logger.info("Loading FT %s", ckpt)
        proc = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
        fix_mms_tokenizer(proc, "lug")
        model.to(device).eval()
        ft = (model, proc)

    rows = []
    n_changed = 0
    n_redecode = 0
    for _, r in tqdm(op.iterrows(), total=len(op), desc="lid-hard"):
        uid = str(r.ID)
        lid = str(r.lid_lang)
        p1 = float(r.p1) if pd.notna(r.p1) else 0.0
        base = normalize_text(str(r.prediction)) or "."
        base_lang = str(r.decode_lang)
        action = "keep_champion"
        hyp = base
        dlang = base_lang
        src = "openset"

        redecode_lang = None
        if args.mode == "ft_lug_hard":
            if lid == "lug" and p1 >= args.p1_thresh and ft is not None:
                redecode_lang = "ft_lug"
        elif args.mode == "lid_single":
            if lid in WAXAL300 and p1 >= args.p1_thresh:
                redecode_lang = lid
            # luo has no waxal → keep
        else:  # keep_luo_multihyp
            if lid == "luo":
                redecode_lang = None  # keep multi-hyp champion
            elif lid in WAXAL300 and p1 >= args.p1_thresh:
                # Force single-lang waxal for high-confidence LID (cuts nyn/sog conf flips)
                redecode_lang = lid
            elif lid not in WAXAL300 and lid != "luo":
                # rare LID: keep champion
                redecode_lang = None

        if redecode_lang == "ft_lug" and ft is not None:
            arr, sr = load_wav(PHASE2_AUDIO / f"{uid}.wav")
            hyp = transcribe_waveform(ft[0], ft[1], arr, sr, device=device)
            dlang, src, action = "lug", "ft_v2", "ft_lug"
            n_redecode += 1
        elif redecode_lang and redecode_lang in WAXAL300:
            arr, sr = load_wav(PHASE2_AUDIO / f"{uid}.wav")
            m, proc = get_waxal(redecode_lang)
            hyp = transcribe_waveform(m, proc, arr, sr, device=device)
            dlang, src, action = redecode_lang, f"waxal_{redecode_lang}", "lid_single"
            n_redecode += 1

        hyp = normalize_text(hyp) or "."
        changed = int(hyp != base)
        n_changed += changed
        rows.append(
            {
                "ID": uid,
                "prediction": hyp,
                "lid_lang": lid,
                "p1": p1,
                "decode_lang": dlang,
                "source": src,
                "action": action,
                "changed": changed,
                "openset_prediction": base,
                "openset_decode_lang": base_lang,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.detail, index=False)
    build_submission(
        df[["ID", "prediction"]], sample_path=PHASE2_SAMPLE, out_path=args.out
    )
    report = check_submission(args.out, PHASE2_SAMPLE)
    report.update(
        {
            "mode": args.mode,
            "p1_thresh": args.p1_thresh,
            "n_redecode": n_redecode,
            "n_changed": n_changed,
            "action_counts": Counter(df.action).most_common(),
            "source_counts": Counter(df.source).most_common(),
            "decode_top": Counter(df.decode_lang.astype(str)).most_common(10),
        }
    )
    (OUTPUT_DIR / "phase2_lid_hard_check.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    logger.info("CHECK %s", report)
    print("UPLOAD", args.out)
    print("CHANGED", n_changed, "REDECODE", n_redecode)
    print("ACTIONS", report["action_counts"])


if __name__ == "__main__":
    main()
