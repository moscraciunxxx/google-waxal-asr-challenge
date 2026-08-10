#!/usr/bin/env python3
"""Phase-2 open-set ASR: full MMS-LID-126 + waxal-benchmarking MMS-300M models.

Hypothesis (supported by portal routing-null ~0.46 and hyp forensics): Phase-2
may include WAXAL languages beyond lin/sna/lug. Closed-set 3-way routing cannot
fix that. Decode with language-matched open WAXAL FT models when available.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
from src.mms_infer import load_mms, pick_device, set_lang, transcribe_waveform
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase2_openset")

PHASE2_AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
PHASE2_SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
LID126_CSV = OUTPUT_DIR / "phase2_lid126_full.csv"

# Map ISO codes from mms-lid to open WAXAL MMS-300M checkpoints (HF)
WAXAL300 = {
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
    "lin": "waxal-benchmarking/mms-300m-waxal-lin",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "ach": "waxal-benchmarking/mms-300m-waxal-ach",
    "sog": "waxal-benchmarking/mms-300m-waxal-sog",
    "nyn": "waxal-benchmarking/mms-300m-waxal-nyn",
    "mas": "waxal-benchmarking/mms-300m-waxal-mas",
    "aka": "waxal-benchmarking/mms-300m-waxal-aka",
    "ewe": "waxal-benchmarking/mms-300m-waxal-ewe",
    "ful": "waxal-benchmarking/mms-300m-waxal-ful",
    "orm": "waxal-benchmarking/mms-300m-waxal-orm",
    "amh": "waxal-benchmarking/mms-300m-waxal-amh",
    "tir": "waxal-benchmarking/mms-300m-waxal-tir",
    "wal": "waxal-benchmarking/mms-300m-waxal-wal",
    "dag": "waxal-benchmarking/mms-300m-waxal-dag",
    "dga": "waxal-benchmarking/mms-300m-waxal-dga",
    "kpo": "waxal-benchmarking/mms-300m-waxal-kpo",
    "sid": "waxal-benchmarking/mms-300m-waxal-sid",
    "mlg": "waxal-benchmarking/mms-300m-waxal-mlg",
}

# Fallback: if LID predicts luo (no waxal-300m ASR), try Acholi (related Nilotic) then lug
FALLBACK = {
    "luo": ["ach", "lug"],
    "swh": ["lug", "lin"],
    "swa": ["lug", "lin"],
    "kin": ["nyn", "lug"],
    "nya": ["sna", "lug"],
    "umb": ["sog", "lug"],
    "nso": ["sna", "lug"],
    "wol": ["ful", "lug"],
    "eng": ["lug", "lin", "sna"],
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


def resolve_model_lang(lid_lang: str) -> str:
    if lid_lang in WAXAL300:
        return lid_lang
    for cand in FALLBACK.get(lid_lang, []):
        if cand in WAXAL300:
            return cand
    # default challenge languages
    if lid_lang in ("lin", "sna", "lug"):
        return lid_lang
    return "lug"


class ModelCache:
    def __init__(self, device: torch.device):
        self.device = device
        self.cache: dict[str, tuple] = {}
        self.zs = None

    def get(self, lang: str):
        if lang in self.cache:
            return self.cache[lang]
        if lang in WAXAL300:
            mid = WAXAL300[lang]
            logger.info("Loading %s", mid)
            try:
                processor = AutoProcessor.from_pretrained(mid)
                model = Wav2Vec2ForCTC.from_pretrained(mid)
                model.to(self.device).eval()
                self.cache[lang] = ("waxal300", model, processor)
                return self.cache[lang]
            except Exception as e:
                logger.warning("Failed %s: %s — fallback ZS adapter", mid, e)
        # fallback MMS-1b-all adapter for lin/sna/lug
        if self.zs is None:
            self.zs = load_mms(device=self.device)
        model, processor, device = self.zs
        if lang in ("lin", "sna", "lug"):
            set_lang(model, processor, lang)
            return ("mms1b", model, processor)
        # last resort lug
        set_lang(model, processor, "lug")
        return ("mms1b_lug", model, processor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lid-csv", type=Path, default=LID126_CSV)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "submission_phase2_openset.csv")
    p.add_argument(
        "--restrict-challenge3",
        action="store_true",
        help="Map everything to lin/sna/lug only (baseline control)",
    )
    args = p.parse_args()

    if not args.lid_csv.exists():
        raise SystemExit(f"Need {args.lid_csv} — run full LID-126 first")

    lid = pd.read_csv(args.lid_csv)
    if args.max_files:
        lid = lid.head(args.max_files)
    device = torch.device(args.device) if args.device else pick_device()
    cache = ModelCache(device)

    detail_path = OUTPUT_DIR / "phase2_openset_detail.csv"
    done = {}
    if args.resume and detail_path.exists():
        prev = pd.read_csv(detail_path)
        for _, r in prev.iterrows():
            done[str(r.ID)] = r.to_dict()
        logger.info("resume %d", len(done))

    rows = list(done.values())
    todo = lid[~lid.ID.astype(str).isin(done)]
    t0 = time.time()
    for n, (_, r) in enumerate(tqdm(todo.iterrows(), total=len(todo), desc="openset"), start=1):
        uid = str(r.ID)
        lid_lang = str(r.lang1)
        path = PHASE2_AUDIO / f"{uid}.wav"
        arr, sr = load_wav(path)

        # Multi-hyp when LID is Luo (no dedicated waxal-300m ASR): try ach + lug + sog
        # and pick by CTC mean logprob. For lug, try lug + nyn + sog (near-Bantu).
        if args.restrict_challenge3:
            candidates = ["lin", "sna", "lug"]
        elif lid_lang == "luo":
            candidates = ["ach", "lug", "sog"]
        elif lid_lang == "lug":
            candidates = ["lug", "nyn", "sog"]
        elif lid_lang in WAXAL300:
            candidates = [lid_lang]
        else:
            candidates = [resolve_model_lang(lid_lang), "lug", "ach"]

        best_hyp, best_lang, best_conf, best_src = ".", candidates[0], -1e9, "?"
        for use_lang in candidates:
            src, model, processor = cache.get(use_lang)
            if src.startswith("mms1b") and use_lang in ("lin", "sna", "lug"):
                set_lang(model, processor, use_lang)
            hyp, conf = transcribe_waveform(
                model, processor, arr, sr, device=device, return_confidence=True
            )
            if conf > best_conf:
                best_hyp, best_lang, best_conf, best_src = hyp, use_lang, conf, src

        rows.append(
            {
                "ID": uid,
                "prediction": best_hyp,
                "lid_lang": lid_lang,
                "p1": float(r.p1),
                "decode_lang": best_lang,
                "confidence": best_conf,
                "source": best_src,
                "candidates": "|".join(candidates),
            }
        )
        if n % 25 == 0 or n == len(todo):
            pd.DataFrame(rows).to_csv(detail_path, index=False)
            rate = n / max(time.time() - t0, 1e-6)
            logger.info(
                "n=%d rate=%.2f eta=%.0f langs=%s",
                n,
                rate,
                (len(todo) - n) / max(rate, 1e-9),
                Counter(x["decode_lang"] for x in rows).most_common(8),
            )

    df = pd.DataFrame(rows)
    df.to_csv(detail_path, index=False)
    build_submission(
        df[["ID", "prediction"]], sample_path=PHASE2_SAMPLE, out_path=args.out
    )
    report = check_submission(args.out, PHASE2_SAMPLE)
    report["lid_top"] = Counter(df.lid_lang).most_common(15)
    report["decode_top"] = Counter(df.decode_lang).most_common(15)
    report["source_counts"] = Counter(df.source).most_common()
    (OUTPUT_DIR / "phase2_openset_check.json").write_text(json.dumps(report, indent=2, default=str))
    logger.info("CHECK %s", report)
    print("UPLOAD", args.out)


if __name__ == "__main__":
    main()
