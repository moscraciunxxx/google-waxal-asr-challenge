#!/usr/bin/env python3
"""Phase-2 hybrid: CLEAR Dholuo w2v-bert hard-replace on LID=luo mass.

Base: margin thr=0.03 + FT-v3 + selective beam CSV (proxy-gated spine).
Luo overlay: CLEAR-Global/w2v-bert-2.0-luo_* hard replace (NO CTC conf-mix).

Motivation: ~785/1500 Phase-2 clips are LID=luo; champion has no true-luo path
(ach|lug|sog multi-hyp). CLEAR reports ~0.29 WER / ~0.09 CER on Luo CV/FLEURS.

Open-source only; no Phase-1 test gold; no cross-family conf pick.
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
from transformers import AutoFeatureExtractor, AutoProcessor, Wav2Vec2BertForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT, TARGET_SR
from src.mms_infer import pick_device
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("clear_luo")

AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
LID_CSV = OUTPUT_DIR / "phase2_lid126_full.csv"
DEFAULT_BASE = PROJECT_ROOT / "submission_phase2_margin03_v3.csv"
DEFAULT_MODEL = "CLEAR-Global/w2v-bert-2.0-luo_19_77h"


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


def length_guard(base: str, cand: str, lo: float = 0.4, hi: float = 2.5) -> tuple[str, bool]:
    bw = max(1, len(base.split()))
    cw = max(1, len(cand.split()))
    r = cw / bw
    if lo <= r <= hi and cand.strip() and cand.strip() != ".":
        return cand, True
    return base, False


@torch.inference_mode()
def transcribe_clear(model, fe, tokenizer, arr: np.ndarray, sr: int, device) -> str:
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr / (float(np.max(np.abs(arr)) + 1e-9))
    inputs = fe(arr, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)[0]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    # w2v-bert tokenizers sometimes use | as space already handled by decode
    text = text.replace("|", " ")
    return normalize_text(text) or "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--min-p1", type=float, default=0.95, help="Only replace LID=luo if p1>=this")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "submission_phase2_margin03_clearluo_p095.csv",
    )
    ap.add_argument(
        "--detail",
        type=Path,
        default=OUTPUT_DIR / "phase2_margin03_clearluo_p095_detail.csv",
    )
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    base = pd.read_csv(args.base)
    base["ID"] = base["ID"].astype(str)
    base = base.set_index("ID")
    lid = pd.read_csv(LID_CSV)
    lid["ID"] = lid["ID"].astype(str)

    logger.info("load CLEAR Luo model %s on %s", args.model, device)
    # Prefer AutoProcessor; fall back to feature extractor + tokenizer pieces
    try:
        proc = AutoProcessor.from_pretrained(args.model)
        fe = proc
        tokenizer = proc
    except Exception:
        fe = AutoFeatureExtractor.from_pretrained(args.model)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        proc = None

    model = Wav2Vec2BertForCTC.from_pretrained(args.model).to(device).eval()

    rows = []
    counts = Counter()
    for _, r in tqdm(lid.iterrows(), total=len(lid), desc="clear-luo"):
        uid = str(r.ID)
        base_hyp = normalize_text(str(base.loc[uid, "Target"])) or "."
        lang1 = str(r.lang1)
        p1 = float(r.p1) if pd.notna(r.p1) else 0.0
        hyp = base_hyp
        src = "base_keep"

        if lang1 == "luo" and p1 >= args.min_p1:
            arr, sr = load_wav(AUDIO / f"{uid}.wav")
            if proc is not None:
                cand = transcribe_clear(model, fe, tokenizer, arr, sr, device)
            else:
                cand = transcribe_clear(model, fe, tokenizer, arr, sr, device)
            hyp, ok = length_guard(base_hyp, cand)
            src = "clear_luo" if ok else "clear_luo_guard_reject"
            if not ok:
                hyp = base_hyp

        counts[src] += 1
        rows.append(
            {
                "ID": uid,
                "prediction": hyp,
                "source": src,
                "lid_lang": lang1,
                "p1": p1,
                "base_prediction": base_hyp,
                "changed": int(hyp != base_hyp),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.detail, index=False)
    build_submission(df[["ID", "prediction"]], sample_path=SAMPLE, out_path=args.out)
    rep = check_submission(args.out, SAMPLE)
    o = pd.read_csv(PROJECT_ROOT / "submission_phase2_openset.csv").set_index("ID")["Target"].astype(str)
    n = df.set_index("ID")["prediction"].astype(str)
    b = base["Target"].astype(str)
    rep.update(
        {
            "n_changed_vs_openset": int((o.reindex(n.index) != n).sum()),
            "n_changed_vs_base": int((b.reindex(n.index) != n).sum()),
            "source_counts": counts.most_common(),
            "min_p1": args.min_p1,
            "clear_model": args.model,
            "base_csv": str(args.base),
            "method": "margin03_v3_base + CLEAR w2v-bert luo hard replace (no conf-mix)",
        }
    )
    check_path = OUTPUT_DIR / f"{args.out.stem}_check.json"
    check_path.write_text(json.dumps(rep, indent=2, default=str))
    logger.info("%s", rep)
    print("UPLOAD", args.out)
    print("CHANGED_VS_BASE", rep["n_changed_vs_base"], dict(counts))


if __name__ == "__main__":
    main()
