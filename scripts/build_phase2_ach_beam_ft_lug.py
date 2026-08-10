#!/usr/bin/env python3
"""Phase-2 candidate: openset routes + ach KenLM beam (guarded) + FT-lug hard replace.

Proxy evidence (outputs/proxy_ach_beam_guard.json):
  waxal-ach greedy 0.7060
  waxal-ach beam+length-guard 0.7335 (+0.0275)  ← BEST for ach
  ach-lmhead greedy 0.7177
  ach-lmhead beam 0.7195

Recipe (no cross-family conf-mix; no global KenLM):
  - Start from phase2_openset_detail routing
  - if decode_lang == ach → re-decode waxal-300m-ach with KenLM beam α=0.3 β=0.5
    beam=50 + length-guard word-count ratio in [0.5, 2.0] vs greedy
  - if decode_lang == lug → hard replace mms-lug-ft-v2
  - else keep openset hyp

Open-source only; no Phase-1 test gold.
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
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, OUTPUT_DIR, PROJECT_ROOT, TARGET_SR
from src.mms_infer import pick_device, transcribe_waveform
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import fix_mms_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ach_beam_ft")

AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
OPENSET = OUTPUT_DIR / "phase2_openset_detail.csv"
WAXAL_ACH = "waxal-benchmarking/mms-300m-waxal-ach"


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


def build_ach_decoder(processor, alpha: float, beta: float):
    vocab = processor.tokenizer.get_vocab()
    id2 = {i: t for t, i in vocab.items()}
    labels = [id2[i] for i in range(len(id2))]
    lm = ROOT / "data" / "lms" / "ach_2gram.arpa"
    uni = ROOT / "data" / "lms" / "ach_unigrams.txt"
    unigrams = [u for u in uni.read_text().splitlines() if u.strip()]
    return build_ctcdecoder(
        labels,
        kenlm_model_path=str(lm),
        unigrams=unigrams,
        alpha=alpha,
        beta=beta,
    )


@torch.inference_mode()
def beam_decode(model, processor, decoder, array, sr, device, beam_width: int = 50) -> str:
    array = np.asarray(array, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.max(np.abs(array)) + 1e-9)
    array = array / peak
    inputs = processor(array, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
    text = decoder.decode(logits, beam_width=beam_width).replace("|", " ")
    return normalize_text(text) or "."


def length_guard(greedy: str, beamed: str) -> tuple[str, bool]:
    gw = max(1, len(greedy.split()))
    bw = max(1, len(beamed.split()))
    r = bw / gw
    if 0.5 <= r <= 2.0:
        return beamed, True
    return greedy, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--beam", type=int, default=50)
    ap.add_argument("--no-ft-lug", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "submission_phase2_ach_beam_ft_lug.csv",
    )
    ap.add_argument("--detail", type=Path, default=OUTPUT_DIR / "phase2_ach_beam_ft_lug_detail.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    op = pd.read_csv(OPENSET)
    op["ID"] = op["ID"].astype(str)

    logger.info("load waxal ach %s", WAXAL_ACH)
    ach_p = AutoProcessor.from_pretrained(WAXAL_ACH)
    ach_m = Wav2Vec2ForCTC.from_pretrained(WAXAL_ACH).to(device).eval()
    decoder = build_ach_decoder(ach_p, args.alpha, args.beta)

    ft = None
    if not args.no_ft_lug:
        ckpt = CHECKPOINT_DIR / "mms-lug-ft-v2"
        if ckpt.exists():
            logger.info("load %s", ckpt)
            p = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
            m = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
            fix_mms_tokenizer(p, "lug")
            m.to(device).eval()
            ft = (m, p)

    rows = []
    counts = Counter()
    guard_stats = Counter()
    for _, r in tqdm(op.iterrows(), total=len(op), desc="ach-beam-ft"):
        uid = str(r.ID)
        base = normalize_text(str(r.prediction)) or "."
        dlang = str(r.decode_lang)
        lid = str(r.lid_lang)
        p1 = float(r.p1) if pd.notna(r.p1) else 0.0
        hyp = base
        src = "openset_keep"

        if dlang == "ach":
            arr, sr = load_wav(AUDIO / f"{uid}.wav")
            greedy = normalize_text(transcribe_waveform(ach_m, ach_p, arr, sr, device=device)) or "."
            beamed = beam_decode(ach_m, ach_p, decoder, arr, sr, device, beam_width=args.beam)
            hyp, ok = length_guard(greedy, beamed)
            src = "ach_beam_guard" if ok and hyp != greedy else ("ach_greedy_guard_reject" if not ok else "ach_beam_eq_greedy")
            if ok:
                guard_stats["accepted"] += 1
            else:
                guard_stats["rejected"] += 1
            if hyp != greedy:
                guard_stats["changed_vs_greedy"] += 1
        elif dlang == "lug" and ft is not None:
            arr, sr = load_wav(AUDIO / f"{uid}.wav")
            hyp = normalize_text(transcribe_waveform(ft[0], ft[1], arr, sr, device=device)) or "."
            src = "ft_lug"

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
            "n_changed": int((o.reindex(n.index) != n).sum()),
            "source_counts": counts.most_common(),
            "guard_stats": dict(guard_stats),
            "alpha": args.alpha,
            "beta": args.beta,
            "beam": args.beam,
            "method": "openset_route + ach KenLM beam length-guard + ft_lug hard",
            "proxy_ref": "outputs/proxy_ach_beam_guard.json",
        }
    )
    (OUTPUT_DIR / "phase2_ach_beam_ft_lug_check.json").write_text(json.dumps(rep, indent=2, default=str))
    logger.info("%s", rep)
    print("UPLOAD", args.out)
    print("CHANGED", rep["n_changed"], dict(counts), dict(guard_stats))


if __name__ == "__main__":
    main()
