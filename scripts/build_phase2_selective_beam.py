#!/usr/bin/env python3
"""Phase-2: selective KenLM beam (ach/lug/nyn) + FT-lug + optional force-luo→ach.

Builds on champion openset routing. Never applies global KenLM.
Never cross-family CTC conf-mix.

Default recipe:
  - decode_lang==ach OR (force_luo_ach and lid==luo and p1>=thr) → waxal-ach beam+guard
  - decode_lang==lug (and not forced to ach) → mms-lug-ft-v2 hard
  - decode_lang==nyn → waxal-nyn beam+guard (if --beam-nyn)
  - else openset keep
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
logger = logging.getLogger("sel_beam")

AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
OPENSET = OUTPUT_DIR / "phase2_openset_detail.csv"
WAXAL = {
    "ach": "waxal-benchmarking/mms-300m-waxal-ach",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
    "nyn": "waxal-benchmarking/mms-300m-waxal-nyn",
}


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    return arr / (float(np.max(np.abs(arr)) + 1e-9)), int(sr)


def build_decoder(processor, lang: str, alpha: float, beta: float):
    vocab = processor.tokenizer.get_vocab()
    id2 = {i: t for t, i in vocab.items()}
    labels = [id2[i] for i in range(len(id2))]
    lm = ROOT / "data" / "lms" / f"{lang}_2gram.arpa"
    uni = ROOT / "data" / "lms" / f"{lang}_unigrams.txt"
    unigrams = [u for u in uni.read_text().splitlines() if u.strip()]
    return build_ctcdecoder(
        labels, kenlm_model_path=str(lm), unigrams=unigrams, alpha=alpha, beta=beta
    )


@torch.inference_mode()
def beam_decode(model, processor, decoder, array, sr, device, beam_width: int = 50) -> str:
    array = np.asarray(array, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
    array = array / (float(np.max(np.abs(array)) + 1e-9))
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
    ap.add_argument("--beam-nyn", action="store_true", default=True)
    ap.add_argument("--no-beam-nyn", action="store_true")
    ap.add_argument("--force-luo-ach-p1", type=float, default=0.95, help="0 disables")
    ap.add_argument("--no-ft-lug", action="store_true")
    ap.add_argument(
        "--ft-ckpt",
        type=Path,
        default=CHECKPOINT_DIR / "mms-lug-ft-v3",
        help="Luganda FT checkpoint (default: mms-lug-ft-v3)",
    )
    ap.add_argument("--alpha-ach", type=float, default=None, help="Override alpha for ach only")
    ap.add_argument("--alpha-nyn", type=float, default=None, help="Override alpha for nyn only")
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "submission_phase2_selective_beam.csv",
    )
    ap.add_argument("--detail", type=Path, default=OUTPUT_DIR / "phase2_selective_beam_detail.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.no_beam_nyn:
        args.beam_nyn = False

    device = torch.device(args.device) if args.device else pick_device()
    op = pd.read_csv(OPENSET)
    op["ID"] = op["ID"].astype(str)

    models = {}
    decoders = {}
    for lang in ("ach", "nyn"):
        if lang == "nyn" and not args.beam_nyn:
            continue
        mid = WAXAL[lang]
        logger.info("load %s", mid)
        p = AutoProcessor.from_pretrained(mid)
        m = Wav2Vec2ForCTC.from_pretrained(mid).to(device).eval()
        models[lang] = (m, p)
        a = args.alpha
        if lang == "ach" and args.alpha_ach is not None:
            a = args.alpha_ach
        if lang == "nyn" and args.alpha_nyn is not None:
            a = args.alpha_nyn
        decoders[lang] = build_decoder(p, lang, a, args.beta)

    ft = None
    if not args.no_ft_lug:
        ckpt = Path(args.ft_ckpt)
        logger.info("load %s", ckpt)
        p = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
        m = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
        fix_mms_tokenizer(p, "lug")
        m.to(device).eval()
        ft = (m, p)

    rows = []
    counts = Counter()
    for _, r in tqdm(op.iterrows(), total=len(op), desc="sel-beam"):
        uid = str(r.ID)
        base = normalize_text(str(r.prediction)) or "."
        dlang = str(r.decode_lang)
        lid = str(r.lid_lang)
        p1 = float(r.p1) if pd.notna(r.p1) else 0.0
        hyp = base
        src = "openset_keep"

        force_ach = (
            args.force_luo_ach_p1 > 0
            and lid == "luo"
            and p1 >= args.force_luo_ach_p1
            and "ach" in models
        )
        use_ach = dlang == "ach" or force_ach

        if use_ach and "ach" in models:
            arr, sr = load_wav(AUDIO / f"{uid}.wav")
            m, p = models["ach"]
            greedy = normalize_text(transcribe_waveform(m, p, arr, sr, device=device)) or "."
            beamed = beam_decode(m, p, decoders["ach"], arr, sr, device, args.beam)
            hyp, ok = length_guard(greedy, beamed)
            src = "force_luo_ach_beam" if force_ach and dlang != "ach" else "ach_beam"
            if not ok:
                src = src + "_guard_reject"
        elif dlang == "lug" and ft is not None:
            arr, sr = load_wav(AUDIO / f"{uid}.wav")
            hyp = normalize_text(transcribe_waveform(ft[0], ft[1], arr, sr, device=device)) or "."
            src = "ft_lug"
        elif dlang == "nyn" and args.beam_nyn and "nyn" in models:
            arr, sr = load_wav(AUDIO / f"{uid}.wav")
            m, p = models["nyn"]
            greedy = normalize_text(transcribe_waveform(m, p, arr, sr, device=device)) or "."
            beamed = beam_decode(m, p, decoders["nyn"], arr, sr, device, args.beam)
            hyp, ok = length_guard(greedy, beamed)
            src = "nyn_beam" if ok else "nyn_beam_guard_reject"

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
            "force_luo_ach_p1": args.force_luo_ach_p1,
            "beam_nyn": args.beam_nyn,
            "alpha": args.alpha,
            "beta": args.beta,
            "beam": args.beam,
            "method": "selective_beam_ach_nyn + ft_lug + optional force_luo_ach",
        }
    )
    (OUTPUT_DIR / "phase2_selective_beam_check.json").write_text(json.dumps(rep, indent=2, default=str))
    logger.info("%s", rep)
    print("UPLOAD", args.out)
    print("CHANGED", rep["n_changed"], dict(counts))


if __name__ == "__main__":
    main()
