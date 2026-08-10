#!/usr/bin/env python3
"""Re-beam margin_selective ach/nyn (and optional waxal-lug) with Phase-2 domain LMs.

Starts from outputs/phase2_margin_selective_detail.csv routing (decode_lang fixed).
Does NOT re-run multi-hyp. Upgrades text only via KenLM beam + length-guard.

LMs: data/lms_phase2_domain/{lang}_merged_2gram.arpa (HF train + Phase-2 self-text).
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
logger = logging.getLogger("domain_beam")

AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
DETAIL_IN = OUTPUT_DIR / "phase2_margin_selective_detail.csv"
LM_DIR = PROJECT_ROOT / "data" / "lms_phase2_domain"
WAXAL = {
    "ach": "waxal-benchmarking/mms-300m-waxal-ach",
    "nyn": "waxal-benchmarking/mms-300m-waxal-nyn",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
}


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    return arr / (float(np.max(np.abs(arr)) + 1e-9)), int(sr)


def mk_dec(proc, lang: str, alpha: float, beta: float, merged: bool):
    vocab = proc.tokenizer.get_vocab()
    id2 = {i: t for t, i in vocab.items()}
    labels = [id2[i] for i in range(len(id2))]
    name = f"{lang}_merged_2gram.arpa" if merged else f"{lang}_2gram.arpa"
    lm = LM_DIR / name
    # unigrams from domain
    uni_path = LM_DIR / f"{lang}_unigrams.txt"
    if not uni_path.exists():
        # derive from train
        train = LM_DIR / f"{lang}_train.txt"
        words = []
        if train.exists():
            for line in train.read_text().splitlines():
                words.extend(line.split())
            uni_path.write_text("\n".join(dict.fromkeys(words)) + "\n")
    unigrams = [u for u in uni_path.read_text().splitlines() if u.strip()] if uni_path.exists() else None
    if not lm.exists():
        raise FileNotFoundError(lm)
    return build_ctcdecoder(
        labels, kenlm_model_path=str(lm), unigrams=unigrams, alpha=alpha, beta=beta
    )


@torch.inference_mode()
def beam_decode(model, proc, dec, arr, sr, device, beam_width: int) -> str:
    arr = np.asarray(arr, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    arr = arr / (float(np.max(np.abs(arr)) + 1e-9))
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
    text = dec.decode(logits, beam_width=beam_width).replace("|", " ")
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
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--merged-lm", action="store_true", default=True)
    ap.add_argument("--no-merged-lm", action="store_true")
    ap.add_argument("--beam-lug-waxal", action="store_true", help="also re-beam lug with waxal+domain LM")
    ap.add_argument("--keep-ft-lug", action="store_true", default=True)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "submission_phase2_domain_beam.csv",
    )
    ap.add_argument(
        "--detail",
        type=Path,
        default=OUTPUT_DIR / "phase2_domain_beam_detail.csv",
    )
    args = ap.parse_args()
    if args.no_merged_lm:
        args.merged_lm = False

    device = torch.device(args.device) if args.device else pick_device()
    det = pd.read_csv(DETAIL_IN)
    det["ID"] = det["ID"].astype(str)

    models = {}
    decoders = {}
    for lang in ("ach", "nyn"):
        mid = WAXAL[lang]
        logger.info("load %s", mid)
        p = AutoProcessor.from_pretrained(mid)
        m = Wav2Vec2ForCTC.from_pretrained(mid).to(device).eval()
        models[lang] = (m, p)
        decoders[lang] = mk_dec(p, lang, args.alpha, args.beta, args.merged_lm)

    if args.beam_lug_waxal:
        mid = WAXAL["lug"]
        logger.info("load %s", mid)
        p = AutoProcessor.from_pretrained(mid)
        m = Wav2Vec2ForCTC.from_pretrained(mid).to(device).eval()
        models["lug"] = (m, p)
        decoders["lug"] = mk_dec(p, "lug", args.alpha, args.beta, args.merged_lm)

    ft = None
    if args.keep_ft_lug and not args.beam_lug_waxal:
        ckpt = CHECKPOINT_DIR / "mms-lug-ft-v2"
        logger.info("keep FT lug %s", ckpt)
        p = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
        m = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
        fix_mms_tokenizer(p, "lug")
        m.to(device).eval()
        ft = (m, p)

    rows = []
    counts = Counter()
    for _, r in tqdm(det.iterrows(), total=len(det), desc="domain-beam"):
        uid = str(r.ID)
        base = normalize_text(str(r.prediction)) or "."
        dlang = str(r.decode_lang)
        hyp = base
        src = "keep_" + str(r.source)

        if dlang in decoders:
            arr, sr = load_wav(AUDIO / f"{uid}.wav")
            m, p = models[dlang]
            greedy = normalize_text(transcribe_waveform(m, p, arr, sr, device=device)) or "."
            beamed = beam_decode(m, p, decoders[dlang], arr, sr, device, args.beam)
            hyp, ok = length_guard(greedy, beamed)
            src = f"domain_{dlang}_beam" if ok else f"domain_{dlang}_guard_reject"
        elif dlang == "lug" and ft is not None:
            # already FT text in base; keep
            hyp = base
            src = "keep_ft_lug"

        counts[src] += 1
        rows.append(
            {
                "ID": uid,
                "prediction": hyp,
                "source": src,
                "decode_lang": dlang,
                "lid_lang": r.lid_lang,
                "prev_prediction": base,
                "changed": int(hyp != base),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.detail, index=False)
    build_submission(df[["ID", "prediction"]], sample_path=SAMPLE, out_path=args.out)
    rep = check_submission(args.out, SAMPLE)
    o = pd.read_csv(PROJECT_ROOT / "submission_phase2_openset.csv").set_index("ID")["Target"].astype(str)
    mdf = df.set_index("ID")["prediction"].astype(str)
    margin = pd.read_csv(PROJECT_ROOT / "submission_phase2_margin_selective.csv").set_index("ID")[
        "Target"
    ].astype(str)
    rep.update(
        {
            "n_changed_vs_openset": int((o.reindex(mdf.index) != mdf).sum()),
            "n_changed_vs_margin": int((margin.reindex(mdf.index) != mdf).sum()),
            "source_counts": counts.most_common(),
            "alpha": args.alpha,
            "beta": args.beta,
            "beam": args.beam,
            "merged_lm": args.merged_lm,
            "method": "margin_route + phase2_domain_kenlm_beam",
        }
    )
    (OUTPUT_DIR / "phase2_domain_beam_check.json").write_text(json.dumps(rep, indent=2, default=str))
    logger.info("%s", rep)
    print("UPLOAD", args.out)
    print(json.dumps({k: rep[k] for k in ("ok", "n_changed_vs_openset", "n_changed_vs_margin", "source_counts")}, indent=2))


if __name__ == "__main__":
    main()
