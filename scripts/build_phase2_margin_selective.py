#!/usr/bin/env python3
"""Phase-2: margin-primary multi-hyp + selective upgrades.

Re-decodes multi-hyp candidates (same sets as openset), applies margin-primary:
  if best_conf - second_conf >= thr → max-conf lang else primary (cands[0]).
Then upgrades: lug→mms-lug-ft-v2, ach→KenLM beam+guard, nyn→beam+guard.

Never global KenLM, never cross-family conf-mix, never Phase-1 test gold.
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
logger = logging.getLogger("margin_sel")

AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
LID_CSV = OUTPUT_DIR / "phase2_lid126_full.csv"
OPENSET = OUTPUT_DIR / "phase2_openset_detail.csv"

WAXAL = {
    "ach": "waxal-benchmarking/mms-300m-waxal-ach",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
    "nyn": "waxal-benchmarking/mms-300m-waxal-nyn",
    "sog": "waxal-benchmarking/mms-300m-waxal-sog",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "mas": "waxal-benchmarking/mms-300m-waxal-mas",
}


def cand_set(lid_lang: str) -> list[str]:
    """Match scripts/run_phase2_openset.py multi-hyp candidate policy."""
    if lid_lang == "luo":
        return ["ach", "lug", "sog"]
    if lid_lang == "lug":
        return ["lug", "nyn", "sog"]
    if lid_lang in WAXAL:
        return [lid_lang]
    # rare LID: prefer openset fallbacks if present, else lug
    return ["lug", "ach"]


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    return arr / (float(np.max(np.abs(arr)) + 1e-9)), int(sr)


def mk_dec(proc, lang: str, alpha: float, beta: float):
    vocab = proc.tokenizer.get_vocab()
    id2 = {i: t for t, i in vocab.items()}
    labels = [id2[i] for i in range(len(id2))]
    lm = ROOT / "data" / "lms" / f"{lang}_2gram.arpa"
    uni = ROOT / "data" / "lms" / f"{lang}_unigrams.txt"
    unigrams = [u for u in uni.read_text().splitlines() if u.strip()] if uni.exists() else None
    return build_ctcdecoder(
        labels,
        kenlm_model_path=str(lm) if lm.exists() else None,
        unigrams=unigrams,
        alpha=alpha,
        beta=beta,
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
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--alpha-ach", type=float, default=0.2)
    ap.add_argument("--alpha-nyn", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "submission_phase2_margin_selective.csv",
    )
    ap.add_argument(
        "--detail",
        type=Path,
        default=OUTPUT_DIR / "phase2_margin_selective_detail.csv",
    )
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument(
        "--ft-ckpt",
        type=Path,
        default=None,
        help="Luganda FT checkpoint dir (default: checkpoints/mms-lug-ft-v2; prefer mms-lug-ft-v3)",
    )
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    logger.info("device %s margin=%s beam=%s", device, args.margin, args.beam)

    lid = pd.read_csv(LID_CSV)
    lid["ID"] = lid["ID"].astype(str)
    if "lang1" not in lid.columns:
        # alternate column names
        for c in ("lid_lang", "lang", "language"):
            if c in lid.columns:
                lid = lid.rename(columns={c: "lang1"})
                break
    if args.max_files:
        lid = lid.head(args.max_files)

    openset = {}
    if OPENSET.exists():
        op = pd.read_csv(OPENSET)
        for _, r in op.iterrows():
            openset[str(r.ID)] = normalize_text(str(r.prediction)) or "."

    models: dict[str, tuple] = {}
    decoders: dict = {}

    def get_model(lang: str):
        if lang not in models:
            if lang not in WAXAL:
                lang = "lug"
            mid = WAXAL[lang]
            logger.info("load %s", mid)
            p = AutoProcessor.from_pretrained(mid)
            m = Wav2Vec2ForCTC.from_pretrained(mid).to(device).eval()
            models[lang] = (m, p)
        return models[lang]

    # preload primary langs
    for L in ("ach", "lug", "nyn", "sog"):
        get_model(L)
    decoders["ach"] = mk_dec(models["ach"][1], "ach", args.alpha_ach, args.beta)
    decoders["nyn"] = mk_dec(models["nyn"][1], "nyn", args.alpha_nyn, args.beta)

    ckpt = Path(args.ft_ckpt) if args.ft_ckpt else (CHECKPOINT_DIR / "mms-lug-ft-v2")
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt if not ckpt.exists() else ckpt
    if not ckpt.exists() and (CHECKPOINT_DIR / "mms-lug-ft-v3").exists():
        ckpt = CHECKPOINT_DIR / "mms-lug-ft-v3"
    logger.info("load FT %s", ckpt)
    p_ft = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
    m_ft = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
    fix_mms_tokenizer(p_ft, "lug")
    m_ft.to(device).eval()

    rows = []
    counts = Counter()
    for _, r in tqdm(lid.iterrows(), total=len(lid), desc="margin-sel"):
        uid = str(r.ID)
        lid_lang = str(r.lang1)
        p1 = float(r.p1) if "p1" in r and pd.notna(r.p1) else (
            float(r["prob1"]) if "prob1" in r and pd.notna(r["prob1"]) else 0.0
        )
        path = AUDIO / f"{uid}.wav"
        arr, sr = load_wav(path)
        cands = cand_set(lid_lang)

        scored = []
        for L in cands:
            m, p = get_model(L)
            hyp, conf = transcribe_waveform(m, p, arr, sr, device=device, return_confidence=True)
            hyp = normalize_text(hyp) or "."
            scored.append((L, hyp, float(conf)))
        scored.sort(key=lambda x: x[2], reverse=True)
        best_L, best_h, best_c = scored[0]
        second_c = scored[1][2] if len(scored) > 1 else -1e9
        margin = best_c - second_c
        if margin >= args.margin:
            dlang, base_h, note = best_L, best_h, "margin_ok"
        else:
            # primary fallback
            pL = cands[0]
            base_h = next((h for L, h, c in scored if L == pL), best_h)
            dlang, note = pL, "primary_fb"

        # selective upgrades
        if dlang == "lug":
            hyp = normalize_text(transcribe_waveform(m_ft, p_ft, arr, sr, device=device)) or "."
            src = "ft_lug"
        elif dlang == "ach" and "ach" in decoders:
            m, p = models["ach"]
            greedy = base_h if note else (
                normalize_text(transcribe_waveform(m, p, arr, sr, device=device)) or "."
            )
            if note == "margin_ok" and dlang == "ach":
                greedy = best_h
            elif dlang == "ach":
                greedy = base_h
            beamed = beam_decode(m, p, decoders["ach"], arr, sr, device, args.beam)
            hyp, ok = length_guard(greedy, beamed)
            src = "ach_beam" if ok else "ach_beam_guard_reject"
        elif dlang == "nyn" and "nyn" in decoders:
            m, p = models["nyn"]
            greedy = base_h
            beamed = beam_decode(m, p, decoders["nyn"], arr, sr, device, args.beam)
            hyp, ok = length_guard(greedy, beamed)
            src = "nyn_beam" if ok else "nyn_beam_guard_reject"
        else:
            hyp = base_h
            src = f"waxal_{dlang}"

        counts[src] += 1
        counts[f"route_{note}"] += 1
        counts[f"dlang_{dlang}"] += 1
        rows.append(
            {
                "ID": uid,
                "prediction": hyp,
                "source": src,
                "lid_lang": lid_lang,
                "p1": p1,
                "decode_lang": dlang,
                "margin": margin,
                "route_note": note,
                "candidates": "|".join(cands),
                "openset_prediction": openset.get(uid, ""),
                "changed": int(hyp != openset.get(uid, hyp)),
            }
        )

    df = pd.DataFrame(rows)
    args.detail.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.detail, index=False)
    build_submission(df[["ID", "prediction"]], sample_path=SAMPLE, out_path=args.out)
    rep = check_submission(args.out, SAMPLE)
    o = pd.read_csv(PROJECT_ROOT / "submission_phase2_openset.csv").set_index("ID")["Target"].astype(str)
    n = df.set_index("ID")["prediction"].astype(str)
    rep.update(
        {
            "n_changed": int((o.reindex(n.index) != n).sum()),
            "source_counts": counts.most_common(),
            "margin": args.margin,
            "alpha_ach": args.alpha_ach,
            "alpha_nyn": args.alpha_nyn,
            "beta": args.beta,
            "beam": args.beam,
            "method": "margin_primary_multihyp + selective ach/nyn beam + ft_lug",
        }
    )
    (OUTPUT_DIR / "phase2_margin_selective_check.json").write_text(
        json.dumps(rep, indent=2, default=str)
    )
    logger.info("%s", rep)
    print("UPLOAD", args.out)
    print("CHANGED", rep.get("n_changed"), dict(counts))


if __name__ == "__main__":
    main()
