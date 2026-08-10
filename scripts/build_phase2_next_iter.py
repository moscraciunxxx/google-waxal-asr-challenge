#!/usr/bin/env python3
"""Compose the next-iteration Phase-2 candidate from proxy-gated route upgrades.

Floor spine: submission_phase2_selective_v3_dual15.csv (public 0.560605696).
Route upgrades (each optional, only applied when its offline gate passed):

  --nyn-ckpt  : redecode the 256 decode_lang==nyn rows with FT MMS-1B nyn
                (greedy, length-guard vs floor) — same class as shipped ft_lug.
  --lug-ckpt  : redecode the 738 decode_lang==lug rows with a stronger lug FT.
  --luo-ckpt  : refresh/expand the dual-agree Luo island: for lid=luo p1>=0.99
                rows, decode ckpt + CLEAR; accept iff char CER(luo_hyp, clear)
                <= 0.15 (the exact shipped dual15 gate); replacement text =
                the MMS-1B(-FT) luo hypothesis; length-guard vs floor.

Never touches other rows. Fail-closed: no ckpt flags -> byte-identical floor.
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

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2BertForCTC, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT, TARGET_SR
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import pick_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("next_iter")

AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
FLOOR = PROJECT_ROOT / "submission_phase2_selective_v3_dual15.csv"
OPENSET_DETAIL = OUTPUT_DIR / "phase2_openset_detail.csv"
CLEAR_ID = "CLEAR-Global/w2v-bert-2.0-luo_19_77h"


def load_wav(path: Path) -> np.ndarray:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak


@torch.inference_mode()
def greedy(model, processor, arr, device) -> str:
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return normalize_text(processor.decode(ids)) or "."


@torch.inference_mode()
def greedy_clear(model, processor, arr, device) -> str:
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_features", "attention_mask")}
    logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)[0]
    txt = processor.batch_decode(ids.unsqueeze(0))[0]
    return normalize_text(txt.replace("|", " ")) or "."


def length_guard(base: str, cand: str, lo: float = 0.4, hi: float = 2.5) -> bool:
    bw = max(1, len(base.split()))
    cw = max(1, len(cand.split()))
    return lo <= cw / bw <= hi and bool(cand.strip()) and cand.strip() != "."


def load_ctc(ckpt: str | Path, device, lang: str | None = None):
    ckpt = str(ckpt)
    proc = AutoProcessor.from_pretrained(ckpt, local_files_only=os.path.isdir(ckpt))
    model = Wav2Vec2ForCTC.from_pretrained(ckpt, local_files_only=os.path.isdir(ckpt))
    if lang and not os.path.isdir(ckpt):
        proc.tokenizer.set_target_lang(lang)
        model.load_adapter(lang)
    return model.to(device).eval(), proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyn-ckpt", type=Path, default=None)
    ap.add_argument("--nyn-beam", action="store_true", help="KenLM beam decode for nyn route (matches eval config)")
    ap.add_argument("--lug-ckpt", type=Path, default=None)
    ap.add_argument("--luo-ckpt", type=str, default=None, help="FT dir or 'zeroshot'")
    ap.add_argument("--luo-thr", type=float, default=0.15)
    ap.add_argument("--luo-min-p1", type=float, default=0.99)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="nextiter")
    args = ap.parse_args()

    device = pick_device(args.device)
    floor = pd.read_csv(FLOOR)
    floor["ID"] = floor["ID"].astype(str)
    floor = floor.set_index("ID")
    detail = pd.read_csv(OPENSET_DETAIL)
    detail["ID"] = detail["ID"].astype(str)
    detail = detail.set_index("ID")

    pred = {uid: normalize_text(str(t)) or "." for uid, t in floor["Target"].items()}
    source = {uid: "floor_keep" for uid in pred}
    counts = Counter()
    audit_rows = []

    # ---- nyn route upgrade
    if args.nyn_ckpt:
        ids = [u for u in pred if detail.loc[u, "decode_lang"] == "nyn"]
        logger.info("nyn redecode: %d rows with %s (beam=%s)", len(ids), args.nyn_ckpt, args.nyn_beam)
        model, proc = load_ctc(args.nyn_ckpt, device)
        decoder = None
        if args.nyn_beam:
            from pyctcdecode import build_ctcdecoder

            vocab = proc.tokenizer.get_vocab()
            labels_v = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
            uni_path = ROOT / "data" / "lms" / "nyn_unigrams.txt"
            unigrams = [w.strip() for w in uni_path.read_text().splitlines() if w.strip()] if uni_path.exists() else None
            decoder = build_ctcdecoder(
                labels_v, kenlm_model_path=str(ROOT / "data" / "lms" / "nyn_2gram.arpa"),
                unigrams=unigrams, alpha=0.3, beta=0.5,
            )
        t0 = time.time()
        for k, uid in enumerate(ids):
            arr = load_wav(AUDIO / f"{uid}.wav")
            if decoder is not None:
                with torch.inference_mode():
                    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
                    lg = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
                g = normalize_text(proc.decode(torch.tensor(lg.argmax(-1)))) or "."
                b = normalize_text(decoder.decode(lg, beam_width=100).replace("|", " ")) or "."
                gw, bw = max(1, len(g.split())), max(1, len(b.split()))
                cand = b if (0.5 <= bw / gw <= 2.0 and b.strip()) else g
            else:
                cand = greedy(model, proc, arr, device)
            if length_guard(pred[uid], cand):
                if cand != pred[uid]:
                    counts["nyn_ft_replace"] += 1
                    audit_rows.append({"ID": uid, "route": "nyn", "old": pred[uid], "new": cand})
                    pred[uid] = cand
                    source[uid] = "nyn_ft"
            else:
                counts["nyn_ft_guard_reject"] += 1
            if (k + 1) % 50 == 0:
                logger.info("nyn %d/%d %.1fs", k + 1, len(ids), time.time() - t0)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    # ---- lug route upgrade
    if args.lug_ckpt:
        ids = [u for u in pred if detail.loc[u, "decode_lang"] == "lug"]
        logger.info("lug redecode: %d rows with %s", len(ids), args.lug_ckpt)
        model, proc = load_ctc(args.lug_ckpt, device)
        t0 = time.time()
        for k, uid in enumerate(ids):
            cand = greedy(model, proc, load_wav(AUDIO / f"{uid}.wav"), device)
            if length_guard(pred[uid], cand):
                if cand != pred[uid]:
                    counts["lug_ft_replace"] += 1
                    audit_rows.append({"ID": uid, "route": "lug", "old": pred[uid], "new": cand})
                    pred[uid] = cand
                    source[uid] = "lug_ft_v5"
            else:
                counts["lug_ft_guard_reject"] += 1
            if (k + 1) % 100 == 0:
                logger.info("lug %d/%d %.1fs", k + 1, len(ids), time.time() - t0)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    # ---- luo island refresh (dual gate, exact shipped thresholds)
    luo_stats = {}
    if args.luo_ckpt:
        pool = [
            u
            for u in pred
            if detail.loc[u, "lid_lang"] == "luo" and float(detail.loc[u, "p1"]) >= args.luo_min_p1
        ]
        logger.info("luo dual pool: %d rows (p1>=%s)", len(pool), args.luo_min_p1)
        if args.luo_ckpt == "zeroshot":
            lmodel, lproc = load_ctc("facebook/mms-1b-all", device, lang="luo")
        else:
            lmodel, lproc = load_ctc(args.luo_ckpt, device)
        cmodel = Wav2Vec2BertForCTC.from_pretrained(CLEAR_ID).to(device).eval()
        cproc = AutoProcessor.from_pretrained(CLEAR_ID)
        n_accept = 0
        t0 = time.time()
        for k, uid in enumerate(pool):
            arr = load_wav(AUDIO / f"{uid}.wav")
            hyp_l = greedy(lmodel, lproc, arr, device)
            hyp_c = greedy_clear(cmodel, cproc, arr, device)
            cer_lc = jiwer.cer(hyp_l or ".", hyp_c or ".")
            accepted = cer_lc <= args.luo_thr and length_guard(pred[uid], hyp_l)
            if accepted:
                n_accept += 1
                if hyp_l != pred[uid]:
                    counts["luo_dual_replace"] += 1
                    audit_rows.append(
                        {"ID": uid, "route": "luo_dual", "cer_lc": cer_lc, "old": pred[uid], "new": hyp_l}
                    )
                    pred[uid] = hyp_l
                    source[uid] = "luo_dual_ft"
            if (k + 1) % 50 == 0:
                logger.info("luo %d/%d accepts=%d %.1fs", k + 1, len(pool), n_accept, time.time() - t0)
        luo_stats = {"pool_n": len(pool), "n_accept": n_accept, "thr": args.luo_thr, "min_p1": args.luo_min_p1}
        logger.info("luo island: %s", luo_stats)
        del lmodel, cmodel
        if device.type == "mps":
            torch.mps.empty_cache()

    out_csv = PROJECT_ROOT / f"submission_phase2_{args.tag}.csv"
    detail_csv = OUTPUT_DIR / "next_iter" / f"phase2_{args.tag}_detail.csv"
    detail_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {"ID": list(pred.keys()), "prediction": [pred[u] for u in pred], "source": [source[u] for u in pred]}
    )
    pd.DataFrame(audit_rows).to_csv(OUTPUT_DIR / "next_iter" / f"phase2_{args.tag}_changed.csv", index=False)
    df.to_csv(detail_csv, index=False)
    build_submission(df[["ID", "prediction"]], sample_path=SAMPLE, out_path=out_csv)
    rep = check_submission(out_csv, SAMPLE)
    floor_t = floor["Target"].astype(str).map(lambda s: normalize_text(s) or ".")
    new_t = df.set_index("ID")["prediction"].astype(str)
    rep.update(
        {
            "n_changed_vs_floor": int((floor_t.reindex(new_t.index) != new_t).sum()),
            "source_counts": counts.most_common(),
            "luo_stats": luo_stats,
            "nyn_ckpt": str(args.nyn_ckpt) if args.nyn_ckpt else None,
            "lug_ckpt": str(args.lug_ckpt) if args.lug_ckpt else None,
            "luo_ckpt": str(args.luo_ckpt) if args.luo_ckpt else None,
            "floor": str(FLOOR),
        }
    )
    (OUTPUT_DIR / "next_iter" / f"phase2_{args.tag}_check.json").write_text(
        json.dumps(rep, indent=2, default=str)
    )
    logger.info("%s", rep)
    print("CANDIDATE", out_csv)
    print("CHANGED_VS_FLOOR", rep["n_changed_vs_floor"], dict(counts))


if __name__ == "__main__":
    main()
