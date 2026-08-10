#!/usr/bin/env python3
"""Acoustic open-set router: Luo vs Acholi vs Bantu-ish using MMS-LID-126.

Open labeled data only for optional calibration; Phase-2 smoke uses waveform+LID.
Does NOT train on test gold. Does NOT mass multi-adapter rewrite lid=luo.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT, SEED, TARGET_SR
from src.mms_infer import pick_device
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("luo_router")

# Coarse buckets for open-set routing
LUO_CODES = frozenset({"luo"})
ACH_CODES = frozenset({"ach", "lwo", "alz"})
BANTU_CODES = frozenset({"lug", "lin", "sna", "nyn", "sog", "swa", "swh", "nya", "kin", "xog"})


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    if peak > 0:
        arr = arr / peak
    return arr, int(sr)


def bucket(lang: str) -> str:
    c = (lang or "").lower()
    if c in LUO_CODES:
        return "luo"
    if c in ACH_CODES:
        return "ach"
    if c in BANTU_CODES:
        return "bantu"
    return "other"


class AcousticRouter:
    def __init__(self, device: torch.device, model_id: str = "facebook/mms-lid-126"):
        self.device = device
        try:
            self.fe = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=True)
            self.model = AutoModelForAudioClassification.from_pretrained(
                model_id, local_files_only=True
            )
        except Exception:
            self.fe = AutoFeatureExtractor.from_pretrained(model_id)
            self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.to(device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.inference_mode()
    def route(self, array: np.ndarray, sr: int) -> dict:
        if sr != TARGET_SR:
            import librosa

            array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
        inputs = self.fe(array, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        # top-3
        topk = torch.topk(probs, k=min(5, probs.numel()))
        top = [
            (self.id2label[int(i)], float(probs[int(i)]))
            for i in topk.indices.tolist()
        ]
        # mass probability on buckets
        mass = {"luo": 0.0, "ach": 0.0, "bantu": 0.0, "other": 0.0}
        for idx in range(probs.numel()):
            lab = self.id2label[idx]
            mass[bucket(lab)] += float(probs[idx])
        route = max(mass, key=mass.get)
        return {
            "route": route,
            "top1_lang": top[0][0],
            "top1_p": top[0][1],
            "mass_luo": mass["luo"],
            "mass_ach": mass["ach"],
            "mass_bantu": mass["bantu"],
            "mass_other": mass["other"],
            "top": top,
            "use_luo_specialist": route == "luo" and mass["luo"] >= 0.5,
        }


def maybe_luo_hyp_mms1b(array: np.ndarray, sr: int, device: torch.device) -> str:
    """Optional Luo specialist hyp via mms-1b-all adapter luo (open weights)."""
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    from src.mms_infer import set_lang, transcribe_waveform

    mid = "facebook/mms-1b-all"
    try:
        proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
    except Exception:
        proc = AutoProcessor.from_pretrained(mid)
        model = Wav2Vec2ForCTC.from_pretrained(mid)
    model.to(device).eval()
    set_lang(model, proc, "luo")
    return transcribe_waveform(model, proc, array, sr, device=device)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", type=Path, default=PROJECT_ROOT / "data" / "phase2" / "audio")
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--max-files", type=int, default=4)
    p.add_argument("--lid-csv", type=Path, default=OUTPUT_DIR / "phase2_lid126_full.csv")
    p.add_argument("--prefer-lid-luo", action="store_true", help="Sample IDs with precomputed lid=luo")
    p.add_argument("--with-luo-asr", action="store_true")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "legit_system" / "luo_router_smoke.csv")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = pick_device()
    router = AcousticRouter(device)

    ids = args.ids
    if not ids and args.prefer_lid_luo and args.lid_csv.exists():
        df = pd.read_csv(args.lid_csv)
        luo = df[df["lang1"].astype(str) == "luo"].head(args.max_files)
        ids = luo["ID"].astype(str).tolist()
    if not ids:
        ids = [p.stem for p in sorted(args.audio_dir.glob("*.wav"))[: args.max_files]]

    rows = []
    for sid in ids:
        path = args.audio_dir / f"{sid}.wav"
        if not path.exists():
            continue
        arr, sr = load_wav(path)
        r = router.route(arr, sr)
        hyp = ""
        if args.with_luo_asr and r["use_luo_specialist"]:
            try:
                hyp = maybe_luo_hyp_mms1b(arr, sr, device)
            except Exception as e:
                logger.warning("luo asr failed: %s", e)
                hyp = ""
        rows.append(
            {
                "ID": sid,
                "route": r["route"],
                "top1_lang": r["top1_lang"],
                "top1_p": r["top1_p"],
                "mass_luo": r["mass_luo"],
                "mass_ach": r["mass_ach"],
                "mass_bantu": r["mass_bantu"],
                "use_luo_specialist": r["use_luo_specialist"],
                "luo_hyp": normalize_text(hyp) if hyp else "",
            }
        )
        logger.info("%s route=%s luo_mass=%.3f use_spec=%s", sid, r["route"], r["mass_luo"], r["use_luo_specialist"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    meta = {
        "n": len(rows),
        "out": str(args.out),
        "bans": [
            "no residual ctc conf thr as primary",
            "no mass multi-adapter rewrite all lid=luo",
            "no test gold train",
        ],
        "note": "Acoustic mass routing only; Luo ASR optional when use_luo_specialist",
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s n=%d", args.out, len(rows))
    if not rows or any(not r["route"] for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
