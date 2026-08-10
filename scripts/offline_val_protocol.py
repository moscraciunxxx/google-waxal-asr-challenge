#!/usr/bin/env python3
"""Shared offline validation protocol for Phase-2 pipeline A/B.

Labeled HF validation only. Language metadata stripped at decode for open-set
pipelines. Never uses Phase-1 test gold for train/tune/select.

Offline score: S = 1 - 0.5*WER - 0.5*CER (higher better). Target S >= 0.8.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")  # may need hub for first load
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORBIDDEN_TRAIN_SPLITS, HF_DATASET, OUTPUT_DIR, TARGET_SR
from src.metrics import score_pairs
from src.mms_infer import pick_device, set_lang, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("offline_val")

# Open-set languages present in Phase-2 LID mass + challenge 3
OFFLINE_LANGS = ("lin", "sna", "lug", "ach", "nyn", "sog")
WAXAL300 = {
    lang: f"waxal-benchmarking/mms-300m-waxal-{lang}"
    for lang in ("lin", "sna", "lug", "ach", "nyn", "sog", "mas")
}

# LID closed-set for multi-lang: among these codes only (when using mms-lid)
LID_CANDIDATES = ("lin", "sna", "lug", "ach", "nyn", "sog", "mas", "luo")


def load_val(lang: str, max_samples: int | None):
    if lang in FORBIDDEN_TRAIN_SPLITS:  # never
        raise ValueError(lang)
    config = f"{lang}_asr"
    # Prefer streaming take for large vals when capped
    if max_samples is not None and max_samples <= 80:
        ds = load_dataset(HF_DATASET, config, split="validation", streaming=True)
        rows = []
        for i, ex in enumerate(ds):
            if i >= max_samples:
                break
            rows.append(ex)
        from datasets import Dataset

        return Dataset.from_list(rows)
    ds = load_dataset(HF_DATASET, config, split="validation")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def audio_array(ex) -> tuple[np.ndarray, int]:
    from src.dataset import _decode_audio_item

    a = _decode_audio_item(ex["audio"], TARGET_SR)
    return np.asarray(a["array"], dtype=np.float32), int(a["sampling_rate"])


class WaxalCache:
    def __init__(self, device: torch.device):
        self.device = device
        self.models: dict[str, tuple] = {}

    def get(self, lang: str):
        if lang not in WAXAL300:
            lang = "lug"
        if lang in self.models:
            return self.models[lang]
        mid = WAXAL300[lang]
        logger.info("load %s", mid)
        try:
            proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
            model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
        except Exception:
            proc = AutoProcessor.from_pretrained(mid)
            model = Wav2Vec2ForCTC.from_pretrained(mid)
        model.to(self.device).eval()
        self.models[lang] = (model, proc)
        return self.models[lang]


class Lid126:
    def __init__(self, device: torch.device):
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        mid = "facebook/mms-lid-126"
        self.device = device
        self.feat = AutoFeatureExtractor.from_pretrained(mid)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(mid).to(device).eval()
        self.label2id = {
            v: int(k) for k, v in self.model.config.id2label.items() if v in LID_CANDIDATES
        }
        self.langs = [l for l in LID_CANDIDATES if l in self.label2id]
        self.idxs = [self.label2id[l] for l in self.langs]
        logger.info("LID closed-set %s", self.langs)

    @torch.inference_mode()
    def predict(self, arr: np.ndarray, sr: int) -> str:
        peak = float(np.max(np.abs(arr)) + 1e-9)
        arr = arr / peak
        if sr != TARGET_SR:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        inputs = self.feat(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        logits = self.model(inputs.input_values.to(self.device)).logits[0]
        sub = logits[self.idxs]
        best = int(torch.argmax(sub).item())
        return self.langs[best]


def map_lid_to_decode(lid: str) -> list[str]:
    """Candidate waxal-300m langs for multi-hyp (openset policy)."""
    if lid == "luo":
        return ["ach", "lug", "sog"]
    if lid == "lug":
        return ["lug", "nyn", "sog"]
    if lid in WAXAL300:
        return [lid]
    if lid in ("lin", "sna"):
        return [lid]
    return ["lug", "ach", "nyn"]


@torch.inference_mode()
def decode_best(cache: WaxalCache, arr, sr, device, candidates: list[str]):
    best_t, best_l, best_c = ".", candidates[0], -1e9
    for lang in candidates:
        model, proc = cache.get(lang)
        text, conf = transcribe_waveform(
            model, proc, arr, sr, device=device, return_confidence=True
        )
        if conf > best_c:
            best_t, best_l, best_c = text, lang, conf
    return best_t, best_l, best_c


def run_pipeline(
    name: str,
    langs: tuple[str, ...],
    max_per_lang: int,
    device: torch.device,
    mode: str,
) -> dict:
    """mode: oracle_waxal | openset_multihyp | lid_map_ach | challenge3_conf"""
    cache = WaxalCache(device)
    lid = None
    if mode in ("openset_multihyp", "lid_map_ach", "challenge3_conf"):
        lid = Lid126(device)

    rows = []
    for lang in langs:
        try:
            ds = load_val(lang, max_per_lang)
        except Exception as e:
            logger.warning("skip %s: %s", lang, e)
            continue
        logger.info("%s %s n=%d", name, lang, len(ds))
        for i in tqdm(range(len(ds)), desc=f"{name}-{lang}"):
            ex = ds[i]
            arr, sr = audio_array(ex)
            ref = normalize_text(ex.get("transcription") or ex.get("text") or "")
            if not ref:
                continue
            if mode == "oracle_waxal":
                # true lang decode with waxal-300m (oracle language)
                model, proc = cache.get(lang if lang in WAXAL300 else "lug")
                hyp = transcribe_waveform(model, proc, arr, sr, device=device)
                dlang = lang
            elif mode == "openset_multihyp":
                pred_lid = lid.predict(arr, sr)
                cands = map_lid_to_decode(pred_lid)
                # ensure true-lang model is in candidates when available (still strip label from pick)
                hyp, dlang, _ = decode_best(cache, arr, sr, device, cands)
            elif mode == "lid_map_ach":
                pred_lid = lid.predict(arr, sr)
                if pred_lid == "luo":
                    dlang = "ach"
                elif pred_lid in WAXAL300:
                    dlang = pred_lid
                else:
                    dlang = "lug"
                model, proc = cache.get(dlang)
                hyp = transcribe_waveform(model, proc, arr, sr, device=device)
            elif mode == "challenge3_conf":
                hyp, dlang, _ = decode_best(cache, arr, sr, device, ["lin", "sna", "lug"])
            else:
                raise ValueError(mode)
            rows.append(
                {
                    "true_lang": lang,
                    "decode_lang": dlang,
                    "ref": ref,
                    "hyp": hyp,
                }
            )

    refs = [r["ref"] for r in rows]
    hyps = [r["hyp"] for r in rows]
    sc = score_pairs(refs, hyps)
    S = 1.0 - sc["score"]
    # per true lang
    by = defaultdict(lambda: {"ref": [], "hyp": []})
    for r in rows:
        by[r["true_lang"]]["ref"].append(r["ref"])
        by[r["true_lang"]]["hyp"].append(r["hyp"])
    per = {}
    for lang, d in by.items():
        p = score_pairs(d["ref"], d["hyp"])
        per[lang] = {"wer": p["wer"], "cer": p["cer"], "S": 1.0 - p["score"], "n": int(p["n"])}
    route_acc = sum(r["true_lang"] == r["decode_lang"] for r in rows) / max(len(rows), 1)
    out = {
        "name": name,
        "mode": mode,
        "n": len(rows),
        "wer": sc["wer"],
        "cer": sc["cer"],
        "error": sc["score"],
        "S": S,
        "route_acc": route_acc,
        "decode_mass": dict(Counter(r["decode_lang"] for r in rows)),
        "true_mass": dict(Counter(r["true_lang"] for r in rows)),
        "per_lang": per,
    }
    logger.info("RESULT %s S=%.4f wer=%.3f cer=%.3f n=%d", name, S, sc["wer"], sc["cer"], len(rows))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--modes",
        nargs="+",
        default=["oracle_waxal", "openset_multihyp", "challenge3_conf"],
    )
    p.add_argument("--langs", nargs="+", default=list(OFFLINE_LANGS))
    p.add_argument("--max-per-lang", type=int, default=40)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "phase2_offline_ab.json")
    p.add_argument("--scratch", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    results = []
    for mode in args.modes:
        res = run_pipeline(
            name=mode,
            langs=tuple(args.langs),
            max_per_lang=args.max_per_lang,
            device=device,
            mode=mode,
        )
        results.append(res)
        if args.scratch:
            args.scratch.mkdir(parents=True, exist_ok=True)
            (args.scratch / f"probe_{mode}.json").write_text(json.dumps(res, indent=2))

    best = max(results, key=lambda r: r["S"])
    payload = {
        "results": results,
        "best": best["name"],
        "best_S": best["S"],
        "target_S": 0.8,
        "met_target": best["S"] >= 0.8,
        "max_per_lang": args.max_per_lang,
        "langs": args.langs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    if args.scratch:
        (args.scratch / "offline_ab.json").write_text(json.dumps(payload, indent=2))
        md = ["# Offline pipeline A/B", ""]
        md.append("| pipeline | S | WER | CER | n | route_acc |")
        md.append("|----------|--:|----:|----:|--:|----------:|")
        for r in sorted(results, key=lambda x: -x["S"]):
            md.append(
                f"| {r['name']} | {r['S']:.4f} | {r['wer']:.3f} | {r['cer']:.3f} | {r['n']} | {r['route_acc']:.3f} |"
            )
        md.append("")
        md.append(f"**Best:** {best['name']} S={best['S']:.4f} target_met={best['S']>=0.8}")
        (args.scratch / "pipeline_ab.md").write_text("\n".join(md))
        (OUTPUT_DIR / "phase2_pipeline_ab.md").write_text("\n".join(md))
    print(json.dumps(payload, indent=2))
    if best["S"] < 0.8:
        sys.exit(2)  # signal not yet at target
    sys.exit(0)


if __name__ == "__main__":
    main()
