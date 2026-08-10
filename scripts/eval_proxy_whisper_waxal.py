#!/usr/bin/env python3
"""Proxy A/B: waxal-benchmarking whisper-small per true lang vs openset multihyp + true-lang CTC.

Uses HF validation only (proxy_val_index). No Phase-1 test gold.
Scores with src.metrics.score_pairs → zindi_est.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("proxy_whisper_waxal")

PROXY_CSV = PROJECT_ROOT / "data" / "proxy_val_index.csv"
WHISPER_LANGS = ("ach", "lug", "nyn", "sog", "mas")
WAXAL_WHISPER = {
    lang: f"waxal-benchmarking/whisper-small-waxal-{lang}" for lang in WHISPER_LANGS
}
CANDS = {
    "ach": ["ach", "lug", "sog"],
    "lug": ["lug", "nyn", "sog"],
    "nyn": ["nyn", "lug", "sog"],
    "sog": ["sog", "lug"],
    "mas": ["mas", "lug"],
}


def zindi_from_pairs(refs, hyps):
    s = score_pairs(refs, hyps)
    return {
        "n": int(s["n"]),
        "wer": float(s["wer"]),
        "cer": float(s["cer"]),
        "error": float(s["score"]),
        "zindi_est": float(1.0 - s["score"]),
    }


def load_waxal_ctc(lang: str, device: torch.device):
    mid = f"waxal-benchmarking/mms-300m-waxal-{lang}"
    p = AutoProcessor.from_pretrained(mid)
    m = Wav2Vec2ForCTC.from_pretrained(mid).to(device).eval()
    return m, p


@torch.inference_mode()
def whisper_transcribe(model, processor, array: np.ndarray, sr: int, device: torch.device) -> str:
    array = np.asarray(array, dtype=np.float32)
    if sr != 16000:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        sr = 16000
    peak = float(np.max(np.abs(array)) + 1e-9)
    array = array / peak
    inputs = processor(array, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    # Avoid max_length / max_new_tokens clash from some generation configs
    gen_kwargs = {
        "max_new_tokens": 128,
        "num_beams": 1,
        "do_sample": False,
    }
    try:
        # strip conflicting fields if present
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None
    except Exception:
        pass
    ids = model.generate(input_features, **gen_kwargs)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return normalize_text(text) or "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=list(WHISPER_LANGS))
    ap.add_argument("--max-per-lang", type=int, default=40)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "proxy_whisper_waxal.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    proxy = pd.read_csv(PROXY_CSV)
    proxy["id"] = proxy["id"].astype(str)

    # Load audio keyed by id
    audio_by_id: dict[str, tuple[np.ndarray, int]] = {}
    refs_by_id: dict[str, str] = {}
    lang_by_id: dict[str, str] = {}
    for lang in args.langs:
        sub = proxy[proxy.language == lang]
        if args.max_per_lang:
            sub = sub.head(args.max_per_lang)
        if sub.empty:
            continue
        logger.info("load HF val audio lang=%s n=%d", lang, len(sub))
        ds = load_hf_asr_split(lang, "validation", max_samples=None)
        # index dataset by constructed id if present
        want = set(sub["id"].tolist())
        for i in range(len(ds)):
            row = ds[i]
            # build id similar to proxy builder
            rid = None
            for key in ("id", "ID", "utterance_id"):
                if key in row and row[key] is not None:
                    rid = str(row[key])
                    break
            if rid is None:
                # fallback: language_index used by build_proxy_val_index
                rid = f"{lang}_{i}"
            # also try lang-prefixed forms
            candidates = {rid, f"{lang}_{rid}", str(row.get("path", ""))}
            matched = want & candidates
            if not matched:
                # match by transcription equality as last resort is expensive; skip
                continue
            mid = next(iter(matched))
            if mid not in want:
                continue
            audio = row["audio"]
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            audio_by_id[mid] = (arr, sr)
            refs_by_id[mid] = normalize_text(str(sub.set_index("id").loc[mid, "transcription"]))
            lang_by_id[mid] = lang
            want.discard(mid)
            if not want:
                break
        # if still missing, iterate all and match by normalized transcription
        if want:
            logger.info("fallback match for %d ids lang=%s", len(want), lang)
            ref_map = {
                normalize_text(str(r.transcription)): str(r.id)
                for _, r in sub.iterrows()
            }
            for i in range(len(ds)):
                row = ds[i]
                t = normalize_text(str(row.get("transcription") or row.get("text") or ""))
                if t in ref_map and ref_map[t] in want:
                    mid = ref_map[t]
                    audio = row["audio"]
                    arr = np.asarray(audio["array"], dtype=np.float32)
                    sr = int(audio["sampling_rate"])
                    audio_by_id[mid] = (arr, sr)
                    refs_by_id[mid] = t
                    lang_by_id[mid] = lang
                    want.discard(mid)
                    if not want:
                        break
        logger.info("matched lang=%s audio=%d missing=%d", lang, sum(1 for x in lang_by_id.values() if x == lang), len(want))

    ids = sorted(audio_by_id.keys())
    logger.info("total matched clips %d", len(ids))
    if len(ids) < 20:
        raise SystemExit(f"too few matched clips: {len(ids)}")

    # 1) true-lang CTC
    ctc_hyps = {}
    ctc_cache = {}
    for lang in sorted(set(lang_by_id.values())):
        logger.info("CTC true-lang %s", lang)
        m, p = load_waxal_ctc(lang, device)
        ctc_cache[lang] = (m, p)
        for uid in ids:
            if lang_by_id[uid] != lang:
                continue
            arr, sr = audio_by_id[uid]
            hyp = normalize_text(transcribe_waveform(m, p, arr, sr, device=device)) or "."
            ctc_hyps[uid] = hyp
        del m
        torch.mps.empty_cache() if device.type == "mps" else None

    # 2) multihyp conf (baseline openset-equivalent)
    # load remaining cand langs
    needed = set()
    for lang in set(lang_by_id.values()):
        needed.update(CANDS.get(lang, [lang]))
    for lang in sorted(needed):
        if lang not in ctc_cache:
            logger.info("load cand CTC %s", lang)
            ctc_cache[lang] = load_waxal_ctc(lang, device)

    multi_hyps = {}
    for uid in tqdm(ids, desc="multihyp"):
        tlang = lang_by_id[uid]
        arr, sr = audio_by_id[uid]
        best_hyp, best_c = None, -1e9
        for cand in CANDS.get(tlang, [tlang]):
            m, p = ctc_cache[cand]
            hyp, conf = transcribe_waveform(m, p, arr, sr, device=device, return_confidence=True)
            hyp = normalize_text(hyp) or "."
            if conf > best_c:
                best_c, best_hyp = conf, hyp
        multi_hyps[uid] = best_hyp

    # free CTC
    ctc_cache.clear()
    if device.type == "mps":
        torch.mps.empty_cache()

    # 3) whisper-small true-lang
    wh_hyps = {}
    for lang in sorted(set(lang_by_id.values())):
        mid = WAXAL_WHISPER.get(lang)
        if not mid:
            logger.warning("no whisper for %s — skip (use CTC hyp)", lang)
            for uid in ids:
                if lang_by_id[uid] == lang:
                    wh_hyps[uid] = ctc_hyps[uid]
            continue
        logger.info("Whisper true-lang %s %s", lang, mid)
        try:
            proc = WhisperProcessor.from_pretrained(mid)
            model = WhisperForConditionalGeneration.from_pretrained(mid)
            model.to(device).eval()
        except Exception as e:
            logger.error("failed load %s: %s", mid, e)
            for uid in ids:
                if lang_by_id[uid] == lang:
                    wh_hyps[uid] = ctc_hyps.get(uid, ".")
            continue
        for uid in tqdm([u for u in ids if lang_by_id[u] == lang], desc=f"wh-{lang}"):
            arr, sr = audio_by_id[uid]
            try:
                wh_hyps[uid] = whisper_transcribe(model, proc, arr, sr, device)
            except Exception as e:
                logger.warning("whisper fail %s: %s", uid, e)
                wh_hyps[uid] = ctc_hyps.get(uid, ".")
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    refs = [refs_by_id[u] for u in ids]
    results = {
        "n": len(ids),
        "langs": sorted(set(lang_by_id.values())),
        "systems": {
            "true_lang_ctc": zindi_from_pairs(refs, [ctc_hyps[u] for u in ids]),
            "openset_multihyp": zindi_from_pairs(refs, [multi_hyps[u] for u in ids]),
            "true_lang_whisper_small": zindi_from_pairs(refs, [wh_hyps[u] for u in ids]),
        },
    }
    base = results["systems"]["openset_multihyp"]["zindi_est"]
    for name, m in results["systems"].items():
        m["delta_vs_multihyp"] = m["zindi_est"] - base
        m["gate_pass"] = (m["zindi_est"] - base) >= 0.01

    # per-lang
    per = {}
    for lang in sorted(set(lang_by_id.values())):
        uids = [u for u in ids if lang_by_id[u] == lang]
        r = [refs_by_id[u] for u in uids]
        per[lang] = {
            "n": len(uids),
            "true_lang_ctc": zindi_from_pairs(r, [ctc_hyps[u] for u in uids]),
            "openset_multihyp": zindi_from_pairs(r, [multi_hyps[u] for u in uids]),
            "true_lang_whisper_small": zindi_from_pairs(r, [wh_hyps[u] for u in uids]),
        }
    results["per_lang"] = per
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.out)
    print(json.dumps(results["systems"], indent=2))


if __name__ == "__main__":
    main()
