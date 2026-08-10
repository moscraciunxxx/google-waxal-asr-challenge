#!/usr/bin/env python3
"""Offline Zindi-est evaluation on HF validation (audio-only at decode).

Target gate: zindi_est = 1 - 0.5*WER - 0.5*CER >= 0.8

Never uses Phase-1 test gold for training/tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LANGUAGES, OUTPUT_DIR, TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.mms_infer import load_mms, pick_device, set_lang, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_offline")

WAXAL300 = {
    "lin": "waxal-benchmarking/mms-300m-waxal-lin",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
    "ach": "waxal-benchmarking/mms-300m-waxal-ach",
    "nyn": "waxal-benchmarking/mms-300m-waxal-nyn",
    "sog": "waxal-benchmarking/mms-300m-waxal-sog",
}


def zindi_from_pairs(refs, hyps) -> dict:
    s = score_pairs(refs, hyps)
    return {
        "wer": s["wer"],
        "cer": s["cer"],
        "error": s["score"],
        "zindi_est": 1.0 - s["score"],
        "n": int(s["n"]),
    }


def load_waxal(lang: str, device: torch.device):
    mid = WAXAL300[lang]
    try:
        processor = AutoProcessor.from_pretrained(mid, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(mid)
        model = Wav2Vec2ForCTC.from_pretrained(mid)
    model.to(device).eval()
    return model, processor


def load_ft_v2(lang: str, device: torch.device):
    from pathlib import Path as P
    from scripts.mms_adapter_ft import fix_mms_tokenizer

    ckpt = P("checkpoints") / f"mms-{lang}-ft-v2"
    if not (ckpt / "model.safetensors").exists() and not (ckpt / "pytorch_model.bin").exists():
        return None, None
    processor = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
    try:
        fix_mms_tokenizer(processor, lang)
    except Exception as e:
        logger.warning("fix_mms_tokenizer: %s", e)
    model.to(device).eval()
    return model, processor


@torch.inference_mode()
def multi_hyp_pick(models: dict, arr, sr, device, candidates: list[str]):
    best_t, best_l, best_c = ".", candidates[0], -1e9
    for lang in candidates:
        m, p = models[lang]
        text, conf = transcribe_waveform(m, p, arr, sr, device=device, return_confidence=True)
        if conf > best_c:
            best_t, best_l, best_c = text, lang, conf
    return best_t, best_l, best_c


def run_method(
    name: str,
    samples: list[dict],
    decode_fn,
) -> dict:
    refs, hyps, picks = [], [], []
    t0 = time.time()
    for s in tqdm(samples, desc=name):
        hyp, pick = decode_fn(s)
        refs.append(s["ref"])
        hyps.append(hyp)
        picks.append(pick)
    metrics = zindi_from_pairs(refs, hyps)
    lid_acc = sum(p == s["lang"] for p, s in zip(picks, samples)) / max(len(samples), 1)
    metrics["method"] = name
    metrics["lid_or_pick_acc"] = lid_acc
    metrics["seconds"] = time.time() - t0
    logger.info(
        "%s zindi=%.4f wer=%.4f cer=%.4f acc=%.3f n=%d",
        name,
        metrics["zindi_est"],
        metrics["wer"],
        metrics["cer"],
        lid_acc,
        metrics["n"],
    )
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-per-lang", type=int, default=40)
    p.add_argument("--device", default=None)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--gate", type=float, default=0.8)
    p.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR / "offline_zindi_eval.json",
    )
    args = p.parse_args()
    device = torch.device(args.device) if args.device else pick_device()
    logger.info("device=%s", device)

    samples = []
    for lang in LANGUAGES:
        n = args.max_per_lang
        if lang == "lin":
            n = min(n, 34)
        ds = load_hf_asr_split(lang, "validation", max_samples=n)
        for i in range(len(ds)):
            ex = ds[i]
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"]["sampling_rate"])
            samples.append(
                {
                    "lang": lang,
                    "ref": normalize_text(ex["transcription"]),
                    "arr": arr,
                    "sr": sr,
                    "id": ex["id"],
                }
            )
    logger.info("samples=%d", len(samples))

    # Load model banks lazily
    waxal_models = {}
    ft_models = {}
    zs_bundle = None

    def ensure_waxal(langs):
        for lang in langs:
            if lang not in waxal_models and lang in WAXAL300:
                logger.info("load waxal %s", lang)
                waxal_models[lang] = load_waxal(lang, device)

    def ensure_ft(langs):
        for lang in langs:
            if lang not in ft_models:
                m, p = load_ft_v2(lang, device)
                if m is not None:
                    ft_models[lang] = (m, p)

    def ensure_zs():
        nonlocal zs_bundle
        if zs_bundle is None:
            zs_bundle = load_mms(device=device)
            model, processor, _ = zs_bundle
            for lang in LANGUAGES:
                set_lang(model, processor, lang)

    results = {"samples": len(samples), "gate": args.gate, "methods": {}}

    def method_oracle_waxal(s):
        ensure_waxal([s["lang"]])
        m, p = waxal_models[s["lang"]]
        hyp = transcribe_waveform(m, p, s["arr"], s["sr"], device=device)
        return hyp, s["lang"]

    def method_oracle_ft(s):
        ensure_ft([s["lang"]])
        if s["lang"] in ft_models:
            m, p = ft_models[s["lang"]]
            hyp = transcribe_waveform(m, p, s["arr"], s["sr"], device=device)
        else:
            ensure_zs()
            model, processor, _ = zs_bundle
            set_lang(model, processor, s["lang"])
            hyp = transcribe_waveform(model, processor, s["arr"], s["sr"], device=device)
        return hyp, s["lang"]

    def method_oracle_zs(s):
        ensure_zs()
        model, processor, _ = zs_bundle
        set_lang(model, processor, s["lang"])
        hyp = transcribe_waveform(model, processor, s["arr"], s["sr"], device=device)
        return hyp, s["lang"]

    def method_multihyp_waxal3(s):
        ensure_waxal(list(LANGUAGES))
        hyp, pick, _ = multi_hyp_pick(waxal_models, s["arr"], s["sr"], device, list(LANGUAGES))
        return hyp, pick

    def method_multihyp_waxal_openset(s):
        # mirrors phase2 openset candidates per true lang group
        cands = {
            "lin": ["lin", "lug", "ach"],
            "sna": ["sna", "lug", "nyn"],
            "lug": ["lug", "nyn", "sog", "ach"],
        }[s["lang"]]
        ensure_waxal(cands)
        hyp, pick, _ = multi_hyp_pick(waxal_models, s["arr"], s["sr"], device, cands)
        return hyp, pick

    def method_multihyp_ft3(s):
        ensure_ft(list(LANGUAGES))
        # fall back missing to zs
        ensure_zs()
        bank = {}
        for lang in LANGUAGES:
            if lang in ft_models:
                bank[lang] = ft_models[lang]
            else:
                model, processor, _ = zs_bundle
                set_lang(model, processor, lang)
                bank[lang] = (model, processor)
        hyp, pick, _ = multi_hyp_pick(bank, s["arr"], s["sr"], device, list(LANGUAGES))
        return hyp, pick

    def method_blend_best(s):
        """Oracle pick between waxal and ft on CTC conf of true lang only, then multi for others."""
        ensure_waxal([s["lang"]])
        ensure_ft([s["lang"]])
        cands = []
        texts = {}
        confs = {}
        m, p = waxal_models[s["lang"]]
        t, c = transcribe_waveform(m, p, s["arr"], s["sr"], device=device, return_confidence=True)
        texts["waxal"] = t
        confs["waxal"] = c
        if s["lang"] in ft_models:
            m, p = ft_models[s["lang"]]
            t, c = transcribe_waveform(m, p, s["arr"], s["sr"], device=device, return_confidence=True)
            texts["ft"] = t
            confs["ft"] = c
        best = max(confs, key=confs.get)
        return texts[best], s["lang"]

    # Beam methods if pyctcdecode available
    beam_decoders = {}

    def try_beam_oracle_waxal(s, beam=50, alpha=0.4, beta=1.0):
        from pyctcdecode import build_ctcdecoder

        ensure_waxal([s["lang"]])
        m, p = waxal_models[s["lang"]]
        lang = s["lang"]
        dkey = (lang, alpha, beta)
        if dkey not in beam_decoders:
            vocab = p.tokenizer.get_vocab()
            id2 = {i: t for t, i in vocab.items()}
            labels = [id2[i] for i in range(len(id2))]
            lm = ROOT / "data" / "lms" / f"{lang}_2gram.arpa"
            uni = ROOT / "data" / "lms" / f"{lang}_unigrams.txt"
            unigrams = [u for u in uni.read_text().splitlines() if u.strip()] if uni.exists() else None
            beam_decoders[dkey] = build_ctcdecoder(
                labels,
                kenlm_model_path=str(lm) if lm.exists() else None,
                unigrams=unigrams,
                alpha=alpha,
                beta=beta,
            )
        decoder = beam_decoders[dkey]
        arr = s["arr"]
        sr = s["sr"]
        if sr != TARGET_SR:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        peak = float(np.max(np.abs(arr)) + 1e-9)
        arr = arr / peak
        inputs = p(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        with torch.inference_mode():
            logits = m(inputs.input_values.to(device)).logits[0].float().detach().cpu().numpy()
        text = decoder.decode(logits, beam_width=beam).replace("|", " ")
        return normalize_text(text) or ".", lang

    catalog = {
        "oracle_zs": method_oracle_zs,
        "oracle_ft_v2": method_oracle_ft,
        "oracle_waxal300": method_oracle_waxal,
        "multihyp_waxal3": method_multihyp_waxal3,
        "multihyp_waxal_openset": method_multihyp_waxal_openset,
        "multihyp_ft3": method_multihyp_ft3,
        "blend_waxal_ft_oracle_lang": method_blend_best,
        "beam_oracle_waxal": lambda s: try_beam_oracle_waxal(s),
        "beam_oracle_waxal_a03": lambda s: try_beam_oracle_waxal(s, alpha=0.3, beta=0.5, beam=80),
        "beam_oracle_waxal_a05": lambda s: try_beam_oracle_waxal(s, alpha=0.5, beta=1.0, beam=100),
    }

    methods = args.methods or list(catalog.keys())
    best = None
    for name in methods:
        if name not in catalog:
            logger.warning("unknown method %s", name)
            continue
        try:
            metrics = run_method(name, samples, catalog[name])
            results["methods"][name] = metrics
            if best is None or metrics["zindi_est"] > best["zindi_est"]:
                best = metrics
        except Exception as e:
            logger.exception("method %s failed: %s", name, e)
            results["methods"][name] = {"error": str(e), "zindi_est": -1}

    results["best"] = best
    results["gate_passed"] = bool(best and best["zindi_est"] >= args.gate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    # also scratch
    scratch = Path(os.environ.get("GROK_SCRATCH", str(ROOT / ".scratch")))
    try:
        (scratch / "offline_zindi_eval.json").write_text(json.dumps(results, indent=2))
    except Exception:
        pass
    print(json.dumps({"best": best, "gate_passed": results["gate_passed"]}, indent=2))
    if not results["gate_passed"]:
        raise SystemExit(f"GATE_FAIL best={best}")


if __name__ == "__main__":
    main()
