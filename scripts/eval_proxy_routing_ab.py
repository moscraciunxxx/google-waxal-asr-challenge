#!/usr/bin/env python3
"""Proxy routing A/B with full MMS-LID-126 + selective beam upgrades.

Scores zindi_est for:
  baseline: openset-style multihyp conf (true_lang candidate map)
  R1: multihyp + ach/nyn beam + ft_lug on multihyp pick (deployable C+)
  R2: LID high-p1 single-lang when lid in waxal; else multihyp; then upgrades
  R3: force lid=lug → ft_lug; lid=luo p1>=thr → ach beam; else R1
  R4: oracle true-lang + upgrades (upper bound)

No Phase-1 test gold. Writes outputs/proxy_routing_ab.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, OUTPUT_DIR, TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import fix_mms_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("proxy_routing")

CANDS = {
    "ach": ["ach", "lug", "sog"],
    "lug": ["lug", "nyn", "sog"],
    "nyn": ["nyn", "lug", "sog"],
    "sog": ["sog", "lug"],
    "mas": ["mas", "lug"],
    "luo": ["ach", "lug", "sog"],
}
WAXAL = {
    lang: f"waxal-benchmarking/mms-300m-waxal-{lang}"
    for lang in ("ach", "lug", "nyn", "sog", "mas")
}


class FullLid:
    def __init__(self, device: torch.device):
        mid = "facebook/mms-lid-126"
        self.device = device
        self.feat = AutoFeatureExtractor.from_pretrained(mid)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(mid).to(device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.inference_mode()
    def predict(self, array: np.ndarray, sr: int) -> tuple[str, float, list[tuple[str, float]]]:
        if sr != TARGET_SR:
            import librosa

            array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
        inputs = self.feat(array, sampling_rate=TARGET_SR, return_tensors="pt")
        logits = self.model(inputs.input_values.to(self.device)).logits[0]
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        order = np.argsort(-probs)[:5]
        top = [(self.id2label[int(i)], float(probs[int(i)])) for i in order]
        return top[0][0], top[0][1], top


def zindi(refs, hyps):
    s = score_pairs(refs, hyps)
    return {
        "n": int(s["n"]),
        "wer": float(s["wer"]),
        "cer": float(s["cer"]),
        "zindi_est": float(1.0 - s["score"]),
    }


def mk_dec(proc, lang, alpha=0.3, beta=0.5):
    vocab = proc.tokenizer.get_vocab()
    id2 = {i: t for t, i in vocab.items()}
    labels = [id2[i] for i in range(len(id2))]
    return build_ctcdecoder(
        labels,
        kenlm_model_path=str(ROOT / "data" / "lms" / f"{lang}_2gram.arpa"),
        unigrams=[
            u
            for u in (ROOT / "data" / "lms" / f"{lang}_unigrams.txt").read_text().splitlines()
            if u.strip()
        ],
        alpha=alpha,
        beta=beta,
    )


@torch.inference_mode()
def beam(model, proc, dec, arr, sr, device, bw=50):
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    arr = arr / (float(np.max(np.abs(arr)) + 1e-9))
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
    return normalize_text(dec.decode(logits, beam_width=bw).replace("|", " ")) or "."


def guard(g, b):
    r = max(1, len(b.split())) / max(1, len(g.split()))
    return b if 0.5 <= r <= 2.0 else g


def main():
    device = pick_device()
    logger.info("device %s", device)
    proxy = pd.read_csv(ROOT / "data" / "proxy_val_index.csv")
    audio, refs, langs = {}, {}, {}
    for lang in ["ach", "lug", "nyn", "sog", "mas"]:
        sub = proxy[proxy.language == lang].head(40)
        ds = load_hf_asr_split(lang, "validation")
        want = {normalize_text(str(r.transcription)): str(r.id) for _, r in sub.iterrows()}
        for i in range(len(ds)):
            row = ds[i]
            t = normalize_text(str(row.get("transcription") or row.get("text") or ""))
            if t in want:
                uid = want[t]
                a = row["audio"]
                audio[uid] = (np.asarray(a["array"], dtype=np.float32), int(a["sampling_rate"]))
                refs[uid] = t
                langs[uid] = lang
                del want[t]
                if not want:
                    break
    ids = sorted(audio.keys())
    logger.info("matched %d", len(ids))

    # LID
    lid_m = FullLid(device)
    lid_pred = {}
    for uid in tqdm(ids, desc="lid"):
        arr, sr = audio[uid]
        lang1, p1, top = lid_m.predict(arr, sr)
        lid_pred[uid] = {"lang1": lang1, "p1": p1, "top": top}
    del lid_m
    if device.type == "mps":
        torch.mps.empty_cache()

    # ASR models
    cache = {}

    def get(lang):
        if lang not in cache:
            mid = WAXAL[lang]
            p = AutoProcessor.from_pretrained(mid)
            m = Wav2Vec2ForCTC.from_pretrained(mid).to(device).eval()
            cache[lang] = (m, p)
        return cache[lang]

    ft_p = AutoProcessor.from_pretrained(str(CHECKPOINT_DIR / "mms-lug-ft-v2"), local_files_only=True)
    ft_m = Wav2Vec2ForCTC.from_pretrained(str(CHECKPOINT_DIR / "mms-lug-ft-v2"), local_files_only=True)
    fix_mms_tokenizer(ft_p, "lug")
    ft_m.to(device).eval()
    ach_m, ach_p = get("ach")
    nyn_m, nyn_p = get("nyn")
    dec_ach = mk_dec(ach_p, "ach")
    dec_nyn = mk_dec(nyn_p, "nyn")

    def multihyp(arr, sr, cands):
        best_h, best_c, best_l = None, -1e9, None
        for cand in cands:
            m, p = get(cand)
            hyp, conf = transcribe_waveform(m, p, arr, sr, device=device, return_confidence=True)
            hyp = normalize_text(hyp) or "."
            if conf > best_c:
                best_h, best_c, best_l = hyp, conf, cand
        return best_h, best_l, best_c

    def upgrade(arr, sr, dlang):
        if dlang == "ach":
            g = normalize_text(transcribe_waveform(ach_m, ach_p, arr, sr, device=device)) or "."
            return guard(g, beam(ach_m, ach_p, dec_ach, arr, sr, device))
        if dlang == "lug":
            return normalize_text(transcribe_waveform(ft_m, ft_p, arr, sr, device=device)) or "."
        if dlang == "nyn":
            g = normalize_text(transcribe_waveform(nyn_m, nyn_p, arr, sr, device=device)) or "."
            return guard(g, beam(nyn_m, nyn_p, dec_nyn, arr, sr, device))
        m, p = get(dlang) if dlang in WAXAL else get("lug")
        return normalize_text(transcribe_waveform(m, p, arr, sr, device=device)) or "."

    # collect hyps per recipe
    recipes = {
        "baseline_multihyp": {},
        "R1_multihyp_upgrade": {},
        "R2_highp1_single_then_upgrade": {},
        "R3_force_lid_lug_and_luo_ach": {},
        "R4_oracle_true_lang_upgrade": {},
    }
    route_stats = {k: {} for k in recipes}

    for uid in tqdm(ids, desc="decode"):
        arr, sr = audio[uid]
        tlang = langs[uid]
        lid1 = lid_pred[uid]["lang1"]
        p1 = lid_pred[uid]["p1"]

        # baseline multihyp by true_lang map (openset-equivalent protocol)
        cands = CANDS.get(tlang, ["lug", "ach"])
        h0, d0, _ = multihyp(arr, sr, cands)
        recipes["baseline_multihyp"][uid] = h0

        # R1: same multihyp then upgrade
        recipes["R1_multihyp_upgrade"][uid] = upgrade(arr, sr, d0)
        route_stats["R1_multihyp_upgrade"][d0] = route_stats["R1_multihyp_upgrade"].get(d0, 0) + 1

        # R2: if LID p1 high and lid in waxal, single; elif lid=luo high, multihyp luo map; else multihyp true map
        if p1 >= 0.9 and lid1 in WAXAL:
            d2 = lid1
            h2 = upgrade(arr, sr, d2)
        elif p1 >= 0.9 and lid1 == "luo":
            h_tmp, d2, _ = multihyp(arr, sr, CANDS["luo"])
            h2 = upgrade(arr, sr, d2)
        else:
            h_tmp, d2, _ = multihyp(arr, sr, cands)
            h2 = upgrade(arr, sr, d2)
        recipes["R2_highp1_single_then_upgrade"][uid] = h2
        route_stats["R2_highp1_single_then_upgrade"][d2] = (
            route_stats["R2_highp1_single_then_upgrade"].get(d2, 0) + 1
        )

        # R3: force lid=lug → ft_lug; lid=luo p1>=0.9 → ach beam; else R1
        if lid1 == "lug" and p1 >= 0.8:
            d3 = "lug"
            h3 = upgrade(arr, sr, "lug")
        elif lid1 == "luo" and p1 >= 0.9:
            d3 = "ach"
            h3 = upgrade(arr, sr, "ach")
        else:
            d3 = d0
            h3 = upgrade(arr, sr, d0)
        recipes["R3_force_lid_lug_and_luo_ach"][uid] = h3
        route_stats["R3_force_lid_lug_and_luo_ach"][d3] = (
            route_stats["R3_force_lid_lug_and_luo_ach"].get(d3, 0) + 1
        )

        # R4 oracle
        recipes["R4_oracle_true_lang_upgrade"][uid] = upgrade(arr, sr, tlang)

    base = zindi([refs[i] for i in ids], [recipes["baseline_multihyp"][i] for i in ids])
    out = {
        "n": len(ids),
        "baseline": base,
        "results": {},
        "lid_vs_true": {
            "acc": float(np.mean([lid_pred[i]["lang1"] == langs[i] for i in ids])),
            "acc_mapped": float(
                np.mean(
                    [
                        (lid_pred[i]["lang1"] == langs[i])
                        or (lid_pred[i]["lang1"] == "luo" and langs[i] == "ach")
                        for i in ids
                    ]
                )
            ),
        },
        "route_stats": route_stats,
    }
    for name, hyps in recipes.items():
        if name == "baseline_multihyp":
            continue
        m = zindi([refs[i] for i in ids], [hyps[i] for i in ids])
        m["delta"] = m["zindi_est"] - base["zindi_est"]
        m["gate_pass"] = m["delta"] >= 0.01
        out["results"][name] = m
        logger.info("%s %s", name, m)

    best = max(out["results"].items(), key=lambda kv: kv[1]["zindi_est"])
    out["best"] = {"name": best[0], **best[1]}
    # best deployable excludes oracle
    dep = {k: v for k, v in out["results"].items() if "oracle" not in k}
    best_d = max(dep.items(), key=lambda kv: kv[1]["zindi_est"])
    out["best_deployable"] = {"name": best_d[0], **best_d[1]}

    path = OUTPUT_DIR / "proxy_routing_ab.json"
    path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", path)
    print(json.dumps({"best_deployable": out["best_deployable"], "best": out["best"], "lid": out["lid_vs_true"]}, indent=2))


if __name__ == "__main__":
    main()
