#!/usr/bin/env python3
"""Proxy A/B: length-normalized CTC conf multi-hyp + optional ROVER vs mean-conf multihyp.

Then apply selective upgrades (ach/nyn beam+guard, FT-lug) on the pick.
Gate: beat openset multihyp mean-conf by ≥0.01; also report vs current selective_beam recipe.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, OUTPUT_DIR, TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import fix_mms_tokenizer

CANDS = {
    "ach": ["ach", "lug", "sog"],
    "lug": ["lug", "nyn", "sog"],
    "nyn": ["nyn", "lug", "sog"],
    "sog": ["sog", "lug"],
    "mas": ["mas", "lug"],
}


def zindi(refs, hyps):
    s = score_pairs(refs, hyps)
    return {
        "n": int(s["n"]),
        "wer": float(s["wer"]),
        "cer": float(s["cer"]),
        "zindi_est": float(1.0 - s["score"]),
    }


@torch.inference_mode()
def decode_with_stats(model, processor, arr, sr, device):
    """Return hyp, mean_conf, nonblank_conf, path_conf, n_frames, n_nonblank.

    hyp uses the same path as src.mms_infer.transcribe_waveform.
    """
    hyp, mean_conf = transcribe_waveform(
        model, processor, arr, sr, device=device, return_confidence=True
    )
    hyp = normalize_text(hyp) or "."
    arr = np.asarray(arr, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    peak = float(np.max(np.abs(arr)) + 1e-9)
    arr = arr / peak
    inputs = processor(arr, sampling_rate=sr, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits  # [1,T,V]
    pred_ids = torch.argmax(logits, dim=-1)[0]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0]
    frame_max = log_probs.max(dim=-1).values
    try:
        blank = int(processor.tokenizer.pad_token_id or 0)
    except Exception:
        blank = 0
    mask = pred_ids != blank
    ln = float(frame_max[mask].mean().item()) if mask.any() else float(frame_max.mean().item())
    selected = log_probs[torch.arange(log_probs.size(0), device=log_probs.device), pred_ids]
    path_mean = float(selected.mean().item())
    # length penalty form used in some CTC systems: mean / (1 + log(1+T))
    t = max(1, int(log_probs.size(0)))
    lp = float(frame_max.mean().item()) / (1.0 + np.log1p(t))
    return hyp, float(mean_conf), ln, path_mean, lp, t, int(mask.sum().item())


def rover_vote(hyps: list[str]) -> str:
    """Simple word-level plurality ROVER (position by order after left-align truncate)."""
    if not hyps:
        return "."
    if len(hyps) == 1:
        return hyps[0]
    tokenized = [h.split() for h in hyps]
    # use longest as anchor alignment: for each position in majority length
    max_len = max(len(t) for t in tokenized)
    if max_len == 0:
        return "."
    out = []
    for i in range(max_len):
        votes = []
        for t in tokenized:
            if i < len(t):
                votes.append(t[i])
        if not votes:
            continue
        c = Counter(votes)
        # require at least 2 agree if 3 hyps else take plurality
        word, n = c.most_common(1)[0]
        if len(tokenized) >= 3 and n < 2:
            # prefer first hyp (best conf order)
            word = tokenized[0][i] if i < len(tokenized[0]) else word
        out.append(word)
    return normalize_text(" ".join(out)) or "."


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
    print("device", device)
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
    print("n", len(ids))

    cache = {}

    def get(lang):
        if lang not in cache:
            mid = f"waxal-benchmarking/mms-300m-waxal-{lang}"
            p = AutoProcessor.from_pretrained(mid)
            m = Wav2Vec2ForCTC.from_pretrained(mid).to(device).eval()
            cache[lang] = (m, p)
        return cache[lang]

    # collect all candidate hyps per id
    all_cands = {}  # uid -> list of dicts
    for uid in tqdm(ids, desc="cands"):
        arr, sr = audio[uid]
        tlang = langs[uid]
        rows = []
        for cand in CANDS[tlang]:
            m, p = get(cand)
            hyp, mean_c, ln, path_m, lp, nf, nnb = decode_with_stats(m, p, arr, sr, device)
            rows.append(
                {
                    "lang": cand,
                    "hyp": hyp,
                    "mean_conf": mean_c,
                    "ln_conf": ln,
                    "path_conf": path_m,
                    "lp_conf": lp,
                    "n_frames": nf,
                    "n_nonblank": nnb,
                }
            )
        all_cands[uid] = rows

    def pick(score_key: str, rover_margin: float | None = None):
        hyps = {}
        decode_lang = {}
        for uid, rows in all_cands.items():
            ordered = sorted(rows, key=lambda r: r[score_key], reverse=True)
            best = ordered[0]
            if rover_margin is not None and len(ordered) >= 2:
                # ROVER among hyps within margin of best score
                pool = [ordered[0]["hyp"]]
                for r in ordered[1:]:
                    if best[score_key] - r[score_key] <= rover_margin:
                        pool.append(r["hyp"])
                if len(pool) >= 2:
                    hyps[uid] = rover_vote(pool)
                    decode_lang[uid] = best["lang"]  # use best for specialist upgrades
                else:
                    hyps[uid] = best["hyp"]
                    decode_lang[uid] = best["lang"]
            else:
                hyps[uid] = best["hyp"]
                decode_lang[uid] = best["lang"]
        return hyps, decode_lang

    # specialist upgrades
    ft_p = AutoProcessor.from_pretrained(str(CHECKPOINT_DIR / "mms-lug-ft-v2"), local_files_only=True)
    ft_m = Wav2Vec2ForCTC.from_pretrained(str(CHECKPOINT_DIR / "mms-lug-ft-v2"), local_files_only=True)
    fix_mms_tokenizer(ft_p, "lug")
    ft_m.to(device).eval()
    ach_m, ach_p = get("ach")
    nyn_m, nyn_p = get("nyn")
    dec_ach = mk_dec(ach_p, "ach")
    dec_nyn = mk_dec(nyn_p, "nyn")

    def upgrade(hyps, dlang):
        out = {}
        for uid in ids:
            arr, sr = audio[uid]
            d = dlang[uid]
            if d == "ach":
                g = normalize_text(transcribe_waveform(ach_m, ach_p, arr, sr, device=device)) or "."
                out[uid] = guard(g, beam(ach_m, ach_p, dec_ach, arr, sr, device))
            elif d == "lug":
                out[uid] = normalize_text(transcribe_waveform(ft_m, ft_p, arr, sr, device=device)) or "."
            elif d == "nyn":
                g = normalize_text(transcribe_waveform(nyn_m, nyn_p, arr, sr, device=device)) or "."
                out[uid] = guard(g, beam(nyn_m, nyn_p, dec_nyn, arr, sr, device))
            else:
                out[uid] = hyps[uid]
        return out

    refs_l = [refs[i] for i in ids]
    base_h, base_d = pick("mean_conf")
    base = zindi(refs_l, [base_h[i] for i in ids])

    results = {"baseline_mean_conf_multihyp": base, "recipes": {}}

    for name, score_key, rover_m in [
        ("mean_conf", "mean_conf", None),
        ("ln_conf", "ln_conf", None),
        ("path_conf", "path_conf", None),
        ("lp_conf", "lp_conf", None),
        ("mean_conf_rover0.05", "mean_conf", 0.05),
        ("ln_conf_rover0.05", "ln_conf", 0.05),
        ("path_conf_rover0.02", "path_conf", 0.02),
        ("lp_conf_rover0.02", "lp_conf", 0.02),
    ]:
        h, d = pick(score_key, rover_m)
        m = zindi(refs_l, [h[i] for i in ids])
        m["delta_vs_baseline"] = m["zindi_est"] - base["zindi_est"]
        # with upgrades
        hu = upgrade(h, d)
        mu = zindi(refs_l, [hu[i] for i in ids])
        mu["delta_vs_baseline"] = mu["zindi_est"] - base["zindi_est"]
        mu["gate_pass"] = mu["delta_vs_baseline"] >= 0.01
        results["recipes"][name] = {"raw": m, "upgraded": mu, "decode_mass": Counter(d.values())}
        print(name, "raw", m["zindi_est"], "upg", mu["zindi_est"], "Δ", mu["delta_vs_baseline"])

    # best upgraded
    best = max(results["recipes"].items(), key=lambda kv: kv[1]["upgraded"]["zindi_est"])
    results["best_upgraded"] = {"name": best[0], **best[1]["upgraded"]}
    results["baseline_zindi"] = base["zindi_est"]
    # compare to known selective_beam
    results["prior_selective_beam_zindi"] = 0.6982687904395678

    path = OUTPUT_DIR / "proxy_conf_rover.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print("BEST", results["best_upgraded"])
    print("wrote", path)


if __name__ == "__main__":
    main()
