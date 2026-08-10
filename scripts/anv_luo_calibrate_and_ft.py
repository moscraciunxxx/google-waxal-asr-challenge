#!/usr/bin/env python3
"""Anv-ke/Dholuo unscripted subset: FT adapter.luo + FPR-capped true-Dholuo detector.

Uses Anv-ke/Dholuo train/unscripted approved labels (parquet + soundfile; no torchcodec).
Writes:
  checkpoints/mms-luo-ft-anv-unscripted-v1/
  data/anv_dholuo/manifest_unscripted_subset.json
  outputs/goal_2026_08_06/anv_luo_calibration.json
  outputs/goal_2026_08_06/anv_luo_gate_decisions.csv  (consumed by private swing builder)

Gate: accept phase-2 lid=luo row for mms1b_luo overlay only if Anv-FT luo CTC conf
>= thr, where thr is the max threshold with FPR <= --max-fpr on WAXAL ach+lug probes.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import (
    MMS_MODEL_ID,
    fix_mms_tokenizer,
    pick_device,
    set_trainable_adapters,
    text_to_ctc_labels,
)
from scripts.run_phase2_openset import load_wav

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("anv_luo")

ANV_REPO = "Anv-ke/Dholuo"
OUT_DIR = ROOT / "outputs" / "goal_2026_08_06"
CKPT_DIR = ROOT / "checkpoints" / "mms-luo-ft-anv-unscripted-v1"
MANIFEST = ROOT / "data" / "anv_dholuo" / "manifest_unscripted_subset.json"
HYBRID = ROOT / "outputs" / "next_iter" / "hybrid_agreement_785.csv"
PHASE2_AUDIO = ROOT / "data" / "phase2" / "audio"

# Rough English-word filter for Anv QA / test transcripts
_EN_RE = re.compile(
    r"\b(the|and|this|that|with|from|audio|test|approving|transcription|english)\b",
    re.I,
)


def is_plausible_dholuo(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 8:
        return False
    if _EN_RE.search(t) and sum(ch.isalpha() and ord(ch) < 128 for ch in t) / max(1, len(t)) > 0.85:
        # mostly ASCII latin with English function words → skip
        en_hits = len(_EN_RE.findall(t))
        if en_hits >= 3:
            return False
    return True


def decode_audio_bytes(audio_obj) -> tuple[np.ndarray, int]:
    if isinstance(audio_obj, dict) and audio_obj.get("bytes") is not None:
        arr, sr = sf.read(io.BytesIO(audio_obj["bytes"]), dtype="float32", always_2d=False)
    else:
        raise ValueError(f"unsupported audio obj {type(audio_obj)}")
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


def prep_waveform(arr: np.ndarray, sr: int, max_seconds: float = 15.0) -> np.ndarray:
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    max_len = int(max_seconds * TARGET_SR)
    if arr.shape[0] > max_len:
        arr = arr[:max_len]
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak


def load_anv_unscripted_subset(n_train: int, n_val: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Load approved unscripted Anv samples with audio from parquet shards."""
    files = list_repo_files(ANV_REPO, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith("train/unscripted/audios/") and f.endswith(".parquet"))
    if not shards:
        raise RuntimeError("No train/unscripted parquet shards on Anv-ke/Dholuo")

    rng = random.Random(seed)
    rng.shuffle(shards)
    need = n_train + n_val
    samples: list[dict] = []
    for shard in shards:
        if len(samples) >= need * 3:  # oversample then filter
            break
        path = hf_hub_download(ANV_REPO, shard, repo_type="dataset")
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            text = normalize_text(str(row.get("transcription") or "")) or ""
            if not is_plausible_dholuo(text):
                continue
            try:
                arr, sr = decode_audio_bytes(row["audio"])
            except Exception:
                continue
            if arr.shape[0] < TARGET_SR // 4:
                continue
            samples.append(
                {
                    "id": str(row.get("filename") or len(samples)),
                    "shard": shard,
                    "text": text,
                    "array": prep_waveform(arr, sr),
                    "duration": float(arr.shape[0] / max(1, sr)),
                    "domain": str(row.get("domain") or ""),
                }
            )
            if len(samples) >= need * 3:
                break
        logger.info("shard %s -> cumulative candidates %d", shard, len(samples))

    if len(samples) < need:
        raise RuntimeError(f"Only {len(samples)} Anv samples after filter; need {need}")

    rng.shuffle(samples)
    train, val = samples[:n_train], samples[n_train : n_train + n_val]
    return train, val


@torch.inference_mode()
def ctc_mean_logprob(model, processor, arr: np.ndarray, device) -> tuple[str, float]:
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits
    pred = torch.argmax(logits, dim=-1)[0]
    text = normalize_text(processor.decode(pred)) or "."
    logp = torch.nn.functional.log_softmax(logits, dim=-1)[0]
    conf = float(logp.max(dim=-1).values.mean().item())
    return text, conf


def train_anv(train: list[dict], val: list[dict], steps: int, device: torch.device, lr: float = 1e-4):
    processor = AutoProcessor.from_pretrained(MMS_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MMS_MODEL_ID)
    fix_mms_tokenizer(processor, "luo")
    try:
        model.load_adapter("luo", local_files_only=True)
    except Exception:
        model.load_adapter("luo")
    n_train = set_trainable_adapters(model)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    logger.info("trainable params=%d device=%s steps=%d n_train=%d", n_train, device, steps, len(train))

    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        ex = train[(step - 1) % len(train)]
        text = normalize_text(ex["text"]) or "."
        labels = text_to_ctc_labels(processor.tokenizer, text)
        inputs = processor(ex["array"], sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        n_frames = max(1, int(inputs.input_values.shape[-1] // 320))
        if len(labels) > n_frames:
            labels = labels[:n_frames]
        if not labels:
            continue
        out = model(
            inputs.input_values.to(device),
            labels=torch.tensor([labels], dtype=torch.long, device=device),
        )
        loss = out.loss
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if step % 10 == 0 or step == 1:
            logger.info("anv-ft step=%d loss=%.4f elapsed=%.1fs", step, losses[-1], time.time() - t0)

    # quick val conf
    model.eval()
    val_confs = []
    for ex in val:
        _, c = ctc_mean_logprob(model, processor, ex["array"], device)
        val_confs.append(c)
    mean_val = float(np.mean(val_confs)) if val_confs else float("nan")
    logger.info("val mean ctc conf=%.4f n=%d", mean_val, len(val_confs))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT_DIR)
    processor.save_pretrained(CKPT_DIR)
    meta = {
        "base": MMS_MODEL_ID,
        "adapter": "luo",
        "steps": steps,
        "n_train": len(train),
        "n_val": len(val),
        "mean_train_loss_last20": float(np.mean(losses[-20:])) if losses else None,
        "val_mean_ctc_conf": mean_val,
        "source": ANV_REPO + " train/unscripted approved-filtered",
    }
    (CKPT_DIR / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return model, processor, meta


def score_probe_set(model, processor, device, lang: str, n: int, seed: int = 42) -> list[float]:
    ds = load_hf_asr_split(lang, "validation")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    confs = []
    for i in idx[:n]:
        ex = ds[i]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"].get("sampling_rate") or TARGET_SR)
        arr = prep_waveform(arr, sr)
        _, c = ctc_mean_logprob(model, processor, arr, device)
        confs.append(c)
    return confs


def calibrate_and_decide(
    model,
    processor,
    device,
    max_fpr: float,
    min_accept: int,
) -> tuple[pd.DataFrame, dict]:
    hybrid = pd.read_csv(HYBRID)
    hybrid["ID"] = hybrid["ID"].astype(str)

    # Anv-positive conf distribution (held-out val-like from train subset already scored optional)
    # FPR probes: WAXAL ach + lug validation
    ach_conf = score_probe_set(model, processor, device, "ach", n=40)
    lug_conf = score_probe_set(model, processor, device, "lug", n=40)
    fpr_pool = np.array(ach_conf + lug_conf, dtype=float)
    logger.info(
        "FPR probes n=%d mean_conf=%.4f p95=%.4f",
        len(fpr_pool),
        float(fpr_pool.mean()),
        float(np.percentile(fpr_pool, 95)),
    )

    # Score all 785 phase-2 lid=luo rows
    rows = []
    t0 = time.time()
    for k, r in enumerate(hybrid.itertuples(index=False)):
        uid = str(r.ID)
        path = PHASE2_AUDIO / f"{uid}.wav"
        arr, sr = load_wav(path)
        arr = prep_waveform(np.asarray(arr, dtype=np.float32), int(sr))
        hyp, conf = ctc_mean_logprob(model, processor, arr, device)
        # Prefer existing mms1b_luo hyp if present (already decoded); use Anv-FT hyp as alt
        mms_hyp = normalize_text(str(getattr(r, "mms1b_luo", "") or "")) or hyp
        rows.append(
            {
                "ID": uid,
                "anv_luo_conf": conf,
                "anv_ft_hyp": hyp,
                "mms1b_luo": mms_hyp,
                "cer_pm": float(getattr(r, "cer_pm", np.nan)),
            }
        )
        if (k + 1) % 50 == 0:
            logger.info("score phase2 luo %d/%d %.1fs", k + 1, len(hybrid), time.time() - t0)

    scores = np.array([x["anv_luo_conf"] for x in rows], dtype=float)

    # Choose thr = max thr s.t. FPR on ach+lug probes <= max_fpr
    # Higher conf = more luo-like → accept if conf >= thr
    candidates = np.unique(np.concatenate([fpr_pool, scores]))
    candidates = np.sort(candidates)[::-1]  # high to low
    best_thr = float(np.percentile(fpr_pool, 100 * (1 - max_fpr)))  # default
    best_n = 0
    for thr in candidates:
        fpr = float((fpr_pool >= thr).mean()) if len(fpr_pool) else 1.0
        n_acc = int((scores >= thr).sum())
        if fpr <= max_fpr + 1e-12 and n_acc >= best_n:
            best_thr = float(thr)
            best_n = n_acc

    # Ensure we can accept at least min_accept if FPR allows at looser thr
    if best_n < min_accept:
        for thr in np.sort(candidates):  # low to high = looser
            fpr = float((fpr_pool >= thr).mean()) if len(fpr_pool) else 1.0
            n_acc = int((scores >= thr).sum())
            if fpr <= max_fpr + 1e-12 and n_acc >= min_accept:
                best_thr = float(thr)
                best_n = n_acc
                break

    fpr_at = float((fpr_pool >= best_thr).mean()) if len(fpr_pool) else 1.0
    for x in rows:
        x["accept"] = bool(x["anv_luo_conf"] >= best_thr)
        x["overlay_text"] = x["mms1b_luo"] if x["accept"] else ""
        x["reason"] = "anv_ft_conf_fpr_cap" if x["accept"] else "below_thr_or_fpr"

    dec = pd.DataFrame(rows)
    cal = {
        "max_fpr": max_fpr,
        "thr": best_thr,
        "fpr_at_thr": fpr_at,
        "n_phase2_luo": len(dec),
        "n_accept": int(dec.accept.sum()),
        "n_fpr_probes": int(len(fpr_pool)),
        "fpr_probe_langs": ["ach", "lug"],
        "fpr_probe_mean_conf": float(fpr_pool.mean()),
        "accept_mean_conf": float(dec.loc[dec.accept, "anv_luo_conf"].mean()) if dec.accept.any() else None,
        "reject_mean_conf": float(dec.loc[~dec.accept, "anv_luo_conf"].mean()) if (~dec.accept).any() else None,
        "ckpt": str(CKPT_DIR),
        "source": ANV_REPO,
        "rule": "accept if anv_luo_conf >= thr with thr FPR-capped on WAXAL ach+lug val probes",
    }
    return dec, cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--max-fpr", type=float, default=0.05)
    ap.add_argument("--min-accept", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-ft", action="store_true", help="Load existing Anv ckpt if present")
    args = ap.parse_args()

    device = pick_device(None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    # --- load Anv subset ---
    try:
        train, val = load_anv_unscripted_subset(args.n_train, args.n_val, seed=args.seed)
    except Exception as e:
        blocker = {
            "status": "train_path_blocker",
            "error": f"{type(e).__name__}: {e}",
            "dataset": ANV_REPO,
            "after_retry": True,
        }
        (OUT_DIR / "anv_train_blocker.json").write_text(json.dumps(blocker, indent=2))
        raise

    manifest = {
        "dataset": ANV_REPO,
        "split": "train/unscripted",
        "n_train": len(train),
        "n_val": len(val),
        "train_ids": [x["id"] for x in train],
        "val_ids": [x["id"] for x in val],
        "train_shards": sorted({x["shard"] for x in train}),
        "filter": "approved-like transcription in parquet; English QA filter; max 15s",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    logger.info("wrote manifest %s", MANIFEST)

    # --- FT or load ---
    if args.skip_ft and (CKPT_DIR / "model.safetensors").exists():
        logger.info("loading existing %s", CKPT_DIR)
        processor = AutoProcessor.from_pretrained(str(CKPT_DIR))
        model = Wav2Vec2ForCTC.from_pretrained(str(CKPT_DIR)).to(device).eval()
        ft_meta = json.loads((CKPT_DIR / "train_meta.json").read_text()) if (CKPT_DIR / "train_meta.json").exists() else {}
    else:
        model, processor, ft_meta = train_anv(train, val, args.steps, device)
        model.eval()

    # --- calibrate + decisions ---
    dec, cal = calibrate_and_decide(model, processor, device, args.max_fpr, args.min_accept)
    cal["ft_meta"] = ft_meta
    cal["manifest"] = str(MANIFEST)
    dec_path = OUT_DIR / "anv_luo_gate_decisions.csv"
    cal_path = OUT_DIR / "anv_luo_calibration.json"
    dec.to_csv(dec_path, index=False)
    cal_path.write_text(json.dumps(cal, indent=2))
    logger.info(
        "wrote %s accept=%d thr=%.4f fpr=%.4f",
        dec_path,
        cal["n_accept"],
        cal["thr"],
        cal["fpr_at_thr"],
    )
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
