#!/usr/bin/env python3
"""Fine-tune MMS adapter mixing HF train + Phase-2 high-conf pseudo labels.

Never uses Phase-1 test gold. Phase-2 has no gold; pseudo labels are model hyps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, PROJECT_ROOT, TARGET_SR
from src.dataset import load_hf_asr_split
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import (
    fix_mms_tokenizer,
    pick_device,
    set_trainable_adapters,
    text_to_ctc_labels,
    eval_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ft_mix_pseudo")

MMS_MODEL_ID = "facebook/mms-1b-all"


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    peak = float(np.max(np.abs(arr)) + 1e-9)
    max_sec = 20.0
    max_n = int(TARGET_SR * max_sec)
    if arr.shape[0] > max_n:
        arr = arr[:max_n]
    return arr / peak, int(sr)


def collate_item(processor, tok, array, text, device):
    labels = text_to_ctc_labels(tok, text)
    if not labels:
        labels = [tok.pad_token_id or 0]
    inputs = processor(array, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    labels_t = torch.tensor([labels], dtype=torch.long, device=device)
    return input_values, labels_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lang",
        default="lug",
        choices=["lug", "ach", "lin", "sna", "nyn"],
        help="Language adapter / WAXAL split + pseudo filter (decode_lang)",
    )
    ap.add_argument("--pseudo-csv", type=Path, default=PROJECT_ROOT / "data" / "phase2_pseudo_index.csv")
    ap.add_argument("--max-hf-train", type=int, default=4000)
    ap.add_argument("--max-pseudo", type=int, default=800)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--out-suffix", default="ft-pseudo-v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pseudo-weight", type=float, default=0.5, help="sample prob for pseudo vs HF")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    logger.info("device %s lang=%s", device, args.lang)

    # model
    if args.init_from and Path(args.init_from).exists():
        logger.info("warm-start %s", args.init_from)
        processor = AutoProcessor.from_pretrained(str(args.init_from), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(args.init_from), local_files_only=True)
    else:
        processor = AutoProcessor.from_pretrained(MMS_MODEL_ID)
        model = Wav2Vec2ForCTC.from_pretrained(MMS_MODEL_ID)
        try:
            model.load_adapter(args.lang)
            model.set_adapter(args.lang)
        except Exception as e:
            logger.warning("adapter load: %s", e)
    fix_mms_tokenizer(processor, args.lang)
    n = set_trainable_adapters(model)
    model.to(device)
    logger.info("trainable %.3fM", n / 1e6)

    # HF train pool
    hf_ds = load_hf_asr_split(args.lang, "train", max_samples=args.max_hf_train)
    hf_pool = []
    for i in range(len(hf_ds)):
        row = hf_ds[i]
        text = normalize_text(str(row.get("transcription") or row.get("text") or ""))
        if not text or len(text.split()) < 2:
            continue
        a = row["audio"]
        arr = np.asarray(a["array"], dtype=np.float32)
        sr = int(a["sampling_rate"])
        if sr != TARGET_SR:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        peak = float(np.max(np.abs(arr)) + 1e-9)
        arr = arr / peak
        max_n = int(TARGET_SR * 20.0)
        if arr.shape[0] > max_n:
            arr = arr[:max_n]
        hf_pool.append((arr, text))
    logger.info("hf_pool %d", len(hf_pool))

    # pseudo pool
    pseudo = pd.read_csv(args.pseudo_csv)
    pseudo = pseudo[pseudo.decode_lang == args.lang].head(args.max_pseudo)
    ps_pool = []
    for _, r in pseudo.iterrows():
        path = PROJECT_ROOT / str(r.audio)
        if not path.exists():
            path = PROJECT_ROOT / "data" / "phase2" / "audio" / f"{r.ID}.wav"
        if not path.exists():
            continue
        arr, sr = load_wav(path)
        text = normalize_text(str(r.text)) or ""
        if len(text.split()) < 3:
            continue
        ps_pool.append((arr, text))
    logger.info("pseudo_pool %d", len(ps_pool))
    if not hf_pool:
        raise SystemExit("empty hf pool")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    t0 = time.time()
    losses = []
    tok = processor.tokenizer
    for step in range(1, args.steps + 1):
        use_ps = ps_pool and random.random() < args.pseudo_weight
        arr, text = random.choice(ps_pool if use_ps else hf_pool)
        try:
            iv, lab = collate_item(processor, tok, arr, text, device)
            # CTC length
            out = model(iv).logits
            log_probs = out.log_softmax(-1).transpose(0, 1)
            input_lengths = torch.full((1,), log_probs.size(0), dtype=torch.long, device=device)
            target_lengths = torch.tensor([lab.size(1)], dtype=torch.long, device=device)
            loss = torch.nn.functional.ctc_loss(
                log_probs,
                lab,
                input_lengths,
                target_lengths,
                blank=model.config.pad_token_id if model.config.pad_token_id is not None else 0,
                zero_infinity=True,
            )
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            del out, log_probs, loss, iv, lab
            if device.type == "mps":
                torch.mps.empty_cache()
        except Exception as e:
            logger.warning("step %d skip: %s", step, e)
            continue
        if step % 50 == 0 or step == 1:
            avg = float(np.mean(losses[-50:])) if losses else 0
            logger.info(
                "step %d/%d loss=%.4f avg50=%.4f elapsed=%.1fs",
                step,
                args.steps,
                losses[-1] if losses else -1,
                avg,
                time.time() - t0,
            )

    # val on HF validation only (never test)
    val_ds = load_hf_asr_split(args.lang, "validation", max_samples=100)
    vloss = eval_loss(model, processor, val_ds, device, max_n=min(32, len(val_ds)))
    logger.info("val_loss=%.4f", vloss)

    out_dir = CHECKPOINT_DIR / f"mms-{args.lang}-{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    try:
        torch.save(model._get_adapters(), out_dir / "adapter_layers.pt")
    except Exception:
        pass
    meta = {
        "lang": args.lang,
        "steps": args.steps,
        "lr": args.lr,
        "hf_pool": len(hf_pool),
        "pseudo_pool": len(ps_pool),
        "pseudo_weight": args.pseudo_weight,
        "val_loss": vloss,
        "avg_last50": float(np.mean(losses[-50:])) if losses else None,
        "init_from": str(args.init_from) if args.init_from else None,
        "rule": "HF train + Phase-2 pseudo only; never Phase-1 test gold",
        "output": str(out_dir),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved %s", out_dir)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
