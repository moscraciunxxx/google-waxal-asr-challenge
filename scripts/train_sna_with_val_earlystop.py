#!/usr/bin/env python3
"""Sna continued FT from waxal-300m with val early-stop (train only labels)."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Wav2Vec2ForCTC, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORBIDDEN_TRAIN_SPLITS, SEED, TARGET_SR
from src.dataset import load_hf_asr_split
from src.legit_fusion import beats_baseline, mean_error
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text

MID = "waxal-benchmarking/mms-300m-waxal-sna"


def collate_fn(processor):
    def _c(batch):
        arrays, texts = [], []
        for ex in batch:
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"]["sampling_rate"])
            if sr != TARGET_SR:
                import librosa

                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
            max_len = TARGET_SR * 12
            if arr.shape[0] > max_len:
                arr = arr[:max_len]
            peak = float(np.max(np.abs(arr)) + 1e-9)
            arrays.append(arr / peak)
            texts.append(normalize_text(ex.get("transcription") or ""))
        batch_in = processor(arrays, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        lab = processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
        lab[lab == processor.tokenizer.pad_token_id] = -100
        batch_in["labels"] = lab
        return batch_in

    return _c


@torch.inference_mode()
def eval_val(model, processor, device, max_n=40):
    model.eval()
    ds = load_hf_asr_split("sna", "validation", max_samples=max_n)
    # baseline frozen
    bproc = AutoProcessor.from_pretrained(MID, local_files_only=True)
    bmodel = Wav2Vec2ForCTC.from_pretrained(MID, local_files_only=True).to(device).eval()
    refs, base_h, own_h = [], [], []
    for i in range(len(ds)):
        ex = ds[i]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        refs.append(normalize_text(ex.get("transcription") or ""))
        base_h.append(transcribe_waveform(bmodel, bproc, arr, sr, device=device))
        own_h.append(transcribe_waveform(model, processor, arr, sr, device=device))
    base = score_pairs(refs, base_h)
    own = score_pairs(refs, own_h)
    return base, own, beats_baseline(own, base)


def main():
    assert "test" in FORBIDDEN_TRAIN_SPLITS
    device = pick_device()
    torch.manual_seed(SEED)
    processor = AutoProcessor.from_pretrained(MID, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(MID, local_files_only=True).to(device)
    train_ds = load_hf_asr_split("sna", "train", max_samples=500)
    rows = [train_ds[i] for i in range(len(train_ds))]
    loader = DataLoader(rows, batch_size=1, shuffle=True, collate_fn=collate_fn(processor))
    optim = torch.optim.AdamW(model.parameters(), lr=8e-6)
    best_state = None
    best_me = 1e9
    best_own = None
    best_base = None
    step = 0
    max_steps = 60
    model.train()
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            step += 1
            if step % 10 == 0:
                base, own, beat = eval_val(model, processor, device, max_n=40)
                me = mean_error(own["wer"], own["cer"])
                print(
                    f"step={step} loss={float(loss):.4f} own_me={me:.4f} base_me={mean_error(base['wer'], base['cer']):.4f} beats={beat}",
                    flush=True,
                )
                if me < best_me:
                    best_me = me
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_own, best_base = own, base
                model.train()
            if step >= max_steps:
                break
    out = Path("checkpoints/mms300-sna-earlystop/best")
    out.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    meta = {
        "best_me": best_me,
        "best_own": best_own,
        "best_base": best_base,
        "beats": beats_baseline(best_own, best_base) if best_own else False,
        "baseline_model": MID,
    }
    (out.parent / "meta.json").write_text(json.dumps(meta, indent=2))
    print("FINAL", meta)
    return 0 if meta.get("beats") else 2


if __name__ == "__main__":
    raise SystemExit(main())
