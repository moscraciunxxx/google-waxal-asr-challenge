#!/usr/bin/env python3
"""Confirm domain beam vs greedy on WAXAL lug val (n=80 seed 42)."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch
from transformers import AutoProcessor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mms_adapter_ft import fix_mms_tokenizer, pick_device
from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.mms_infer import transcribe_waveform
from src.text_norm import normalize_text

def main():
    device = pick_device("mps")
    ckpt = ROOT / "checkpoints" / "mms-lug-ft-v3"
    arpa = ROOT / "data" / "lms_phase2_domain" / "lug_merged_2gram.arpa"
    proc = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
    fix_mms_tokenizer(proc, "lug")
    model.to(device).eval()
    vocab = proc.tokenizer.get_vocab()
    id2 = {i: t for t, i in vocab.items()}
    labels = [id2[i] for i in range(len(id2))]
    uni = (ROOT / "data" / "lms" / "lug_unigrams.txt").read_text().splitlines()
    dec = build_ctcdecoder(labels, kenlm_model_path=str(arpa), unigrams=[u for u in uni if u.strip()], alpha=0.3, beta=0.5)
    ds = load_hf_asr_split("lug", "validation", max_samples=None)
    rng = np.random.default_rng(42)
    idxs = list(range(len(ds))); rng.shuffle(idxs); idxs = idxs[:80]
    refs, g_hyps, b_hyps = [], [], []
    for k, i in enumerate(idxs):
        ex = ds[int(i)]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        refs.append(normalize_text(str(ex.get("transcription") or "")))
        g_hyps.append(normalize_text(transcribe_waveform(model, proc, arr, sr, device=device)) or ".")
        if sr != TARGET_SR:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        peak = float(np.max(np.abs(arr)) + 1e-9)
        arr = arr / peak
        inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
        text = normalize_text(dec.decode(logits, beam_width=100).replace("|", " ")) or "."
        b_hyps.append(text)
        if (k+1)%20==0: print(k+1)
    def pack(h):
        s = score_pairs(refs, h)
        return {"wer": s["wer"], "cer": s["cer"], "zindi": 1-s["score"], "n": int(s["n"])}
    out = {"greedy": pack(g_hyps), "domain_beam": pack(b_hyps)}
    out["delta"] = out["domain_beam"]["zindi"] - out["greedy"]["zindi"]
    Path("outputs/beat075/val_beam_confirm.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
