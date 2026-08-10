#!/usr/bin/env python3
"""Floor-first prize candidate: LID=lug high-p1 + decode=nyn → mms-lug-ft-v3.

Defaults to floor Targets; only replaces a documented high-precision set.
Never overwrites submission_phase2_FINAL.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORBIDDEN_TRAIN_SPLITS
from src.mms_infer import pick_device, transcribe_waveform
from src.prize_pack import align_ids, apply_replace_set, length_guard_ok
from src.text_norm import normalize_text


def load_wav(path: Path):
    import soundfile as sf

    try:
        arr, sr = sf.read(str(path), dtype="float32")
    except Exception:
        import librosa

        arr, sr = librosa.load(str(path), sr=None, mono=True)
        arr = arr.astype(np.float32)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=1)
    return arr, int(sr)


def main() -> int:
    assert "test" in FORBIDDEN_TRAIN_SPLITS
    p = argparse.ArgumentParser()
    p.add_argument("--thr", type=float, default=0.995)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/prize_path/submission_phase2_prize_candidate.csv")
    args = p.parse_args()

    floor_path = ROOT / "submission_phase2_FINAL.csv"
    sample = ROOT / "data/phase2/SampleSubmission.csv"
    lid_path = ROOT / "outputs/phase2_lid126_full.csv"
    openset_path = ROOT / "outputs/phase2_openset_detail.csv"
    audio = ROOT / "data/phase2/audio"
    ckpt = ROOT / "checkpoints/mms-lug-ft-v3"

    floor = {r["ID"]: r["Target"].strip() for r in csv.DictReader(floor_path.open())}
    ids = [r["ID"] for r in csv.DictReader(sample.open())]
    lid = {r["ID"]: r for r in csv.DictReader(lid_path.open())}
    openset = {r["ID"]: r for r in csv.DictReader(openset_path.open())}

    device = pick_device()
    proc = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True).to(device).eval()

    replaces = []
    for sid in ids:
        L = lid[sid]
        if (L.get("lang1") or "").lower() != "lug":
            continue
        p1 = float(L.get("p1") or 0)
        if p1 < args.thr:
            continue
        if (openset[sid].get("decode_lang") or "").lower() != "nyn":
            continue
        arr, sr = load_wav(audio / f"{sid}.wav")
        hyp = normalize_text(transcribe_waveform(model, proc, arr, sr, device=device)) or "."
        old = floor[sid]
        if hyp == old or not length_guard_ok(old, hyp):
            continue
        replaces.append(
            {
                "ID": sid,
                "p1": p1,
                "own_hyp": hyp,
                "floor_hyp": old,
                "reason": f"lid_lug_p1>={args.thr}_decode_nyn_to_mms_lug_ft_v3",
            }
        )

    targets = apply_replace_set(floor, replaces)
    rows = align_ids(ids, targets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "Target"])
        w.writeheader()
        w.writerows(rows)

    meta = {
        "out": str(args.out),
        "thr": args.thr,
        "n_replace": len(replaces),
        "floor_sha256": hashlib.sha256(floor_path.read_bytes()).hexdigest(),
        "method": "floor_first LID=lug vs decode=nyn → mms-lug-ft-v3",
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
