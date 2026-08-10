#!/usr/bin/env python3
"""Matched validation A/B for nolimitsxl/wav2vec2-large-mms-1b-waxal.

The checkpoint is a joint Lingala/Shona/Luganda CTC model with one shared
vocabulary.  This script evaluates it on deterministic WAXAL validation rows
and compares it against the current production hypotheses when cached proxy
hypotheses are available.  It deliberately writes per-utterance hypotheses so
that later LM/ROVER experiments reuse the expensive 1B forward pass.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.text_norm import normalize_text


def pick_device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prep_audio(example: dict) -> np.ndarray:
    audio = example["audio"]
    arr = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio.get("sampling_rate") or TARGET_SR)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    return arr


def metric(refs: list[str], hyps: list[str]) -> dict[str, float]:
    score = score_pairs(refs, hyps)
    return {
        "wer": float(score["wer"]),
        "cer": float(score["cer"]),
        "zindi": float(1.0 - score["score"]),
    }


@torch.inference_mode()
def decode_one(model, processor, audio: np.ndarray, device: torch.device) -> str:
    batch = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    input_values = batch.input_values.to(device)
    attention_mask = batch.get("attention_mask")
    kwargs = {"input_values": input_values}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(device)
    logits = model(**kwargs).logits
    ids = torch.argmax(logits, dim=-1)[0]
    text = processor.decode(ids, skip_special_tokens=True)
    return normalize_text(text) or "."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "checkpoints" / "nolimitsxl-mms1b-waxal-base",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional PEFT LoRA directory to apply after loading the full checkpoint.",
    )
    parser.add_argument("--langs", nargs="+", default=["lin", "sna", "lug"])
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "goal_2026_08_08" / "nolimitsxl_ab",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    # The repository's tokenizer config adds ``<pad>``/``<unk>`` after the
    # explicit CTC vocabulary.  The acoustic head was trained with [PAD]=49 as
    # the CTC blank; leaving AutoProcessor's appended <pad>=56 active emits the
    # literal word "pad" between nearly every character.
    processor.tokenizer.pad_token = "[PAD]"
    processor.tokenizer.unk_token = "[UNK]"
    processor.tokenizer.word_delimiter_token = "|"
    model = Wav2Vec2ForCTC.from_pretrained(
        str(args.model), local_files_only=True, low_cpu_mem_usage=True
    )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(args.adapter), local_files_only=True, is_trainable=False
        )
    model = model.to(device).eval()

    report: dict = {
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "device": str(device),
        "seed": args.seed,
        "n_per_language": args.n,
        "languages": {},
    }

    for lang in args.langs:
        dataset = load_hf_asr_split(lang, "validation")
        indices = list(range(len(dataset)))
        random.Random(args.seed).shuffle(indices)
        indices = indices[: args.n]

        rows: list[dict] = []
        started = time.time()
        for position, index in enumerate(indices, start=1):
            example = dataset[index]
            ref = normalize_text(example.get("transcription") or "") or "."
            hyp = decode_one(model, processor, prep_audio(example), device)
            rows.append(
                {
                    "ID": str(example.get("id") or example.get("ID") or index),
                    "language": lang,
                    "reference": ref,
                    "hypothesis": hyp,
                }
            )
            if position % 10 == 0 or position == len(indices):
                elapsed = time.time() - started
                print(f"{lang}: {position}/{len(indices)} ({elapsed / position:.2f}s/utt)", flush=True)

        frame = pd.DataFrame(rows)
        frame.to_csv(args.out_dir / f"{lang}_hyps.csv", index=False)
        lang_metrics = metric(list(frame.reference), list(frame.hypothesis))
        lang_metrics.update(
            {
                "n": len(frame),
                "seconds": time.time() - started,
                "hypotheses": str(args.out_dir / f"{lang}_hyps.csv"),
            }
        )
        report["languages"][lang] = lang_metrics
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(f"{lang}: {json.dumps(lang_metrics)}", flush=True)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
