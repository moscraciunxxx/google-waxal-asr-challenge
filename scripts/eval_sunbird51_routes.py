#!/usr/bin/env python3
"""Evaluate the cached Sunbird 51-language Whisper model on route proxies.

Sunbird reuses otherwise unrelated Whisper language-token IDs for several
African languages.  ``get_decoder_prompt_ids(language=...)`` therefore gives
the wrong prompt for Acholi, Luganda, Runyankole and Lusoga.  The explicit IDs
below are copied from the model card and are part of the checkpoint protocol.
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
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.text_norm import normalize_text

MODEL_ID = "Sunbird/asr-whisper-51-african-languages"
PROXY = ROOT / "data" / "proxy_val_index.csv"

# https://huggingface.co/Sunbird/asr-whisper-51-african-languages#usage
LANGUAGE_TOKEN = {
    "ach": 50357,
    "lin": 50353,
    "lug": 50332,
    "luo": 50331,
    "nyn": 50322,
    "sna": 50324,
    "sog": 50310,  # model-card language code xog (Lusoga)
}


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


@torch.inference_mode()
def decode_one(model, processor, audio: np.ndarray, lang: str, device: torch.device) -> str:
    features = processor(
        audio,
        sampling_rate=TARGET_SR,
        do_normalize=True,
        return_tensors="pt",
    ).input_features.to(device)
    forced = [
        (1, LANGUAGE_TOKEN[lang]),
        (2, processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")),
        (3, processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")),
    ]
    ids = model.generate(
        features,
        forced_decoder_ids=forced,
        num_beams=1,
        do_sample=False,
        max_new_tokens=256,
    )
    text = processor.batch_decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return normalize_text(text) or "."


def proxy_examples(lang: str, limit: int | None) -> list[dict]:
    proxy = pd.read_csv(PROXY)
    wanted = proxy.loc[proxy.language == lang, "id"].astype(str).tolist()
    if limit is not None:
        wanted = wanted[:limit]
    wanted_set = set(wanted)
    dataset = load_hf_asr_split(lang, "validation")
    by_id = {}
    for index, uid in enumerate(dataset["id"]):
        uid = str(uid)
        if uid in wanted_set:
            by_id[uid] = index
    missing = [uid for uid in wanted if uid not in by_id]
    if missing:
        raise RuntimeError(f"{lang}: missing proxy IDs: {missing[:5]}")
    return [dataset[by_id[uid]] for uid in wanted]


def seeded_examples(lang: str, limit: int | None, seed: int) -> list[dict]:
    """Match the seed-42 shuffled validation protocol used by route A/B gates."""
    dataset = load_hf_asr_split(lang, "validation")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    if limit is not None:
        indices = indices[:limit]
    return [dataset[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["ach", "nyn", "lug"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-mode", choices=["proxy", "seeded"], default="proxy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "goal_2026_08_08" / "sunbird51_routes",
    )
    args = parser.parse_args()

    unsupported = sorted(set(args.langs) - set(LANGUAGE_TOKEN))
    if unsupported:
        raise ValueError(f"No verified Sunbird token mapping for {unsupported}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    processor = WhisperProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()

    report = {
        "model": MODEL_ID,
        "device": str(device),
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "languages": {},
    }
    for lang in args.langs:
        if args.sample_mode == "seeded":
            examples = seeded_examples(lang, args.limit, args.seed)
        else:
            examples = proxy_examples(lang, args.limit)
        rows = []
        started = time.time()
        for position, example in enumerate(examples, start=1):
            uid = str(example.get("id") or example.get("ID"))
            ref = normalize_text(example.get("transcription") or "") or "."
            hyp = decode_one(model, processor, prep_audio(example), lang, device)
            rows.append({"ID": uid, "language": lang, "reference": ref, "hypothesis": hyp})
            if position % 10 == 0 or position == len(examples):
                print(f"{lang}: {position}/{len(examples)}", flush=True)

        frame = pd.DataFrame(rows)
        out_csv = args.out_dir / f"{lang}_hyps.csv"
        frame.to_csv(out_csv, index=False)
        score = score_pairs(list(frame.reference), list(frame.hypothesis))
        metrics = {
            "n": len(frame),
            "wer": float(score["wer"]),
            "cer": float(score["cer"]),
            "zindi": float(1.0 - score["score"]),
            "seconds": time.time() - started,
            "hypotheses": str(out_csv),
        }
        report["languages"][lang] = metrics
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(f"{lang}: {json.dumps(metrics)}", flush=True)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
