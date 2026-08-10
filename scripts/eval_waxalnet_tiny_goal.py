#!/usr/bin/env python3
"""Screen a public WAXALNet Whisper-Tiny checkpoint on locked Luganda validation.

This intentionally uses the existing seed-42 validation protocol and never
opens Phase-2 labels or prediction files.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_luganda_public_scout import decode_model, load_sample, metric  # noqa: E402
from src.config import TARGET_SR  # noqa: E402
from src.text_norm import normalize_text  # noqa: E402


@torch.inference_mode()
def decode(model_dir: Path, rows: list[dict], device: torch.device) -> list[str]:
    processor = WhisperProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    output: list[str] = []
    for row in rows:
        features = processor(
            row["audio"], sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(device)
        tokens = model.generate(features, do_sample=False, num_beams=1, max_new_tokens=256)
        output.append(
            normalize_text(
                processor.batch_decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            )
            or "."
        )
    model.to("cpu")
    del model, processor
    gc.collect()
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = load_sample()
    frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "audio"} for row in rows]
    )
    device = torch.device("cpu")
    candidate = decode(args.model, rows, device)
    baseline = decode_model(ROOT / "checkpoints/mms-lug-ft-v3", rows, device, incumbent=True)
    frame["candidate"] = candidate
    refs = frame["reference"].tolist()
    frame["baseline"] = baseline
    joined = frame
    report = {
        "protocol": {
            "dataset": "google/WaxalNLP Luganda validation",
            "seed": 42,
            "sample_size": len(rows),
            "screen_rows": 10,
            "holdout_rows": 10,
            "test_labels_used": False,
            "phase2_predictions_used": False,
        },
        "model": str(args.model.resolve()),
        "metrics": {
            "baseline_all": metric(refs, baseline),
            "candidate_all": metric(refs, candidate),
            "baseline_screen": metric(refs[:10], baseline[:10]),
            "candidate_screen": metric(refs[:10], candidate[:10]),
            "baseline_holdout": metric(refs[10:], baseline[10:]),
            "candidate_holdout": metric(refs[10:], candidate[10:]),
        },
        "candidate_minus_baseline": {
            "all": metric(refs, candidate)["zindi"] - metric(refs, baseline)["zindi"],
            "screen": metric(refs[:10], candidate[:10])["zindi"]
            - metric(refs[:10], baseline[:10])["zindi"],
            "holdout": metric(refs[10:], candidate[10:])["zindi"]
            - metric(refs[10:], baseline[10:])["zindi"],
        },
        "id_sha256": hashlib.sha256("\n".join(frame["ID"].tolist()).encode()).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.out.with_suffix(".csv"), index=False)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
