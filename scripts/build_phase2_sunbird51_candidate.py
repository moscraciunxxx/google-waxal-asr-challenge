#!/usr/bin/env python3
"""Build a Phase-2 candidate with selected Sunbird-51 route replacements.

The default is intentionally Runyankole-only: on the fixed n=40 proxy the
correctly prompted Sunbird model improved the route score from 0.67557 to
0.81693 (32 wins / 8 losses).  Luganda and Acholi are not promoted by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from src.submission import check_phase2_submission
from src.text_norm import normalize_text

MODEL_ID = "Sunbird/asr-whisper-51-african-languages"
LANGUAGE_TOKEN = {"ach": 50357, "lin": 50353, "lug": 50332, "luo": 50331, "nyn": 50322, "sna": 50324, "sog": 50310}
DEFAULT_BASE = ROOT / "outputs" / "goal_2026_08_07" / "badrex_tiers" / "submission_phase2_badrex_sna_sim99_lug_splitjoin.csv"
DEFAULT_INDEX = ROOT / "outputs" / "beat075" / "public_visible_index.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "goal_2026_08_08" / "sunbird51_phase2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pick_device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if int(sr) != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=int(sr), target_sr=TARGET_SR)
    return np.asarray(audio, dtype=np.float32)


@torch.inference_mode()
def decode_batch(
    model,
    processor,
    paths: list[Path],
    lang: str,
    device: torch.device,
) -> list[str]:
    features = processor(
        [load_wav(path) for path in paths],
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
    texts = processor.batch_decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return [normalize_text(text) or "." for text in texts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--routes", nargs="+", default=["nyn"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only")
    parser.add_argument(
        "--word-ratio-min",
        type=float,
        default=None,
        help="Keep the base when Sunbird/base word count is below this value",
    )
    parser.add_argument(
        "--word-ratio-max",
        type=float,
        default=None,
        help="Keep the base when Sunbird/base word count is above this value",
    )
    args = parser.parse_args()

    unsupported = sorted(set(args.routes) - set(LANGUAGE_TOKEN))
    if unsupported:
        raise ValueError(f"No verified Sunbird token for {unsupported}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.base, dtype={"ID": str})
    route_index = pd.read_csv(args.index, dtype={"ID": str})
    selected = route_index[route_index.decode_lang.isin(args.routes)].copy()
    if args.limit is not None:
        selected = selected.head(args.limit)
    if selected.empty:
        raise RuntimeError(f"No rows selected for routes {args.routes}")

    cache_path = args.out_dir / ("hyps_" + "_".join(args.routes) + ".csv")
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"ID": str})
    else:
        cached = pd.DataFrame(columns=["ID", "route", "prediction"])
    predictions = dict(zip(cached.ID.astype(str), cached.prediction.astype(str)))

    device = pick_device(args.device)
    missing = selected[~selected.ID.isin(predictions)].copy()
    if len(missing):
        processor = WhisperProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID, local_files_only=True, low_cpu_mem_usage=True
        ).to(device).eval()
        started = time.time()
        done = 0
        for route, route_rows in missing.groupby("decode_lang", sort=False):
            records = list(route_rows.itertuples(index=False))
            for start in range(0, len(records), args.batch_size):
                batch = records[start : start + args.batch_size]
                batch_hyps = decode_batch(
                    model,
                    processor,
                    [Path(row.audio) for row in batch],
                    str(route),
                    device,
                )
                for row, hyp in zip(batch, batch_hyps):
                    predictions[str(row.ID)] = hyp
                done += len(batch)
                cache_rows = [
                    {"ID": item.ID, "route": item.decode_lang, "prediction": predictions[str(item.ID)]}
                    for item in selected.itertuples(index=False)
                    if str(item.ID) in predictions
                ]
                pd.DataFrame(cache_rows).to_csv(cache_path, index=False)
                print(
                    f"decoded {done}/{len(missing)} new rows "
                    f"({(time.time() - started) / done:.2f}s/utt)",
                    flush=True,
                )

    candidate = base.copy()
    base_targets = base.set_index("ID").Target.astype(str)
    replacement_ids = set(selected.ID.astype(str))
    guarded_ids: set[str] = set()
    if args.word_ratio_min is not None or args.word_ratio_max is not None:
        for uid in replacement_ids:
            base_words = max(1, len(base_targets[uid].split()))
            ratio = len(predictions[uid].split()) / base_words
            if args.word_ratio_min is not None and ratio < args.word_ratio_min:
                guarded_ids.add(uid)
            if args.word_ratio_max is not None and ratio > args.word_ratio_max:
                guarded_ids.add(uid)
        replacement_ids -= guarded_ids
    candidate.loc[candidate.ID.isin(replacement_ids), "Target"] = candidate.loc[
        candidate.ID.isin(replacement_ids), "ID"
    ].map(predictions)
    route_tag = "_".join(args.routes)
    suffix = f"_lim{args.limit}" if args.limit is not None else ""
    if args.word_ratio_min is not None or args.word_ratio_max is not None:
        low = "none" if args.word_ratio_min is None else str(args.word_ratio_min).replace(".", "p")
        high = "none" if args.word_ratio_max is None else str(args.word_ratio_max).replace(".", "p")
        suffix += f"_wr{low}_{high}"
    out_path = args.out_dir / f"submission_phase2_sim99_sunbird51_{route_tag}{suffix}.csv"
    candidate.to_csv(out_path, index=False)

    base_by_id = base.set_index("ID").Target.astype(str)
    cand_by_id = candidate.set_index("ID").Target.astype(str)
    changed = [uid for uid in candidate.ID if cand_by_id[uid] != base_by_id[uid]]
    validation = check_phase2_submission(out_path, strict=True)
    meta = {
        "output": str(out_path),
        "base": str(args.base),
        "base_sha256": sha256(args.base),
        "output_sha256": sha256(out_path),
        "model": MODEL_ID,
        "routes": args.routes,
        "selected_rows": len(selected),
        "guarded_rows": len(guarded_ids),
        "word_ratio_guard": [args.word_ratio_min, args.word_ratio_max],
        "changed_rows": len(changed),
        "changed_ids_head": changed[:20],
        "validation": validation,
        "offline_gate": {
            "nyn_n": 40,
            "sunbird_zindi": 0.8169339517331501,
            "waxal_same_ids_zindi": 0.6755710839230684,
            "wins_losses_ties": [32, 8, 0],
            "seeded_nyn_n": 120,
            "seed": 42,
            "sunbird_seeded_zindi": 0.7836004378617666,
            "production_ft_beam_zindi": 0.7507117150307621,
            "delta_vs_production": 0.0328887228310045,
            "sunbird_seeded_wer": 0.32950591510090466,
            "production_ft_beam_wer": 0.394919972164231,
            "sunbird_seeded_cer": 0.10329320917556212,
            "production_ft_beam_cer": 0.10365659777424484,
        },
    }
    (args.out_dir / f"meta_{route_tag}{suffix}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if not validation["ok"]:
        raise RuntimeError(f"Candidate validation failed: {validation['errors']}")


if __name__ == "__main__":
    main()
