#!/usr/bin/env python3
"""Fetch corrected WAXAL label columns without downloading embedded audio.

Source: Harcuracy/google_waxal_asr_challenge (CC-BY-4.0).  Parquet column
projection keeps this operation small even though the full dataset is 12.2 GB.
Only train/validation metadata and transcripts are written locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "Harcuracy/google_waxal_asr_challenge"
LANGUAGES = ("lin", "sna", "lug")
SPLITS = ("train", "validation")
OUT_DIR = ROOT / "data" / "corrected_waxal"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(lang: str, split: str, out_dir: Path) -> dict:
    fs = HfFileSystem()
    pattern = f"datasets/{REPO_ID}/{lang}_asr/{split}-*.parquet"
    shards = sorted(fs.glob(pattern))
    if not shards:
        raise RuntimeError(f"No shards found for {pattern}")

    frames: list[pd.DataFrame] = []
    columns = ["id", "speaker_id", "transcription", "language", "gender"]
    for position, shard in enumerate(shards, start=1):
        with fs.open(shard, "rb") as handle:
            table = pq.ParquetFile(handle).read(columns=columns)
        frames.append(table.to_pandas())
        print(f"{lang}/{split}: projected {position}/{len(shards)} shards", flush=True)

    frame = pd.concat(frames, ignore_index=True)
    frame["id"] = frame["id"].astype(str)
    if frame.id.duplicated().any():
        raise RuntimeError(f"Duplicate IDs in {lang}/{split}")
    if frame.transcription.isna().any() or frame.transcription.astype(str).str.strip().eq("").any():
        raise RuntimeError(f"Empty corrected transcripts in {lang}/{split}")
    if not frame.language.astype(str).eq(lang).all():
        raise RuntimeError(f"Language mismatch in {lang}/{split}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{lang}_{split}_labels.csv"
    frame.to_csv(out_path, index=False)
    return {
        "language": lang,
        "split": split,
        "rows": len(frame),
        "speakers": int(frame.speaker_id.astype(str).nunique()),
        "shards": len(shards),
        "output": str(out_path),
        "sha256": sha256(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    revision = HfApi().dataset_info(REPO_ID).sha
    reports = [fetch(lang, split, args.out_dir) for lang in args.langs for split in args.splits]
    manifest = {
        "source": REPO_ID,
        "source_revision": revision,
        "license": "cc-by-4.0",
        "contains_test_labels": False,
        "files": reports,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
