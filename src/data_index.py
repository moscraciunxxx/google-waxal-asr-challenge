"""Build train/val/test indices from Hugging Face WAXAL without using test gold for training.

Rules enforced here:
  - Phase-1 test transcriptions are NEVER written into Train.csv or any train split.
  - Test.csv / SampleSubmission.csv contain IDs (and empty Target for sample) only.
  - Validation gold is allowed for local eval / early stopping (not the Phase-1 test set).

Metadata is loaded by projecting non-audio columns from HF parquet shards (columnar
read skips audio bytes in memory). Falls back to the datasets-server rows API when
available.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from src.config import (
    DATA_DIR,
    FORBIDDEN_TRAIN_SPLITS,
    HF_CONFIGS,
    HF_DATASET,
    ID_COL,
    INDEX_CSV,
    LANGUAGES,
    METADATA_CACHE,
    SAMPLE_SUBMISSION_CSV,
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
)

logger = logging.getLogger(__name__)

ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
META_COLS = ["id", "speaker_id", "transcription", "language", "gender"]


def _fetch_rows_page(config: str, split: str, offset: int, length: int = PAGE_SIZE) -> dict:
    params = urlencode(
        {
            "dataset": HF_DATASET,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{ROWS_API}?{params}"
    req = Request(url, headers={"User-Agent": "waxal-asr-solution/1.0"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_split_via_rows_api(lang: str, split: str) -> pd.DataFrame:
    """Fast path when datasets-server works (no multi-GB download)."""
    config = HF_CONFIGS[lang]
    rows: list[dict] = []
    offset = 0
    total = None
    while True:
        payload = _fetch_rows_page(config, split, offset, PAGE_SIZE)
        if total is None:
            total = int(payload.get("num_rows_total") or 0)
        batch = payload.get("rows") or []
        if not batch:
            break
        for item in batch:
            ex = item.get("row") if isinstance(item, dict) and "row" in item else item
            rows.append(
                {
                    ID_COL: ex["id"],
                    "speaker_id": ex.get("speaker_id") or "",
                    TARGET_COL: ex.get("transcription") or "",
                    "language": ex.get("language") or lang,
                    "gender": ex.get("gender") or "",
                    "split": split,
                    "hf_config": config,
                }
            )
        offset += len(batch)
        if total is not None and offset >= total:
            break
        if len(batch) < PAGE_SIZE:
            break
        time.sleep(0.02)
    return pd.DataFrame(rows)


def _list_split_parquets(lang: str, split: str) -> list[str]:
    """List parquet paths under data/ASR/{lang}/ for a given split prefix."""
    api = HfApi()
    prefix = f"data/ASR/{lang}/"
    files = api.list_repo_files(HF_DATASET, repo_type="dataset")
    # e.g. lin-train-00000.parquet, lin-test-00001.parquet
    needle = f"{lang}-{split}-"
    paths = sorted(
        f for f in files if f.startswith(prefix) and needle in f and f.endswith(".parquet")
    )
    if not paths:
        # some repos use different layout
        paths = sorted(
            f
            for f in files
            if f.endswith(".parquet") and f"/{lang}/" in f and f"-{split}-" in f
        )
    return paths


def _load_split_via_streaming_meta(lang: str, split: str) -> pd.DataFrame:
    """Stream rows with audio decode disabled — metadata only, progressive download."""
    from datasets import Audio, load_dataset

    config = HF_CONFIGS[lang]
    logger.info("Streaming metadata for %s/%s (audio decode=False)", config, split)
    ds = load_dataset(HF_DATASET, config, split=split, streaming=True)
    # Avoid torchcodec: do not decode audio for index build
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    rows = []
    for ex in ds:
        rows.append(
            {
                ID_COL: ex["id"],
                "speaker_id": ex.get("speaker_id") or "",
                TARGET_COL: ex.get("transcription") or "",
                "language": ex.get("language") or lang,
                "gender": ex.get("gender") or "",
                "split": split,
                "hf_config": config,
            }
        )
        if len(rows) % 500 == 0:
            logger.info("  ... %d rows", len(rows))
    return pd.DataFrame(rows)


def _load_split_via_parquet(lang: str, split: str) -> pd.DataFrame:
    """Download parquet shards and read only non-audio columns (with retries)."""
    config = HF_CONFIGS[lang]
    paths = _list_split_parquets(lang, split)
    if not paths:
        raise FileNotFoundError(f"No parquet shards for {lang}/{split}")
    logger.info("Reading %d parquet shards for %s/%s (column projection)", len(paths), lang, split)
    frames = []
    for rel in paths:
        local = None
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                local = hf_hub_download(
                    repo_id=HF_DATASET,
                    filename=rel,
                    repo_type="dataset",
                )
                break
            except Exception as e:
                last_err = e
                wait = 3 * (attempt + 1)
                logger.warning("download %s failed (%s); retry in %ss", rel, e, wait)
                time.sleep(wait)
        if local is None:
            raise RuntimeError(f"Failed to download {rel}: {last_err}")
        # Project only metadata columns — audio bytes stay on disk unread
        schema_names = set(pq.read_schema(local).names)
        cols = [c for c in META_COLS if c in schema_names]
        table = pq.read_table(local, columns=cols)
        frames.append(table.to_pandas())
    df = pd.concat(frames, ignore_index=True)
    # Normalize column names to our schema
    rename = {"id": ID_COL, "transcription": TARGET_COL}
    df = df.rename(columns=rename)
    if ID_COL not in df.columns:
        raise RuntimeError(f"Missing id column in {lang}/{split}")
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = ""
    if "language" not in df.columns:
        df["language"] = lang
    if "speaker_id" not in df.columns:
        df["speaker_id"] = ""
    if "gender" not in df.columns:
        df["gender"] = ""
    df["split"] = split
    df["hf_config"] = config
    df["language"] = df["language"].fillna(lang).replace("", lang)
    return df[
        [ID_COL, "speaker_id", TARGET_COL, "language", "gender", "split", "hf_config"]
    ]


def _load_split_metadata(lang: str, split: str) -> pd.DataFrame:
    config = HF_CONFIGS[lang]
    logger.info("Loading metadata %s config=%s split=%s", HF_DATASET, config, split)
    # Prefer rows API (cheap) when healthy
    try:
        df = _load_split_via_rows_api(lang, split)
        if len(df) > 0:
            logger.info("  rows API: %d rows", len(df))
            return df
    except Exception as e:
        logger.warning("  rows API failed (%s); trying streaming meta", e)

    try:
        df = _load_split_via_streaming_meta(lang, split)
        if len(df) > 0:
            logger.info("  streaming meta: %d rows", len(df))
            return df
    except Exception as e:
        logger.warning("  streaming meta failed (%s); falling back to parquet projection", e)

    df = _load_split_via_parquet(lang, split)
    logger.info("  parquet projection: %d rows", len(df))
    return df


def build_index(
    languages: Iterable[str] = LANGUAGES,
    cache_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download HF metadata for train/validation/test and write Zindi-style CSVs.

    Train.csv  = train split only (with Target) — never test gold.
    Val rows   = kept in dataset_index.csv for local evaluation only.
    Test.csv   = test IDs only (no Target column).
    SampleSubmission.csv = test IDs + empty Target placeholder.
    """
    cache_dir = Path(cache_dir or METADATA_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for lang in languages:
        for split in ("train", "validation", "test"):
            cache_path = cache_dir / f"{lang}_{split}.parquet"
            if cache_path.exists() and not force:
                logger.info("Using cached metadata %s", cache_path)
                df = pd.read_parquet(cache_path)
            else:
                df = _load_split_metadata(lang, split)
                df.to_parquet(cache_path, index=False)
            frames.append(df)

    index = pd.concat(frames, ignore_index=True)

    train_df = index[index["split"] == "train"].copy()
    assert not train_df["split"].isin(FORBIDDEN_TRAIN_SPLITS).any()
    test_ids = set(index.loc[index["split"] == "test", ID_COL].astype(str))
    assert train_df[ID_COL].astype(str).isin(test_ids).sum() == 0, "Test IDs leaked into train"

    train_out = train_df[[ID_COL, TARGET_COL, "language"]].copy()
    train_out.to_csv(TRAIN_CSV, index=False)

    test_df = index[index["split"] == "test"].copy()
    test_out = test_df[[ID_COL, "language"]].copy()
    test_out.to_csv(TEST_CSV, index=False)

    sample = test_df[[ID_COL]].copy()
    sample[TARGET_COL] = ""
    sample.to_csv(SAMPLE_SUBMISSION_CSV, index=False)

    index.to_csv(INDEX_CSV, index=False)

    summary = {
        "n_train": int((index["split"] == "train").sum()),
        "n_validation": int((index["split"] == "validation").sum()),
        "n_test": int((index["split"] == "test").sum()),
        "languages": list(languages),
        "train_csv": str(TRAIN_CSV),
        "test_csv": str(TEST_CSV),
        "sample_submission_csv": str(SAMPLE_SUBMISSION_CSV),
        "index_csv": str(INDEX_CSV),
        "rule": "test gold never included in Train.csv / training splits",
        "per_language": {
            lang: {
                split: int(((index["language"] == lang) & (index["split"] == split)).sum())
                for split in ("train", "validation", "test")
            }
            for lang in languages
        },
    }
    (DATA_DIR / "index_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Index summary: %s", summary)
    return index


def load_training_frame(index_path: Path | None = None) -> pd.DataFrame:
    """Return only rows allowed for training/tuning (train + optional validation)."""
    path = Path(index_path or INDEX_CSV)
    df = pd.read_csv(path)
    if "split" not in df.columns:
        raise ValueError(f"{path} missing 'split' column")
    allowed = df[~df["split"].isin(FORBIDDEN_TRAIN_SPLITS)].copy()
    if allowed["split"].eq("test").any():
        raise RuntimeError("Invariant violated: test rows present in training frame")
    return allowed


def load_eval_frame(
    index_path: Path | None = None,
    split: str = "validation",
) -> pd.DataFrame:
    path = Path(index_path or INDEX_CSV)
    df = pd.read_csv(path)
    return df[df["split"] == split].copy()


def assert_no_test_gold_in_training(
    train_csv: Path | None = None, index_path: Path | None = None
) -> None:
    train = pd.read_csv(train_csv or TRAIN_CSV)
    index = pd.read_csv(index_path or INDEX_CSV)
    test_ids = set(index.loc[index["split"] == "test", ID_COL].astype(str))
    train_ids = set(train[ID_COL].astype(str))
    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError(f"Train/test ID overlap ({len(overlap)} ids) — rules violation")
