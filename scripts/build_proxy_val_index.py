#!/usr/bin/env python3
"""Build a small Phase-2-like offline proxy index from HF WaxalNLP validation.

Reads only text metadata columns from cached/local validation parquets
(no full audio materialization). Never uses Phase-1 test gold.

Output: data/proxy_val_index.csv with columns:
  id, language, split, transcription

Default proxy langs (Phase-2 domain / champion multi-hyp):
  ach, nyn, lug, sog, mas  — max 40 validation rows each (SEED=42).

Usage:
  .venv/bin/python scripts/build_proxy_val_index.py
  .venv/bin/python scripts/build_proxy_val_index.py --langs ach nyn lug --max-per-lang 40
  .venv/bin/python scripts/build_proxy_val_index.py --download-missing

Reload audio later via existing patterns:
  from src.dataset import load_hf_asr_split
  ds = load_hf_asr_split("ach", "validation", max_samples=40)
  # or filter by id from this CSV after loading a larger val slice
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, FORBIDDEN_TRAIN_SPLITS, HF_DATASET, SEED  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_proxy_val_index")

# Phase-2 LID mass is ~luo+lug; champion multi-hyp uses ach/lug/sog/nyn.
DEFAULT_PROXY_LANGS = ("ach", "nyn", "lug", "sog", "mas")
DEFAULT_MAX_PER_LANG = 40
META_COLS = ("id", "transcription", "language")


def _hub_snapshot_asr_dir(lang: str) -> Path | None:
    import os

    cache_root = Path(
        os.environ.get("HF_HUB_CACHE") or (Path.home() / ".cache/huggingface/hub")
    )
    snap_root = cache_root / f"datasets--{HF_DATASET.replace('/', '--')}" / "snapshots"
    if not snap_root.is_dir():
        return None
    for snap in sorted(snap_root.iterdir(), reverse=True):
        asr_dir = snap / "data" / "ASR" / lang
        if asr_dir.is_dir():
            return asr_dir
    return None


def list_remote_split_paths(lang: str, split: str) -> list[str]:
    url = (
        f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main/"
        f"data/ASR/{lang}?recursive=false"
    )
    req = Request(url, headers={"User-Agent": "waxal-asr-solution/1.0"})
    with urlopen(req, timeout=60) as resp:
        tree = json.loads(resp.read().decode("utf-8"))
    needle = f"{lang}-{split}-"
    return sorted(
        item["path"]
        for item in tree
        if isinstance(item, dict)
        and str(item.get("path", "")).endswith(".parquet")
        and needle in str(item.get("path", ""))
    )


def resolve_split_parquets(
    lang: str,
    split: str,
    *,
    download_missing: bool,
) -> list[Path]:
    """Prefer hub snapshot cache; optionally download only matching shards."""
    if split in FORBIDDEN_TRAIN_SPLITS:
        raise ValueError(
            f"Refusing split={split!r} for proxy index (forbidden for train/tune; "
            "proxy gate uses validation only)"
        )

    asr_dir = _hub_snapshot_asr_dir(lang)
    local: list[Path] = []
    if asr_dir is not None:
        needle = f"{lang}-{split}-"
        found = sorted(
            p.resolve()
            for p in asr_dir.glob("*.parquet")
            if needle in p.name and p.resolve().is_file()
        )
        if found:
            logger.info("%s/%s: %d cached parquet(s) under %s", lang, split, len(found), asr_dir)
            return found

    if not download_missing:
        raise FileNotFoundError(
            f"No local validation parquet for {lang}/{split}. "
            f"Re-run with --download-missing or pre-cache data/ASR/{lang}/{lang}-{split}-*.parquet"
        )

    from huggingface_hub import hf_hub_download

    paths = list_remote_split_paths(lang, split)
    if not paths:
        raise FileNotFoundError(f"No remote parquet for {lang}/{split}")
    for p in paths:
        lp = hf_hub_download(HF_DATASET, p, repo_type="dataset", local_files_only=False)
        local.append(Path(lp).resolve())
        logger.info("%s/%s: downloaded %s", lang, split, p)
    return local


def read_text_meta(parquet_paths: list[Path]) -> pd.DataFrame:
    """Read id/transcription/language only — does not load audio bytes into memory."""
    frames: list[pd.DataFrame] = []
    for path in parquet_paths:
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in META_COLS if c in schema_names]
        # some local metadata exports use ID/Target
        if "id" not in cols and "ID" in schema_names:
            cols.append("ID")
        if "transcription" not in cols and "Target" in schema_names:
            cols.append("Target")
        if "language" not in cols and "language" in schema_names:
            cols.append("language")
        table = pq.read_table(path, columns=cols)
        df = table.to_pandas()
        if "id" not in df.columns and "ID" in df.columns:
            df = df.rename(columns={"ID": "id"})
        if "transcription" not in df.columns and "Target" in df.columns:
            df = df.rename(columns={"Target": "transcription"})
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for c in ("id", "transcription", "language"):
        if c not in out.columns:
            raise KeyError(f"missing column {c} after read; got {list(out.columns)}")
    return out[["id", "transcription", "language"]]


def sample_proxy(
    df: pd.DataFrame,
    lang: str,
    max_per_lang: int,
    seed: int,
) -> pd.DataFrame:
    sub = df.copy()
    sub["language"] = lang
    sub["split"] = "validation"
    if len(sub) > max_per_lang:
        sub = sub.sample(n=max_per_lang, random_state=seed)
    sub = sub.sort_values("id").reset_index(drop=True)
    return sub[["id", "language", "split", "transcription"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--langs",
        nargs="+",
        default=list(DEFAULT_PROXY_LANGS),
        help="Languages for proxy gate (ISO codes matching WaxalNLP ASR)",
    )
    p.add_argument("--max-per-lang", type=int, default=DEFAULT_MAX_PER_LANG)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--split",
        default="validation",
        help="Labeled split for proxy (must be validation; test forbidden)",
    )
    p.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing split parquet shards from HF hub",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "proxy_val_index.csv",
    )
    args = p.parse_args()

    if args.split in FORBIDDEN_TRAIN_SPLITS:
        raise SystemExit(
            f"Refusing --split={args.split} (FORBIDDEN_TRAIN_SPLITS). Use validation."
        )

    parts: list[pd.DataFrame] = []
    stats: dict[str, dict] = {}
    for lang in args.langs:
        paths = resolve_split_parquets(
            lang, args.split, download_missing=args.download_missing
        )
        meta = read_text_meta(paths)
        n_full = len(meta)
        sampled = sample_proxy(meta, lang, args.max_per_lang, args.seed)
        parts.append(sampled)
        stats[lang] = {
            "n_split_full": n_full,
            "n_proxy": len(sampled),
            "parquet_files": [str(x) for x in paths],
        }
        logger.info(
            "%s: full_%s=%d proxy=%d",
            lang,
            args.split,
            n_full,
            len(sampled),
        )

    out_df = pd.concat(parts, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    summary_path = args.out.with_suffix(".summary.json")
    summary = {
        "dataset": HF_DATASET,
        "split": args.split,
        "max_per_lang": args.max_per_lang,
        "seed": args.seed,
        "langs": list(args.langs),
        "n_total": len(out_df),
        "per_lang": stats,
        "csv": str(args.out),
        "rule": "validation-only proxy; never Phase-1 test gold for train/tune",
        "load_hint": (
            "from src.dataset import load_hf_asr_split; "
            "ds = load_hf_asr_split(lang, 'validation', max_samples=40)  "
            "# or filter loaded val by ids in this CSV"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out} rows={len(out_df)}")
    print(f"wrote {summary_path}")
    print(out_df.groupby("language").size().to_string())


if __name__ == "__main__":
    main()
