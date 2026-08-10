#!/usr/bin/env python3
"""Fetch HF metadata and write Train.csv / Test.csv / SampleSubmission.csv / index."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LANGUAGES
from src.data_index import assert_no_test_gold_in_training, build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    args = p.parse_args()
    build_index(languages=args.languages, force=args.force)
    assert_no_test_gold_in_training()
    print("OK: data index built; no test gold in training.")


if __name__ == "__main__":
    main()
