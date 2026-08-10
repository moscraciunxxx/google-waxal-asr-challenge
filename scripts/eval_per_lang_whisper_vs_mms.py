#!/usr/bin/env python3
"""Eval per-lang own Whisper ckpts vs WAXALNet MMS-300m on validation (same protocol)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_specialist_vs_whisper_val import eval_lang
from src.config import FORBIDDEN_TRAIN_SPLITS, OUTPUT_DIR, SEED
from src.legit_fusion import beats_baseline, mean_error
from src.mms_infer import pick_device
from src.train import set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_per_lang")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", type=Path, default=Path("checkpoints/whisper-per-lang-legit"))
    p.add_argument("--max-per-lang", type=int, default=60)
    p.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR / "multi_agent_push" / "t1_val_report.json",
    )
    # Fallback: evaluate open WAXALNet whisper specialists if own ckpt missing
    p.add_argument("--allow-waxal-whisper-fallback", action="store_true")
    args = p.parse_args(argv)
    set_all_seeds(SEED)
    device = pick_device()

    waxal_wh = {
        "lin": "waxal-benchmarking/whisper-small-waxal-lin",
        "sna": "waxal-benchmarking/whisper-small-waxal-sna",
        "lug": "waxal-benchmarking/whisper-small-waxal-lug",
    }
    results = []
    for lang in ("lin", "sna", "lug"):
        own = args.ckpt_root / lang / "best"
        if own.exists():
            ckpt = str(own)
        elif args.allow_waxal_whisper_fallback:
            ckpt = waxal_wh[lang]
            logger.warning("Using open WAXALNet whisper specialist as interim for %s", lang)
        else:
            raise SystemExit(f"Missing own ckpt {own}")
        logger.info("eval %s ckpt=%s", lang, ckpt)
        results.append(eval_lang(lang, args.max_per_lang, ckpt, device))

    all_beat = all(r["beats"] for r in results)
    report = {
        "seed": SEED,
        "protocol": "validation only; same clips; normalize; jiwer; never test gold",
        "forbidden_train_splits": list(FORBIDDEN_TRAIN_SPLITS),
        "ckpt_root": str(args.ckpt_root),
        "max_per_lang": args.max_per_lang,
        "per_language": results,
        "all_languages_beat_baseline": all_beat,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s all_beat=%s", args.out, all_beat)
    for r in results:
        logger.info(
            "%s base_me=%.4f own_me=%.4f beats=%s",
            r["lang"],
            r["mean_error_baseline"],
            r["mean_error_own"],
            r["beats"],
        )
    return 0 if all_beat else 2


if __name__ == "__main__":
    raise SystemExit(main())
