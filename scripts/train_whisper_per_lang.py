#!/usr/bin/env python3
"""Per-language continued Whisper FT from WAXALNet specialists (train split only).

Init: waxal-benchmarking/whisper-small-waxal-{lang} when available, else openai/whisper-small.
Never loads test gold.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, SEED
from src.train import run_train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_per_lang")

WAXAL_WHISPER = {
    "lin": "waxal-benchmarking/whisper-small-waxal-lin",
    "sna": "waxal-benchmarking/whisper-small-waxal-sna",
    "lug": "waxal-benchmarking/whisper-small-waxal-lug",
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=["lin", "sna", "lug"])
    p.add_argument("--max-per-lang-split", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-root", type=Path, default=CHECKPOINT_DIR / "whisper-per-lang-legit")
    args = p.parse_args(argv)

    results = []
    for lang in args.languages:
        init = WAXAL_WHISPER.get(lang, "openai/whisper-small")
        # Prefer local cache; if missing HF will download
        out = args.out_root / lang
        logger.info("=== FT lang=%s init=%s out=%s ===", lang, init, out)
        try:
            best = run_train(
                model_id=init,
                output_dir=out,
                max_per_lang_split=args.max_per_lang_split,
                num_epochs=3.0,
                languages=(lang,),
                seed=args.seed,
                learning_rate=args.lr,
                max_steps=args.max_steps,
            )
            results.append({"lang": lang, "init": init, "best": str(best), "ok": True})
        except Exception as e:
            logger.exception("FT failed for %s: %s", lang, e)
            # fallback openai/whisper-small
            if init != "openai/whisper-small":
                logger.info("Retry %s from openai/whisper-small", lang)
                best = run_train(
                    model_id="openai/whisper-small",
                    output_dir=out,
                    max_per_lang_split=args.max_per_lang_split,
                    num_epochs=3.0,
                    languages=(lang,),
                    seed=args.seed,
                    learning_rate=args.lr,
                    max_steps=args.max_steps,
                )
                results.append(
                    {
                        "lang": lang,
                        "init": "openai/whisper-small",
                        "best": str(best),
                        "ok": True,
                        "fallback": True,
                    }
                )
            else:
                results.append({"lang": lang, "ok": False, "error": str(e)})

    summary = {"results": results, "seed": args.seed, "forbidden": ["test"]}
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "per_lang_train_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("summary %s", summary)
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
