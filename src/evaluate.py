"""Local WER/CER evaluation on the validation holdout (never Phase-1 test gold for tuning)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import LANGUAGES, OUTPUT_DIR, PROJECT_ROOT
from src.infer import load_model, transcribe_batch_from_hf
from src.metrics import score_by_language

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


def evaluate_checkpoint(
    checkpoint: Path | str,
    split: str = "validation",
    max_per_lang: int | None = None,
    languages: tuple[str, ...] = LANGUAGES,
    out_json: Path | None = None,
    num_beams: int = 5,
    audio_only: bool = False,
    baseline_metrics: dict | None = None,
) -> dict:
    model, processor, device = load_model(checkpoint)
    df = transcribe_batch_from_hf(
        model,
        processor,
        languages=languages,
        split=split,
        device=device,
        max_samples=max_per_lang,
        num_beams=num_beams,
        audio_only=audio_only,
    )
    if "reference" not in df.columns:
        raise RuntimeError(f"Split '{split}' has no references — cannot compute WER/CER")

    metrics = score_by_language(
        df["reference"].tolist(),
        df["prediction"].tolist(),
        df["language"].tolist(),
    )
    result = {
        "checkpoint": str(checkpoint),
        "split": split,
        "max_per_lang": max_per_lang,
        "audio_only": audio_only,
        "num_beams": num_beams,
        "metrics": metrics,
    }
    if baseline_metrics is not None:
        result["baseline_metrics"] = baseline_metrics
        try:
            result["improved_vs_baseline"] = (
                metrics["overall"]["score"] < baseline_metrics["overall"]["score"]
            )
        except Exception:
            result["improved_vs_baseline"] = None

    out_json = Path(out_json or (OUTPUT_DIR / "local_metrics.json"))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    preds_path = out_json.with_suffix(".preds.csv")
    df.to_csv(preds_path, index=False)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))
    logger.info("Wrote %s", out_json)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate WAXAL ASR checkpoint")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--split", default="validation")
    p.add_argument("--max-per-lang", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--audio-only", action="store_true")
    p.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    args = parse_args(argv)
    evaluate_checkpoint(
        checkpoint=args.checkpoint,
        split=args.split,
        max_per_lang=args.max_per_lang,
        languages=tuple(args.languages),
        out_json=args.out,
        num_beams=args.num_beams,
        audio_only=args.audio_only,
    )


if __name__ == "__main__":
    main()
