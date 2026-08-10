#!/usr/bin/env python3
"""End-to-end: prepare → (optional baseline) → train → eval → submit → phase2 smoke."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, DEFAULT_MODEL_ID, LANGUAGES, OUTPUT_DIR, PROJECT_ROOT
from src.data_index import assert_no_test_gold_in_training, build_index
from src.evaluate import evaluate_checkpoint
from src.infer import load_model, transcribe_batch_from_hf, run_predict_test
from src.submission import build_submission, check_submission
from src.train import run_train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Tiny data / short train")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--max-per-lang-split", type=int, default=None)
    p.add_argument("--eval-max-per-lang", type=int, default=None)
    p.add_argument("--scratch", type=Path, default=None, help="Copy evidence logs here")
    args = p.parse_args()

    if args.smoke:
        max_split = args.max_per_lang_split or 8
        epochs = args.epochs or 1.0
        eval_max = args.eval_max_per_lang or 3
    else:
        max_split = args.max_per_lang_split
        epochs = args.epochs or 3.0
        eval_max = args.eval_max_per_lang

    scratch = args.scratch
    if scratch:
        scratch.mkdir(parents=True, exist_ok=True)

    # 1) Data
    logger.info("=== prepare data ===")
    build_index(languages=LANGUAGES, force=False)
    assert_no_test_gold_in_training()
    if scratch:
        (scratch / "rules_check.log").write_text(
            "open-source only: torch, transformers, datasets, jiwer, whisper-small\n"
            "test gold excluded from Train.csv and train loops (FORBIDDEN_TRAIN_SPLITS)\n"
            "assert_no_test_gold_in_training: OK\n"
        )

    # 2) Optional zero-shot baseline on validation subset
    baseline_metrics = None
    if not args.skip_baseline:
        logger.info("=== zero-shot baseline ===")
        base_out = OUTPUT_DIR / "baseline_metrics.json"
        try:
            base_result = evaluate_checkpoint(
                checkpoint=args.model_id,
                split="validation",
                max_per_lang=eval_max if eval_max else 32,
                out_json=base_out,
                num_beams=1,
            )
            baseline_metrics = base_result["metrics"]
        except Exception as e:
            logger.warning("Baseline failed (continuing): %s", e)

    # 3) Train
    ckpt = args.checkpoint
    if not args.skip_train:
        logger.info("=== train ===")
        ckpt = run_train(
            model_id=args.model_id,
            output_dir=CHECKPOINT_DIR / ("whisper-waxal-smoke" if args.smoke else "whisper-waxal"),
            max_per_lang_split=max_split,
            num_epochs=epochs,
            languages=LANGUAGES,
        )
    if ckpt is None:
        raise SystemExit("No checkpoint; pass --checkpoint or enable training")

    # 4) Eval
    logger.info("=== evaluate ===")
    metrics_path = OUTPUT_DIR / "local_metrics.json"
    result = evaluate_checkpoint(
        checkpoint=ckpt,
        split="validation",
        max_per_lang=eval_max,
        out_json=metrics_path,
        num_beams=5,
        baseline_metrics=baseline_metrics,
    )
    if scratch:
        shutil.copy(metrics_path, scratch / "local_metrics.json")
        (scratch / "eval.log").write_text(json.dumps(result, indent=2))

    # 5) Test predictions + submission
    logger.info("=== predict test + submission ===")
    pred_path = run_predict_test(
        checkpoint=ckpt,
        out_csv=OUTPUT_DIR / "test_predictions.csv",
        max_per_lang=eval_max if args.smoke else None,
        audio_only=False,
        num_beams=5,
    )
    import pandas as pd

    preds = pd.read_csv(pred_path)
    sub = build_submission(preds, out_path=PROJECT_ROOT / "submission.csv")
    # also under outputs/
    build_submission(preds, out_path=OUTPUT_DIR / "submission.csv")
    report = check_submission(PROJECT_ROOT / "submission.csv")
    if scratch:
        shutil.copy(PROJECT_ROOT / "submission.csv", scratch / "submission.csv")
        (scratch / "submission_check.log").write_text(json.dumps(report, indent=2))
        shutil.copy(pred_path, scratch / "test_predictions.csv")

    # 6) Phase-2 audio-only smoke
    logger.info("=== phase-2 audio-only smoke ===")
    model, processor, device = load_model(ckpt)
    phase2 = transcribe_batch_from_hf(
        model,
        processor,
        languages=LANGUAGES,
        split="validation",
        device=device,
        max_samples=eval_max or 3,
        num_beams=3,
        audio_only=True,
    )
    assert len(phase2) >= 1
    assert phase2["prediction"].astype(str).str.len().gt(0).all()
    phase2_path = OUTPUT_DIR / "phase2_smoke_preds.csv"
    phase2.to_csv(phase2_path, index=False)
    if scratch:
        (scratch / "phase2_mode_smoke.log").write_text(
            f"audio_only=True rows={len(phase2)}\n"
            f"all non-empty={phase2['prediction'].astype(str).str.strip().ne('').all()}\n"
            f"sample:\n{phase2.head(5).to_string()}\n"
        )

    logger.info("Pipeline complete. metrics=%s", result["metrics"]["overall"])


if __name__ == "__main__":
    main()
