"""Fine-tune an open Whisper backbone on WAXAL lin/sna/lug (no test gold)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

from src.config import (
    CHECKPOINT_DIR,
    DEFAULT_MODEL_ID,
    GRAD_ACCUM,
    LEARNING_RATE,
    MAX_LABEL_LENGTH,
    NUM_EPOCHS,
    PROJECT_ROOT,
    SEED,
    TARGET_SR,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    WARMUP_RATIO,
)
from src.dataset import WhisperDataCollator, load_labeled_splits, prepare_whisper_example

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_all_seeds(seed: int = SEED) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(
    processor: WhisperProcessor,
    max_per_lang_split: int | None,
    languages: tuple[str, ...],
):
    """Train on HF train only; validate on HF validation only. Never test."""
    train_raw = load_labeled_splits(
        languages=languages,
        splits=("train",),
        max_per_lang_split=max_per_lang_split,
    )
    val_raw = load_labeled_splits(
        languages=languages,
        splits=("validation",),
        max_per_lang_split=max_per_lang_split,
    )

    def _map(batch):
        return prepare_whisper_example(batch, processor)

    cols_to_remove = [
        c
        for c in train_raw.column_names
        if c not in ("input_features", "labels")
    ]
    train_ds = train_raw.map(
        _map,
        remove_columns=train_raw.column_names,
        desc="Prepare train features",
    )
    val_ds = val_raw.map(
        _map,
        remove_columns=val_raw.column_names,
        desc="Prepare val features",
    )
    # Parent set_transform(lazy audio) must not leak onto feature-only datasets
    train_ds.reset_format()
    val_ds.reset_format()
    return train_ds, val_ds


def run_train(
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: Path | None = None,
    max_per_lang_split: int | None = None,
    num_epochs: float = NUM_EPOCHS,
    languages: tuple[str, ...] = ("lin", "sna", "lug"),
    seed: int = SEED,
    learning_rate: float = LEARNING_RATE,
    max_steps: int | None = None,
) -> Path:
    set_all_seeds(seed)
    output_dir = Path(output_dir or (CHECKPOINT_DIR / "whisper-waxal"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    logger.info(
        "Device: %s | model: %s | max_per_lang_split=%s | max_steps=%s | langs=%s",
        device,
        model_id,
        max_per_lang_split,
        max_steps,
        languages,
    )

    # Prefer local hub cache (avoids multi-minute hub HEAD hangs); fall back online
    try:
        processor = WhisperProcessor.from_pretrained(model_id, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_id, local_files_only=True)
        logger.info("Loaded model/processor from local cache")
    except Exception as e:
        logger.warning("local_files_only load failed (%s); using hub", e)
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
    # Multilingual / audio-only friendly: do not force a language token at train time
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    train_ds, val_ds = build_datasets(processor, max_per_lang_split, languages)
    logger.info("Train size=%d val size=%d", len(train_ds), len(val_ds))

    collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    use_mps = device.type == "mps"
    # Short smoke: step-based eval/save so max_steps runs produce a checkpoint + metric
    if max_steps is not None and max_steps > 0:
        eval_strategy = "steps"
        save_strategy = "steps"
        eval_steps = max(1, min(max_steps, 5))
        save_steps = eval_steps
        logging_steps = 1
        load_best = False
    else:
        eval_strategy = "epoch"
        save_strategy = "epoch"
        eval_steps = None
        save_steps = None
        logging_steps = 10
        load_best = True

    ta_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=learning_rate,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=num_epochs,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        predict_with_generate=False,  # keep train loop light; separate eval script
        fp16=False,
        bf16=False,
        report_to=[],
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_num_workers=0,
        # MPS: use default; gradient checkpointing can help memory
        gradient_checkpointing=True,
    )
    if max_steps is not None and max_steps > 0:
        ta_kwargs["max_steps"] = int(max_steps)
        ta_kwargs["eval_steps"] = eval_steps
        ta_kwargs["save_steps"] = save_steps
    args = Seq2SeqTrainingArguments(**ta_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    # transformers API drift: prefer processing_class, fall back to tokenizer=
    try:
        trainer = Seq2SeqTrainer(
            **trainer_kwargs, processing_class=processor.feature_extractor
        )
    except TypeError:
        trainer = Seq2SeqTrainer(
            **trainer_kwargs, tokenizer=processor.feature_extractor
        )

    trainer.train()
    trainer.save_model(str(output_dir / "best"))
    processor.save_pretrained(str(output_dir / "best"))

    meta = {
        "model_id": model_id,
        "output_dir": str(output_dir / "best"),
        "seed": seed,
        "languages": list(languages),
        "max_per_lang_split": max_per_lang_split,
        "num_epochs": num_epochs,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "device": str(device),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "own_ft_priority": "whisper-first",
        "rule": "trained only on train split; validation for eval_loss; never test gold",
        "forbidden_splits": ["test"],
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved checkpoint to %s", output_dir / "best")
    return output_dir / "best"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Whisper on WAXAL ASR (open-source only)")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-per-lang-split", type=int, default=None, help="Smoke: limit samples per lang/split")
    p.add_argument("--epochs", type=float, default=NUM_EPOCHS)
    p.add_argument("--max-steps", type=int, default=None, help="Smoke: cap optimizer steps (train only)")
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--languages", nargs="+", default=list(("lin", "sna", "lug")))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Ensure project root on path when run as script
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    run_train(
        model_id=args.model_id,
        output_dir=args.output_dir,
        max_per_lang_split=args.max_per_lang_split,
        num_epochs=args.epochs,
        languages=tuple(args.languages),
        seed=args.seed,
        learning_rate=args.lr,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
