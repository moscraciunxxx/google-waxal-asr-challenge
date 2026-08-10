"""Fine-tune MMS adapters on WAXAL train (lin/sna/lug) without test gold."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    set_seed,
)

from src.config import CHECKPOINT_DIR, FORBIDDEN_TRAIN_SPLITS, PROJECT_ROOT, SEED, TARGET_SR
from src.dataset import load_hf_asr_split
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mms_train")

MMS_MODEL_ID = "facebook/mms-1b-all"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DataCollatorCTC:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(
            labels=label_features, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    audio = batch["audio"]
    array = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    if sr != TARGET_SR:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
    batch["input_values"] = processor(array, sampling_rate=TARGET_SR).input_values[0]
    batch["labels"] = processor.tokenizer(normalize_text(batch["transcription"])).input_ids
    return batch


def train_lang(
    lang: str,
    max_train: int | None = None,
    max_val: int | None = 64,
    epochs: float = 3.0,
    lr: float = 1e-4,
    output_dir: Path | None = None,
    seed: int = SEED,
) -> Path:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir or (CHECKPOINT_DIR / f"mms-{lang}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Match official MMS adapter usage: load full model, then switch adapter + tokenizer lang.
    # Do not pass target_lang into from_pretrained (that resizes lm_head before adapter load).
    processor = AutoProcessor.from_pretrained(MMS_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(
        MMS_MODEL_ID,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )
    processor.tokenizer.set_target_lang(lang)
    model.load_adapter(lang)

    # Freeze everything, then train only language adapters + CTC head.
    # Full 1B fine-tune OOMs / stalls on MPS; adapters are ~2M params.
    for p in model.parameters():
        p.requires_grad = False
    n_train = 0
    for name, p in model.named_parameters():
        if "adapter_layer" in name or name.startswith("lm_head"):
            p.requires_grad = True
            n_train += p.numel()
    logger.info(
        "Trainable params: %.3fM / %.1fM (adapters+lm_head only)",
        n_train / 1e6,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    if n_train == 0:
        raise RuntimeError("No trainable adapter/lm_head params found")

    train_ds = load_hf_asr_split(lang, "train", max_samples=max_train)
    val_ds = load_hf_asr_split(lang, "validation", max_samples=max_val)

    def _map(batch):
        return prepare_dataset(batch, processor)

    train_ds = train_ds.map(_map, remove_columns=train_ds.column_names, desc=f"prep train {lang}")
    val_ds = val_ds.map(_map, remove_columns=val_ds.column_names, desc=f"prep val {lang}")

    collator = DataCollatorCTC(processor=processor)
    # transformers>=5 dropped group_by_length / tokenizer Trainer kwarg
    # batch=1 + accum keeps MPS memory under control for 1B forward pass
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_steps=max(10, int(0.05 * max(1, (max_train or 1000) // 16))),
        fp16=False,
        logging_steps=5,
        report_to=[],
        seed=seed,
        data_seed=seed,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )
    trainer.train()
    best = output_dir / "best"
    trainer.save_model(str(best))
    processor.save_pretrained(str(best))
    meta = {
        "model_id": MMS_MODEL_ID,
        "lang": lang,
        "max_train": max_train,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "output": str(best),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "rule": "train split only; never test gold",
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved %s", best)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=["lin", "sna", "lug"])
    p.add_argument("--max-train", type=int, default=None, help="None = all train")
    p.add_argument("--max-val", type=int, default=128)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    for lang in args.languages:
        train_lang(
            lang,
            max_train=args.max_train,
            max_val=args.max_val,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
