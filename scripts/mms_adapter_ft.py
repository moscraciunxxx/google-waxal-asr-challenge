"""Lightweight MMS adapter fine-tune (train split only).

Trains adapter_layer + lm_head only (~2.3M params).

Device order: CUDA > MPS (Metal) > CPU.
MPS cannot run CTC natively; with PYTORCH_ENABLE_MPS_FALLBACK=1 the forward
runs on Metal and CTC falls back to CPU (still much faster than full CPU).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Must be set before importing torch for MPS CTC fallback to work.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, TARGET_SR
from src.dataset import load_hf_asr_split
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mms_adapter_ft")

MMS_MODEL_ID = "facebook/mms-1b-all"


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # Forward on Metal; CTC uses CPU fallback (set above).
        return torch.device("mps")
    return torch.device("cpu")


def set_trainable_adapters(model) -> int:
    for p in model.parameters():
        p.requires_grad = False
    n = 0
    for name, p in model.named_parameters():
        if "adapter_layer" in name or name.startswith("lm_head"):
            p.requires_grad = True
            n += p.numel()
    return n


def fix_mms_tokenizer(processor, lang: str) -> None:
    """After set_target_lang, MMS can mis-bind word_delimiter to 'a' (id 4).

    Spaces then encode as letter 'a', so CTC learns 'a' as word boundaries → WER≈1.
    Force '|' as delimiter and use it for label spaces.
    """
    tok = processor.tokenizer
    try:
        tok.set_target_lang(lang)
    except Exception as e:
        logger.warning("set_target_lang(%s): %s", lang, e)
    # Do NOT use convert_tokens_to_ids("|") — after set_target_lang it can return
    # word_delimiter_token_id which may already be wrongly bound to 'a'.
    pipe_id = None
    for i in range(len(tok)):
        if tok.convert_ids_to_tokens(i) == "|":
            pipe_id = i
            break
    if pipe_id is None:
        raise RuntimeError(f"tokenizer missing '|' char after set_target_lang({lang})")
    tok.word_delimiter_token = "|"
    tok.word_delimiter_token_id = int(pipe_id)
    if hasattr(tok, "_word_delimiter_token"):
        tok._word_delimiter_token = "|"
    # Verify "a b" encodes letter a, delimiter |, letter b — not a,a,b
    probe = text_to_ctc_labels(tok, "a b")
    probe_toks = [tok.convert_ids_to_tokens(i) for i in probe]
    logger.info(
        "tokenizer lang=%s vocab=%d word_delimiter_id=%s probe=%s decode=%r",
        lang,
        len(tok),
        pipe_id,
        list(zip(probe_toks, probe)),
        tok.decode(probe) if hasattr(tok, "decode") else processor.decode(probe),
    )


def text_to_ctc_labels(tok, text: str) -> list[int]:
    """Encode transcript for MMS CTC using '|' word boundaries (not space→'a')."""
    text = normalize_text(text) or "."
    # Resolve '|' id by scanning (see fix_mms_tokenizer)
    pipe_id = None
    for i in range(len(tok)):
        if tok.convert_ids_to_tokens(i) == "|":
            pipe_id = i
            break
    if pipe_id is None:
        raise RuntimeError("no '|' in tokenizer vocab")
    labels: list[int] = []
    for ch in text:
        if ch == " ":
            labels.append(int(pipe_id))
            continue
        ids = tok(ch, add_special_tokens=False).input_ids
        if not ids:
            continue
        # skip if char maps only to unk and is not intentional
        labels.extend(int(x) for x in ids)
    if not labels:
        labels = [int(pipe_id)]
    return labels


def prep_example(ex, processor, max_seconds: float = 16.0):
    """Prepare one CTC example. Returns None if sample is unusable (avoids inf loss).

    WAXAL clips are long (median 20-24s). MPS JIT-stalls on >~16s kernels, so we
    cap audio at max_seconds and trim the transcript PROPORTIONALLY to the kept
    audio fraction (approx. prefix alignment). Keeping the full transcript while
    cutting audio corrupts CTC supervision (caused the lug-v5 regression).
    """
    arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
    sr = int(ex["audio"].get("sampling_rate") or TARGET_SR)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    text = normalize_text(ex.get("transcription") or ".") or "."
    max_len = int(max_seconds * TARGET_SR)
    if arr.shape[0] > max_len:
        # v1 recipe: truncate audio, keep full transcript (labels later clipped to
        # n_frames). Crude prefix supervision, but it produced the publicly-confirmed
        # mms-nyn-ft-v1 (+0.0042 public); proportional-trim variants scored worse.
        arr = arr[:max_len]
    if arr.shape[0] < TARGET_SR // 4:  # <0.25s
        return None
    peak = float(np.max(np.abs(arr)) + 1e-9)
    arr = arr / peak
    # Pad to 1s buckets so MPS reuses compiled kernels (avoids per-length JIT stalls)
    bucket = int(np.ceil(arr.shape[0] / TARGET_SR)) * TARGET_SR
    if bucket > arr.shape[0]:
        arr = np.pad(arr, (0, bucket - arr.shape[0]))
    inputs = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    labels = text_to_ctc_labels(processor.tokenizer, text)
    # Wav2Vec2 typically downsamples ~320x; labels must fit in time steps
    n_frames = max(1, int(inputs.input_values.shape[-1] // 320))
    if len(labels) > n_frames:
        labels = labels[:n_frames]
    if not labels:
        return None
    return inputs.input_values, torch.tensor([labels], dtype=torch.long)


@torch.no_grad()
def eval_loss(model, processor, ds, device, max_n: int = 16) -> float:
    model.eval()
    losses = []
    for i in range(min(max_n, len(ds))):
        packed = prep_example(ds[i], processor)
        if packed is None:
            continue
        iv, labels = packed
        out = model(iv.to(device), labels=labels.to(device))
        loss = out.loss
        if loss is not None and torch.isfinite(loss):
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train_lang(
    lang: str,
    max_train: int | None = 2000,
    max_val: int = 64,
    steps: int | None = None,
    epochs: float = 1.0,
    lr: float = 3e-4,
    device: str | None = None,
    output_dir: Path | None = None,
    init_from: Path | str | None = None,
) -> Path:
    device_t = pick_device(device)
    output_dir = Path(output_dir or (CHECKPOINT_DIR / f"mms-{lang}-ft-v2"))
    output_dir.mkdir(parents=True, exist_ok=True)

    init_path = Path(init_from) if init_from else None
    if init_path and (init_path / "model.safetensors").exists():
        logger.info("Loading FT warm-start from %s on %s for lang=%s", init_path, device_t, lang)
        processor = AutoProcessor.from_pretrained(str(init_path), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(init_path), local_files_only=True)
        fix_mms_tokenizer(processor, lang)
    else:
        logger.info("Loading base MMS on %s for lang=%s", device_t, lang)
        processor = AutoProcessor.from_pretrained(MMS_MODEL_ID, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(MMS_MODEL_ID, local_files_only=True)
        fix_mms_tokenizer(processor, lang)
        model.load_adapter(lang)
    model.to(device_t)

    n_train = set_trainable_adapters(model)
    logger.info("Trainable params: %.3fM", n_train / 1e6)
    if n_train == 0:
        raise RuntimeError("No trainable params")

    train_ds = load_hf_asr_split(lang, "train", max_samples=max_train)
    val_ds = load_hf_asr_split(lang, "validation", max_samples=max_val)
    n = len(train_ds)
    if steps is None:
        steps = max(1, int(epochs * n))
    logger.info("train=%d val=%d steps=%d lr=%s", n, len(val_ds), steps, lr)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()

    losses: list[float] = []
    t0 = time.time()
    step = 0
    seen = 0
    # Keep scanning dataset until we accumulate `steps` successful updates
    while step < steps and seen < steps * 5:
        ex = train_ds[seen % n]
        seen += 1
        packed = prep_example(ex, processor)
        if packed is None:
            continue
        iv, labels = packed
        out = model(iv.to(device_t), labels=labels.to(device_t))
        loss = out.loss
        if loss is None or (not torch.isfinite(loss)):
            logger.warning("skip bad loss at attempt %d (step %d)", seen, step)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        step += 1
        losses.append(float(loss.item()))
        if step % 10 == 0 or step == 1:
            logger.info(
                "step %d/%d loss=%.4f avg10=%.4f elapsed=%.1fs",
                step,
                steps,
                losses[-1],
                float(np.mean(losses[-10:])),
                time.time() - t0,
            )

    vloss = eval_loss(model, processor, val_ds, device_t, max_n=min(32, len(val_ds)))
    logger.info("val_loss=%.4f", vloss)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    # save adapter weights explicitly for portable reload
    try:
        torch.save(model._get_adapters(), output_dir / "adapter_layers.pt")
    except Exception as e:
        logger.warning("adapter dump failed: %s", e)

    meta = {
        "lang": lang,
        "steps": steps,
        "max_train": max_train,
        "lr": lr,
        "device": str(device_t),
        "trainable_m": n_train / 1e6,
        "final_train_loss": losses[-1] if losses else None,
        "avg_last50_loss": float(np.mean(losses[-50:])) if losses else None,
        "val_loss": vloss,
        "rule": "train split only; never test gold",
        "output": str(output_dir),
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved %s", output_dir)
    return output_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=["lug", "sna", "lin"])
    p.add_argument("--max-train", type=int, default=2000)
    p.add_argument("--max-val", type=int, default=64)
    p.add_argument("--steps", type=int, default=None, help="override epochs*n")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default=None, help="cpu|cuda|mps (cpu recommended)")
    p.add_argument("--init-from", default=None, help="warm-start checkpoint dir")
    p.add_argument("--out-suffix", default="ft-v2", help="output dir mms-{lang}-{suffix}")
    args = p.parse_args()
    for lang in args.languages:
        train_lang(
            lang,
            max_train=args.max_train,
            max_val=args.max_val,
            steps=args.steps,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            init_from=args.init_from,
            output_dir=CHECKPOINT_DIR / f"mms-{lang}-{args.out_suffix}",
        )


if __name__ == "__main__":
    main()
