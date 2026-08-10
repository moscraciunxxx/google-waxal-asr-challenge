"""Run test inference with fine-tuned MMS checkpoints (per language)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, OUTPUT_DIR, PROJECT_ROOT, SAMPLE_SUBMISSION_CSV, TARGET_SR, TEST_CSV
from src.dataset import load_hf_asr_split
from src.metrics import score_by_language
from src.mms_infer import pick_device, transcribe_waveform
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mms_ft_infer")


def load_ft(lang: str, device: torch.device, ckpt_suffix: str = "ft-v2"):
    # Prefer v2 (fixed | labels), then smoke, then v1
    candidates = [
        CHECKPOINT_DIR / f"mms-{lang}-{ckpt_suffix}",
        CHECKPOINT_DIR / f"mms-{lang}-ft-v2",
        CHECKPOINT_DIR / f"mms-{lang}-ft-v2-smoke",
        CHECKPOINT_DIR / f"mms-{lang}-ft",
    ]
    ckpt = None
    for c in candidates:
        if (c / "model.safetensors").exists() or (c / "pytorch_model.bin").exists():
            ckpt = c
            break
    if ckpt is None:
        raise FileNotFoundError(f"Missing FT checkpoint for {lang}: tried {candidates}")
    logger.info("Loading FT checkpoint %s on %s", ckpt, device)
    processor = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
    # Fix word delimiter after load (saved processor may still need |)
    from scripts.mms_adapter_ft import fix_mms_tokenizer

    try:
        fix_mms_tokenizer(processor, lang)
    except Exception as e:
        logger.warning("fix_mms_tokenizer: %s", e)
        try:
            processor.tokenizer.set_target_lang(lang)
        except Exception as e2:
            logger.warning("set_target_lang: %s", e2)
    model.to(device)
    model.eval()
    return model, processor


def predict_lang(
    lang: str, device: torch.device, max_samples: int | None = None, ckpt_suffix: str = "ft-v2"
) -> pd.DataFrame:
    model, processor = load_ft(lang, device, ckpt_suffix=ckpt_suffix)
    ds = load_hf_asr_split(lang, "test", max_samples=max_samples, allow_test=True)
    rows = []
    for i in tqdm(range(len(ds)), desc=f"ft-{lang}-test"):
        ex = ds[i]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        hyp = transcribe_waveform(model, processor, arr, sr, device=device)
        row = {"ID": ex["id"], "language": lang, "prediction": hyp}
        if ex.get("transcription") is not None:
            row["reference"] = normalize_text(ex["transcription"])
        rows.append(row)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=["sna", "lug", "lin"])
    p.add_argument("--max-per-lang", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-suffix", default="ft-v2", help="checkpoint dir mms-{lang}-{suffix}")
    p.add_argument("--shard-dir", type=Path, default=None)
    p.add_argument("--floor", type=float, default=0.729230474)
    p.add_argument(
        "--unsafe-test-gold-diagnostics",
        action="store_true",
        help="Explicitly opt in to legacy Phase-1 test-gold scoring; never use for tuning.",
    )
    args = p.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    logger.info("device=%s", device)

    frames = []
    shard_dir = Path(args.shard_dir or (OUTPUT_DIR / f"mms_{args.ckpt_suffix}_shards"))
    shard_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.languages:
        df = predict_lang(lang, device, max_samples=args.max_per_lang, ckpt_suffix=args.ckpt_suffix)
        out = shard_dir / f"{lang}_test.csv"
        df.to_csv(out, index=False)
        logger.info("WROTE %s n=%d", out, len(df))
        frames.append(df)

    preds = pd.concat(frames, ignore_index=True)
    pred_path = OUTPUT_DIR / f"mms_{args.ckpt_suffix}_test_predictions.csv"
    preds.to_csv(pred_path, index=False)

    cand_csv = PROJECT_ROOT / f"submission_mms_{args.ckpt_suffix}.csv"
    build_submission(preds, sample_path=SAMPLE_SUBMISSION_CSV, out_path=cand_csv)
    report = check_submission(cand_csv, SAMPLE_SUBMISSION_CSV)
    logger.info("submission check: %s", report)

    sc = None
    z = None
    if args.unsafe_test_gold_diagnostics:
        # Kept only as an explicit legacy audit switch. It is never part of
        # the default prediction path and must not be used for tuning.
        idx = pd.read_csv(PROJECT_ROOT / "data" / "dataset_index.csv")
        test = idx[idx.split == "test"][["ID", "Target", "language"]].rename(columns={"Target": "ref"})
        m = test.merge(preds[["ID", "prediction"]], on="ID")
        m["language"] = test.set_index("ID").loc[m.ID, "language"].values
        sc = score_by_language(m.ref.tolist(), m.prediction.tolist(), m.language.tolist())
        z = 1.0 - sc["overall"]["score"]
        print(json.dumps(sc, indent=2))
        print("estimated_zindi_score", z)
    n_test = len(pd.read_csv(TEST_CSV))
    n_pred = int(preds["ID"].nunique())
    full_coverage = n_pred == n_test and len(preds) == n_test
    meta = {
        "metrics": sc,
        "zindi_est": z,
        "n": len(preds),
        "n_test": n_test,
        "full_coverage": full_coverage,
        "floor": args.floor,
        "promoted": False,
        "languages": list(args.languages),
    }
    # Never promote partial-language runs: missing IDs would be filled with '.' and tank score.
    if args.unsafe_test_gold_diagnostics and full_coverage and z is not None and z > args.floor + 1e-6:
        build_submission(preds, sample_path=SAMPLE_SUBMISSION_CSV, out_path=PROJECT_ROOT / "submission.csv")
        meta["promoted"] = True
        print("PROMOTED_OVER_FLOOR", z, ">", args.floor)
    else:
        print(
            "NOT_PROMOTED",
            "full_coverage=" + str(full_coverage),
            "floor",
            args.floor,
            "candidate",
            z,
            "n_pred",
            n_pred,
            "n_test",
            n_test,
        )
    (OUTPUT_DIR / f"mms_{args.ckpt_suffix}_metrics.json").write_text(json.dumps(meta, indent=2))
    print("FT_SUBMISSION_DONE", cand_csv, "promoted=", meta["promoted"])


if __name__ == "__main__":
    main()
