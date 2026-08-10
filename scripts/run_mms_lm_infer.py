"""MMS FT greedy vs KenLM/pyctcdecode beam; write shards and score diagnostics."""

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

from pyctcdecode import build_ctcdecoder
from scripts.mms_adapter_ft import fix_mms_tokenizer
from src.config import CHECKPOINT_DIR, OUTPUT_DIR, PROJECT_ROOT, SAMPLE_SUBMISSION_CSV, TARGET_SR
from src.dataset import load_hf_asr_split
from src.metrics import score_by_language
from src.mms_infer import pick_device, transcribe_waveform
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mms_lm_infer")


def build_decoder(processor, lang: str, alpha: float, beta: float):
    vocab = processor.tokenizer.get_vocab()
    id2 = {i: t for t, i in vocab.items()}
    labels = [id2[i] for i in range(len(id2))]
    lm = ROOT / "data" / "lms" / f"{lang}_2gram.arpa"
    uni_path = ROOT / "data" / "lms" / f"{lang}_unigrams.txt"
    unigrams = [u for u in uni_path.read_text().splitlines() if u.strip()]
    return build_ctcdecoder(
        labels,
        kenlm_model_path=str(lm),
        unigrams=unigrams,
        alpha=alpha,
        beta=beta,
    )


@torch.inference_mode()
def decode_lm(model, processor, decoder, array, sr, device, beam_width: int = 50) -> str:
    array = np.asarray(array, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.max(np.abs(array)) + 1e-9)
    if peak > 0:
        array = array / peak
    inputs = processor(array, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
    text = decoder.decode(logits, beam_width=beam_width)
    text = text.replace("|", " ")
    return normalize_text(text) or "."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=["sna", "lug"])
    p.add_argument("--ckpt-suffix", default="ft-v2")
    p.add_argument("--device", default=None)
    p.add_argument("--max-per-lang", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beam", type=int, default=50)
    p.add_argument("--floor", type=float, default=0.729230474)
    p.add_argument(
        "--unsafe-test-gold-diagnostics",
        action="store_true",
        help="Explicitly opt in to legacy Phase-1 test-gold scoring; never use for tuning.",
    )
    args = p.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    shard_dir = OUTPUT_DIR / f"mms_{args.ckpt_suffix}_lm_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    for lang in args.languages:
        ckpt = CHECKPOINT_DIR / f"mms-{lang}-{args.ckpt_suffix}"
        if not (ckpt / "model.safetensors").exists():
            raise FileNotFoundError(ckpt)
        logger.info("lang=%s ckpt=%s device=%s", lang, ckpt, device)
        processor = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
        fix_mms_tokenizer(processor, lang)
        model.to(device).eval()
        decoder = build_decoder(processor, lang, args.alpha, args.beta)
        ds = load_hf_asr_split(lang, "test", max_samples=args.max_per_lang, allow_test=True)
        rows = []
        for i in tqdm(range(len(ds)), desc=f"lm-{lang}"):
            ex = ds[i]
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"]["sampling_rate"])
            hyp = decode_lm(model, processor, decoder, arr, sr, device, beam_width=args.beam)
            row = {"ID": ex["id"], "language": lang, "prediction": hyp}
            if ex.get("transcription") is not None:
                row["reference"] = normalize_text(ex["transcription"])
            rows.append(row)
        df = pd.DataFrame(rows)
        out = shard_dir / f"{lang}_test.csv"
        df.to_csv(out, index=False)
        logger.info("WROTE %s n=%d", out, len(df))
        frames.append(df)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    preds = pd.concat(frames, ignore_index=True)
    sc = None
    z = None
    if args.unsafe_test_gold_diagnostics:
        # Explicit legacy audit only. This reads Phase-1 test gold and must
        # never be used for model or route selection.
        idx = pd.read_csv(PROJECT_ROOT / "data" / "dataset_index.csv")
        test = idx[idx.split == "test"][["ID", "Target", "language"]].rename(columns={"Target": "ref"})
        m = test[["ID", "ref"]].merge(preds[["ID", "prediction", "language"]], on="ID")
        sc = score_by_language(m.ref.tolist(), m.prediction.tolist(), m.language.tolist())
        z = 1.0 - sc["overall"]["score"]
        print(json.dumps(sc, indent=2))
        print("lm_zindi_est", z)
    meta = {
        "metrics": sc,
        "zindi_est": z,
        "unsafe_test_gold_diagnostics": args.unsafe_test_gold_diagnostics,
        "alpha": args.alpha,
        "beta": args.beta,
        "beam": args.beam,
    }
    (OUTPUT_DIR / f"mms_{args.ckpt_suffix}_lm_metrics.json").write_text(json.dumps(meta, indent=2))
    print("LM_INFER_DONE", shard_dir)


def pd_rows(rows):
    import pandas as pd

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
