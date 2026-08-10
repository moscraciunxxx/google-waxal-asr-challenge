#!/usr/bin/env python3
"""Phase-2 candidate: waxal-benchmarking whisper-small hard-replace by decode_lang.

Only runs languages that exist as whisper-small-waxal-* on HF.
Default: upgrade ach/lug/nyn/sog/mas decode routes; keep other openset hyps.

No cross-family CTC conf. Open-source only. No Phase-1 test gold training.
Gate: run scripts/eval_proxy_whisper_waxal.py first; only upload if
true_lang_whisper_small beats openset_multihyp by ≥0.01 on proxy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT
from src.mms_infer import pick_device
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("whisper_cand")

AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
OPENSET = OUTPUT_DIR / "phase2_openset_detail.csv"
WAXAL_WHISPER = {
    "ach": "waxal-benchmarking/whisper-small-waxal-ach",
    "lug": "waxal-benchmarking/whisper-small-waxal-lug",
    "nyn": "waxal-benchmarking/whisper-small-waxal-nyn",
    "sog": "waxal-benchmarking/whisper-small-waxal-sog",
    "mas": "waxal-benchmarking/whisper-small-waxal-mas",
}


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


@torch.inference_mode()
def whisper_transcribe(model, processor, array: np.ndarray, sr: int, device: torch.device) -> str:
    if sr != 16000:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        sr = 16000
    inputs = processor(array, sampling_rate=sr, return_tensors="pt")
    feats = inputs.input_features.to(device)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    ids = model.generate(feats, max_new_tokens=128, num_beams=1, do_sample=False)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return normalize_text(text) or "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["ach", "lug", "nyn"])
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "submission_phase2_whisper_waxal_candidate.csv",
    )
    ap.add_argument("--detail", type=Path, default=OUTPUT_DIR / "phase2_whisper_waxal_detail.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    op = pd.read_csv(OPENSET)
    op["ID"] = op["ID"].astype(str)

    pred = {str(r.ID): normalize_text(str(r.prediction)) or "." for _, r in op.iterrows()}
    src = {uid: "openset_keep" for uid in pred}
    dlang = {str(r.ID): str(r.decode_lang) for _, r in op.iterrows()}

    for lang in args.langs:
        mid = WAXAL_WHISPER.get(lang)
        if not mid:
            logger.warning("no whisper for %s", lang)
            continue
        ids = [uid for uid, d in dlang.items() if d == lang]
        if not ids:
            continue
        logger.info("load %s for %d clips", mid, len(ids))
        proc = WhisperProcessor.from_pretrained(mid)
        model = WhisperForConditionalGeneration.from_pretrained(mid).to(device).eval()
        for uid in tqdm(ids, desc=f"wh-{lang}"):
            arr, sr = load_wav(AUDIO / f"{uid}.wav")
            try:
                pred[uid] = whisper_transcribe(model, proc, arr, sr, device)
                src[uid] = f"whisper_{lang}"
            except Exception as e:
                logger.warning("%s fail: %s", uid, e)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    rows = [
        {
            "ID": uid,
            "prediction": pred[uid],
            "source": src[uid],
            "decode_lang": dlang[uid],
        }
        for uid in op["ID"].astype(str).tolist()
    ]
    df = pd.DataFrame(rows)
    df.to_csv(args.detail, index=False)
    build_submission(df[["ID", "prediction"]], sample_path=SAMPLE, out_path=args.out)
    rep = check_submission(args.out, SAMPLE)
    o = pd.read_csv(PROJECT_ROOT / "submission_phase2_openset.csv").set_index("ID")["Target"].astype(str)
    n = df.set_index("ID")["prediction"].astype(str)
    rep.update(
        {
            "n_changed": int((o.reindex(n.index) != n).sum()),
            "source_counts": Counter(df.source).most_common(),
            "method": "openset + hard waxal-whisper-small by decode_lang",
            "langs": args.langs,
        }
    )
    (OUTPUT_DIR / "phase2_whisper_waxal_check.json").write_text(json.dumps(rep, indent=2, default=str))
    logger.info("%s", rep)
    print("UPLOAD", args.out)
    print("CHANGED", rep["n_changed"], dict(Counter(df.source)))


if __name__ == "__main__":
    main()
