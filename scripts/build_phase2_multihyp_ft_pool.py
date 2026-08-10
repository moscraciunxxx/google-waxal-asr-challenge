#!/usr/bin/env python3
"""Phase-2 multi-hyp with same-family upgraded models in the candidate pool.

Unlike post-hoc hard replace (goal_candidate), this re-runs multi-hyp CTC conf
among upgraded waxal-300m checkpoints when available:

  lid=luo → ach_lmhead | lug_ft_or_waxal | sog
  lid=lug → lug_ft_or_waxal | nyn | sog

Prefer:
  - checkpoints/waxal-ach-lmhead-ft for ach
  - checkpoints/waxal-lug-lmhead-ft if present else checkpoints/mms-lug-ft-v2
    (note: mms-lug-ft-v2 is 1B family — only used as hard route specialist if
    --allow-1b-lug is set; default prefers waxal-lug-lmhead-ft only)

No Phase-1 test gold. Open-source only.
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
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, OUTPUT_DIR, PROJECT_ROOT
from src.mms_infer import pick_device, transcribe_waveform
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import fix_mms_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("multihyp_ft_pool")

AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
LID = OUTPUT_DIR / "phase2_lid126_full.csv"
OPENSET = OUTPUT_DIR / "phase2_openset_detail.csv"

WAXAL = {
    "ach": "waxal-benchmarking/mms-300m-waxal-ach",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
    "nyn": "waxal-benchmarking/mms-300m-waxal-nyn",
    "sog": "waxal-benchmarking/mms-300m-waxal-sog",
    "mas": "waxal-benchmarking/mms-300m-waxal-mas",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "lin": "waxal-benchmarking/mms-300m-waxal-lin",
}


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


def load_model(path_or_id: str, device: torch.device, lang_hint: str | None = None):
    p = Path(path_or_id)
    if p.exists():
        proc = AutoProcessor.from_pretrained(str(p), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(p), local_files_only=True)
        if lang_hint:
            try:
                fix_mms_tokenizer(proc, lang_hint)
            except Exception:
                pass
        tag = f"local:{p.name}"
    else:
        proc = AutoProcessor.from_pretrained(path_or_id)
        model = Wav2Vec2ForCTC.from_pretrained(path_or_id)
        tag = f"hf:{path_or_id.split('/')[-1]}"
    model.to(device).eval()
    return model, proc, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-1b-lug", action="store_true", help="Use mms-lug-ft-v2 if no waxal-lug-lmhead")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "submission_phase2_multihyp_ft_pool.csv")
    ap.add_argument("--detail", type=Path, default=OUTPUT_DIR / "phase2_multihyp_ft_pool_detail.csv")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    lid = pd.read_csv(LID)
    if args.max_files:
        lid = lid.head(args.max_files)

    # Resolve model paths per logical lang
    ach_path = CHECKPOINT_DIR / "waxal-ach-lmhead-ft"
    lug_waxal_ft = CHECKPOINT_DIR / "waxal-lug-lmhead-ft"
    lug_1b = CHECKPOINT_DIR / "mms-lug-ft-v2"
    if ach_path.exists():
        ach_id = str(ach_path)
    else:
        ach_id = WAXAL["ach"]
    if lug_waxal_ft.exists() and (lug_waxal_ft / "model.safetensors").exists():
        lug_id = str(lug_waxal_ft)
        lug_hint = None
    elif args.allow_1b_lug and lug_1b.exists():
        lug_id = str(lug_1b)
        lug_hint = "lug"
        logger.warning("using 1B mms-lug-ft-v2 in multi-hyp pool (cross-size; conf still same-path)")
    else:
        lug_id = WAXAL["lug"]
        lug_hint = None

    cache: dict[str, tuple] = {}

    def get(lang: str):
        if lang in cache:
            return cache[lang]
        if lang == "ach":
            cache[lang] = load_model(ach_id, device)
        elif lang == "lug":
            cache[lang] = load_model(lug_id, device, lang_hint=lug_hint)
        else:
            cache[lang] = load_model(WAXAL.get(lang, WAXAL["lug"]), device)
        logger.info("loaded %s -> %s", lang, cache[lang][2])
        return cache[lang]

    rows = []
    for _, r in tqdm(lid.iterrows(), total=len(lid), desc="mh-ft-pool"):
        uid = str(r.ID)
        lid_lang = str(r.lang1)
        p1 = float(r.p1)
        arr, sr = load_wav(AUDIO / f"{uid}.wav")
        if lid_lang == "luo":
            cands = ["ach", "lug", "sog"]
        elif lid_lang == "lug":
            cands = ["lug", "nyn", "sog"]
        elif lid_lang in WAXAL:
            cands = [lid_lang]
        else:
            cands = ["lug", "ach"]

        best_hyp, best_lang, best_conf, best_src = ".", cands[0], -1e9, "?"
        for lang in cands:
            m, p, tag = get(lang)
            hyp, conf = transcribe_waveform(m, p, arr, sr, device=device, return_confidence=True)
            hyp = normalize_text(hyp) or "."
            if conf > best_conf:
                best_hyp, best_lang, best_conf, best_src = hyp, lang, conf, tag
        rows.append(
            {
                "ID": uid,
                "prediction": best_hyp,
                "lid_lang": lid_lang,
                "p1": p1,
                "decode_lang": best_lang,
                "confidence": best_conf,
                "source": best_src,
                "candidates": "|".join(cands),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.detail, index=False)
    build_submission(df[["ID", "prediction"]], sample_path=SAMPLE, out_path=args.out)
    rep = check_submission(args.out, SAMPLE)
    o = pd.read_csv(PROJECT_ROOT / "submission_phase2_openset.csv").set_index("ID")["Target"].astype(str)
    n = df.set_index("ID")["prediction"].astype(str)
    rep.update(
        {
            "n_changed": int((o.reindex(n.index) != n).sum()),
            "decode_top": Counter(df.decode_lang).most_common(),
            "source_counts": Counter(df.source).most_common(),
            "ach_model": ach_id,
            "lug_model": lug_id,
            "method": "multihyp_same_family_ft_pool",
        }
    )
    (OUTPUT_DIR / "phase2_multihyp_ft_pool_check.json").write_text(json.dumps(rep, indent=2, default=str))
    logger.info("%s", rep)
    print("UPLOAD", args.out)
    print("CHANGED", rep["n_changed"])


if __name__ == "__main__":
    main()
