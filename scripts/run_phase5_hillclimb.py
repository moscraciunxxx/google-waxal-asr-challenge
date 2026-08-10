#!/usr/bin/env python3
"""Phase-5 hill-climb for Phase-2: margin multi-hyp + optional FT-lug.

Combines Phase-3/4 proxy winners into a deployable Phase-2 pipeline:

1. LID from outputs/phase2_lid126_full.csv (lang1, p1)
2. Candidate sets (same family waxal-300m only), primary = first cand:
   - luo → ach | lug | sog
   - lug → lug | nyn | sog
   - nyn → nyn | lug
   - sog → sog | lug
   - mas → mas | lug
   - else → resolve + lug + ach (as openset)
3. Decode all cands; if conf_best - conf_second >= --margin → best else **primary**
4. If final decode_lang == lug and p1 >= --ft-p1 and --use-ft:
   replace with mms-lug-ft-v2 (hard assign, no conf mix)

No KenLM (public toxic except selective proxy; skip for safety on full set).
No mms-1b conf mix.

Outputs:
  submission_phase2_phase5.csv
  outputs/phase5_hillclimb_detail.csv
  outputs/phase5_hillclimb_check.json
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase5_hillclimb")

PHASE2_AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
PHASE2_SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
LID_CSV = OUTPUT_DIR / "phase2_lid126_full.csv"

WAXAL300 = {
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
    "lin": "waxal-benchmarking/mms-300m-waxal-lin",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "ach": "waxal-benchmarking/mms-300m-waxal-ach",
    "sog": "waxal-benchmarking/mms-300m-waxal-sog",
    "nyn": "waxal-benchmarking/mms-300m-waxal-nyn",
    "mas": "waxal-benchmarking/mms-300m-waxal-mas",
}

FALLBACK = {
    "luo": ["ach", "lug", "sog"],
    "swh": ["lug", "lin"],
    "kin": ["nyn", "lug"],
    "nya": ["sna", "lug"],
    "nso": ["sna", "lug"],
    "umb": ["sog", "lug"],
    "wol": ["lug"],
    "eng": ["lug", "lin", "sna"],
}


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    if peak > 0:
        arr = arr / peak
    return arr, int(sr)


def candidates_for(lid: str) -> list[str]:
    if lid == "luo":
        return ["ach", "lug", "sog"]
    if lid == "lug":
        return ["lug", "nyn", "sog"]
    if lid == "nyn":
        return ["nyn", "lug"]
    if lid == "sog":
        return ["sog", "lug"]
    if lid == "mas":
        return ["mas", "lug"]
    if lid in WAXAL300:
        return [lid]
    cands = []
    for x in FALLBACK.get(lid, ["lug", "ach"]):
        if x in WAXAL300 and x not in cands:
            cands.append(x)
    if not cands:
        cands = ["lug", "ach"]
    return cands


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--use-ft", action="store_true", help="Hard FT-v2 when route is lug")
    p.add_argument("--ft-p1", type=float, default=0.95)
    p.add_argument("--device", default=None)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "submission_phase2_phase5.csv",
    )
    p.add_argument(
        "--detail",
        type=Path,
        default=OUTPUT_DIR / "phase5_hillclimb_detail.csv",
    )
    args = p.parse_args()

    if not LID_CSV.exists():
        raise SystemExit(f"Need {LID_CSV}")

    device = torch.device(args.device) if args.device else pick_device()
    logger.info(
        "device=%s margin=%.3f use_ft=%s ft_p1=%.2f",
        device,
        args.margin,
        args.use_ft,
        args.ft_p1,
    )

    lid_df = pd.read_csv(LID_CSV)
    lid_df["ID"] = lid_df["ID"].astype(str)
    if args.max_files:
        lid_df = lid_df.head(args.max_files)

    done: dict[str, dict] = {}
    if args.resume and args.detail.exists():
        prev = pd.read_csv(args.detail)
        for _, r in prev.iterrows():
            done[str(r.ID)] = r.to_dict()
        logger.info("resume %d", len(done))

    cache: dict[str, tuple] = {}

    def get_waxal(lang: str):
        if lang not in cache:
            mid = WAXAL300[lang]
            logger.info("load %s", mid)
            proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
            model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
            model.to(device).eval()
            cache[lang] = (model, proc)
        return cache[lang]

    ft = None
    if args.use_ft:
        from scripts.mms_adapter_ft import fix_mms_tokenizer

        ckpt = CHECKPOINT_DIR / "mms-lug-ft-v2"
        logger.info("load FT %s", ckpt)
        proc = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
        fix_mms_tokenizer(proc, "lug")
        model.to(device).eval()
        ft = (model, proc)

    rows: list[dict] = list(done.values())
    todo = lid_df[~lid_df.ID.isin(done)]
    for n, (_, r) in enumerate(tqdm(todo.iterrows(), total=len(todo), desc="phase5"), 1):
        uid = str(r.ID)
        lid_lang = str(r.lang1)
        p1 = float(r.p1)
        cands = candidates_for(lid_lang)
        primary = cands[0]
        arr, sr = load_wav(PHASE2_AUDIO / f"{uid}.wav")

        scored: list[tuple[str, str, float]] = []
        for lang in cands:
            if lang not in WAXAL300:
                continue
            m, proc = get_waxal(lang)
            hyp, conf = transcribe_waveform(
                m, proc, arr, sr, device=device, return_confidence=True
            )
            scored.append((normalize_text(hyp) or ".", lang, float(conf)))

        if not scored:
            hyp, dlang, conf, rule = ".", primary, -1e9, "empty"
        else:
            scored_sorted = sorted(scored, key=lambda x: x[2], reverse=True)
            best_h, best_l, best_c = scored_sorted[0]
            if len(scored_sorted) >= 2:
                second_c = scored_sorted[1][2]
                margin = best_c - second_c
            else:
                margin = 1e9
            if margin >= args.margin:
                hyp, dlang, conf, rule = best_h, best_l, best_c, "margin_best"
            else:
                # primary
                prim = next((x for x in scored if x[1] == primary), scored[0])
                hyp, dlang, conf, rule = prim[0], prim[1], prim[2], "margin_primary"

        src = f"waxal_{dlang}"
        if (
            args.use_ft
            and ft is not None
            and dlang == "lug"
            and p1 >= args.ft_p1
        ):
            hyp = normalize_text(
                transcribe_waveform(ft[0], ft[1], arr, sr, device=device)
            ) or "."
            src = "ft_v2"
            rule = rule + "+ft"

        rows.append(
            {
                "ID": uid,
                "prediction": hyp,
                "lid_lang": lid_lang,
                "p1": p1,
                "decode_lang": dlang if src != "ft_v2" else "lug",
                "confidence": conf,
                "source": src,
                "rule": rule,
                "candidates": "|".join(cands),
                "margin": float(margin) if scored else None,
            }
        )
        if n % 25 == 0 or n == len(todo):
            pd.DataFrame(rows).to_csv(args.detail, index=False)

    df = pd.DataFrame(rows)
    # restore LID order
    order = lid_df["ID"].tolist()
    df = df.set_index("ID").reindex(order).reset_index()
    df.to_csv(args.detail, index=False)

    build_submission(
        df[["ID", "prediction"]], sample_path=PHASE2_SAMPLE, out_path=args.out
    )
    report = check_submission(args.out, PHASE2_SAMPLE)
    report.update(
        {
            "margin": args.margin,
            "use_ft": args.use_ft,
            "ft_p1": args.ft_p1,
            "rule_counts": Counter(df.rule.astype(str)).most_common(),
            "source_counts": Counter(df.source.astype(str)).most_common(),
            "decode_top": Counter(df.decode_lang.astype(str)).most_common(10),
        }
    )
    # vs champion if present
    champ = PROJECT_ROOT / "submission_phase2_openset.csv"
    if champ.exists():
        o = pd.read_csv(champ).set_index("ID")["Target"].astype(str)
        npred = df.set_index("ID")["prediction"].astype(str)
        report["n_changed_vs_openset"] = int((o.reindex(npred.index) != npred).sum())
    (OUTPUT_DIR / "phase5_hillclimb_check.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    logger.info("CHECK %s", report)
    print("UPLOAD", args.out)
    print("RULES", report["rule_counts"][:8])
    print("CHANGED_VS_OPENSET", report.get("n_changed_vs_openset"))


if __name__ == "__main__":
    main()
