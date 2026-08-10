#!/usr/bin/env python3
"""NEW mechanism: same-family MMS-1B multi-adapter pick on lid=luo mass.

Unlike blind all-LID=luo (public reject ~0.511), for each lid=luo clip decode
with adapters {luo, ach, lug} under facebook/mms-1b-all and pick max CTC conf
(same thermometer). Overlay on public floor for non-luo / frozen paths as chosen.

Open-source only; no Phase-1 test gold; no cross-family conf mix.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT
from src.mms_infer import load_mms, pick_device, set_lang, transcribe_waveform
from src.submission import check_submission
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mms_multadapter_luo")

AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
FLOOR = PROJECT_ROOT / "submission_phase2_selective_v3_dual15.csv"
OPENSET = OUTPUT_DIR / "phase2_openset_detail.csv"
DEFAULT_ADAPTERS = ("luo", "ach", "lug")


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


def preload_adapters(model, processor, langs):
    for lang in langs:
        set_lang(model, processor, lang)
    logger.info("Preloaded adapters: %s", langs)


@torch.inference_mode()
def score_clip(model, processor, device, arr, sr, langs):
    best_text, best_conf, best_lang = ".", -1e9, langs[0]
    scores = {}
    for lang in langs:
        set_lang(model, processor, lang)
        text, conf = transcribe_waveform(
            model, processor, arr, sr, device=device, return_confidence=True
        )
        text = normalize_text(text) or "."
        scores[lang] = {"text": text, "conf": float(conf)}
        if conf > best_conf:
            best_text, best_conf, best_lang = text, float(conf), lang
    return best_text, best_conf, best_lang, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", type=str, default="luo,ach,lug")
    ap.add_argument("--limit", type=int, default=0, help="debug limit of luo clips")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="only replace floor if best conf - floor_lang_conf >= margin; "
                         "0 means always take multi-adapter pick for lid=luo")
    ap.add_argument("--require-pick-luo", action="store_true",
                    help="only overlay when multi-adapter picks luo")
    ap.add_argument("--name", type=str, default="mms_multadapter")
    ap.add_argument("--resume", type=Path, default=None, help="partial scores CSV")
    args = ap.parse_args()
    langs = tuple(a.strip() for a in args.adapters.split(",") if a.strip())

    floor = pd.read_csv(FLOOR)
    openset = pd.read_csv(OPENSET)
    luo_ids = openset.loc[openset.lid_lang == "luo", "ID"].astype(str).tolist()
    if args.limit > 0:
        luo_ids = luo_ids[: args.limit]
    logger.info("lid=luo clips: %d", len(luo_ids))

    scores_path = OUTPUT_DIR / f"phase2_{args.name}_scores.csv"
    done = {}
    if args.resume and Path(args.resume).exists():
        prev = pd.read_csv(args.resume)
        for _, r in prev.iterrows():
            done[str(r["ID"])] = r.to_dict()
        logger.info("Resume with %d done", len(done))
    elif scores_path.exists():
        prev = pd.read_csv(scores_path)
        for _, r in prev.iterrows():
            done[str(r["ID"])] = r.to_dict()
        logger.info("Resume existing scores %d", len(done))

    todo = [i for i in luo_ids if i not in done]
    if todo:
        model, processor, device = load_mms()
        preload_adapters(model, processor, langs)
        rows = list(done.values())
        t0 = time.time()
        for i, id_ in enumerate(tqdm(todo, desc="mms-multadapter")):
            wav = AUDIO / f"{id_}.wav"
            if not wav.exists():
                # try without assumption
                cands = list(AUDIO.glob(f"*{id_}*"))
                if not cands:
                    logger.warning("missing audio %s", id_)
                    continue
                wav = cands[0]
            arr, sr = load_wav(wav)
            text, conf, lang, scores = score_clip(model, processor, device, arr, sr, langs)
            row = {
                "ID": id_,
                "pick_lang": lang,
                "pick_conf": conf,
                "prediction": text,
            }
            for L in langs:
                row[f"text_{L}"] = scores[L]["text"]
                row[f"conf_{L}"] = scores[L]["conf"]
            rows.append(row)
            done[id_] = row
            if (i + 1) % 25 == 0:
                pd.DataFrame(rows).to_csv(scores_path, index=False)
                rate = (i + 1) / max(1e-9, time.time() - t0)
                logger.info("checkpoint %d rate=%.2f clips/s eta=%.1f min",
                            len(rows), rate, (len(todo) - i - 1) / max(rate, 1e-9) / 60)
        pd.DataFrame(rows).to_csv(scores_path, index=False)
    else:
        logger.info("All scores present at %s", scores_path)

    sc = pd.read_csv(scores_path)
    sc["ID"] = sc["ID"].astype(str)
    pick_counts = sc["pick_lang"].value_counts().to_dict()
    logger.info("pick_lang mass: %s", pick_counts)

    # Build overlay on floor
    out = floor.copy()
    out["ID"] = out["ID"].astype(str)
    smap = sc.set_index("ID")
    n_changed = 0
    n_pick_luo = 0
    for i, row in out.iterrows():
        id_ = row["ID"]
        if id_ not in smap.index:
            continue
        s = smap.loc[id_]
        if args.require_pick_luo and str(s["pick_lang"]) != "luo":
            continue
        new_t = normalize_text(str(s["prediction"])) or "."
        if new_t != str(row["Target"]):
            # optional margin vs conf_ach if present
            if args.min_margin > 0 and "conf_luo" in s and "conf_ach" in s:
                if float(s["conf_luo"]) - float(s["conf_ach"]) < args.min_margin:
                    continue
            out.at[i, "Target"] = new_t
            n_changed += 1
            if str(s["pick_lang"]) == "luo":
                n_pick_luo += 1

    sub_path = PROJECT_ROOT / f"submission_phase2_beat_k63_{args.name}.csv"
    out.to_csv(sub_path, index=False)
    check = check_submission(sub_path, PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv")
    rep = {
        "scores": str(scores_path),
        "submission": str(sub_path),
        "n_luo_scored": int(len(sc)),
        "pick_lang_mass": pick_counts,
        "n_changed_vs_floor": int(n_changed),
        "n_changed_pick_luo": int(n_pick_luo),
        "require_pick_luo": bool(args.require_pick_luo),
        "min_margin": args.min_margin,
        "adapters": list(langs),
        "check": check,
        "method": "floor + same-family MMS multi-adapter max-conf on lid=luo",
        "new_mechanism": True,
        "banned_levers_avoided": [
            "margin_primary_reroute",
            "blind_force_all_luo_adapter",
            "decode_lang_lug_rewrite_without_gate",
            "dual_thr_expand_alone",
        ],
    }
    rep_path = OUTPUT_DIR / f"phase2_{args.name}_check.json"
    rep_path.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
