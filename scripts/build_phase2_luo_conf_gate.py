#!/usr/bin/env python3
"""NEW method: CTC confidence gate for residual Luo (not dual-CER thr expand).

Public bans we do NOT use:
  - margin re-route
  - decode_lang==lug rewrite (FT-v4 / KenLM)
  - dual thr>0.15 CER(MMS,CLEAR)
  - blind all-LID=luo

Recipe:
  base = public floor selective_v3_dual15
  pool = lid_lang==luo & decode_lang==ach & NOT already dual-overlay
  for each pool clip:
    run MMS-1B adapter luo  → (text_luo, conf_luo)
    run MMS-1B adapter ach  → (text_ach, conf_ach)
    if conf_luo - conf_ach >= margin: take text_luo else keep floor
  margin calibrated on FLEURS luo_ke (true Luo) vs WAXAL ach proxy (false Luo)

Open-source only; no Phase-1 test gold.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import io
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
from src.metrics import compute_cer, compute_wer
from src.dataset import load_hf_asr_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("luo_conf_gate")

AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
FLOOR = PROJECT_ROOT / "submission_phase2_selective_v3_dual15.csv"
SEL_V3 = PROJECT_ROOT / "submission_phase2_selective_v3.csv"
DETAIL = OUTPUT_DIR / "phase2_selective_v3_detail.csv"


def load_wav(path: Path | str | bytes | dict):
    if isinstance(path, dict):
        if path.get("bytes") is not None:
            src = io.BytesIO(path["bytes"])
        else:
            src = str(path.get("path") or path)
    elif isinstance(path, (bytes, bytearray)):
        src = io.BytesIO(path)
    else:
        src = str(path)
    arr, sr = sf.read(src, dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


def score_pair(model, processor, device, arr, sr):
    set_lang(model, processor, "luo")
    text_l, conf_l = transcribe_waveform(
        model, processor, arr, sr, device=device, return_confidence=True
    )
    set_lang(model, processor, "ach")
    text_a, conf_a = transcribe_waveform(
        model, processor, arr, sr, device=device, return_confidence=True
    )
    return (
        normalize_text(text_l) or ".",
        float(conf_l),
        normalize_text(text_a) or ".",
        float(conf_a),
    )


def zindi(refs, hyps):
    w = float(compute_wer(refs, hyps))
    c = float(compute_cer(refs, hyps))
    return 1.0 - 0.5 * w - 0.5 * c, w, c


def calibrate_margin(model, processor, device, max_luo=80, max_ach=40, seed=42):
    """Return recommended margin and offline metrics."""
    rng = np.random.default_rng(seed)
    rows = []

    # True Luo: FLEURS luo_ke validation
    try:
        from datasets import Audio, load_dataset

        ds = load_dataset("google/fleurs", "luo_ke", split="validation")
        # soundfile path (avoid torchcodec)
        ds = ds.cast_column("audio", Audio(decode=False))
        n = min(max_luo, len(ds))
        idx = rng.choice(len(ds), size=n, replace=False)
        for i in tqdm(idx, desc="calib-fleurs-luo"):
            ex = ds[int(i)]
            aud = ex["audio"]
            arr, sr = load_wav(aud if isinstance(aud, dict) else Path(str(aud)))
            ref = normalize_text(ex.get("transcription") or ex.get("raw_transcription") or "")
            tl, cl, ta, ca = score_pair(model, processor, device, arr, sr)
            rows.append(
                {
                    "domain": "fleurs_luo",
                    "true_luo": 1,
                    "ref": ref,
                    "text_luo": tl,
                    "conf_luo": cl,
                    "text_ach": ta,
                    "conf_ach": ca,
                    "delta": cl - ca,
                }
            )
    except Exception as e:
        logger.warning("FLEURS calib failed: %s", e)

    # False Luo: WAXAL ach validation (HF)
    try:
        proxy = pd.read_csv(PROJECT_ROOT / "data" / "proxy_val_index.csv")
        ach_ids = set(proxy.loc[proxy.language == "ach", "id"].astype(str))
        ds_ach = load_hf_asr_split("ach", "validation", max_samples=None)
        taken = 0
        for i in tqdm(range(len(ds_ach)), desc="calib-ach"):
            if taken >= max_ach:
                break
            ex = ds_ach[i]
            eid = str(ex.get("id") or "")
            if ach_ids and eid not in ach_ids and len(ach_ids) > 5:
                continue
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"]["sampling_rate"])
            ref = normalize_text(ex.get("transcription") or "")
            tl, cl, ta, ca = score_pair(model, processor, device, arr, sr)
            rows.append(
                {
                    "domain": "waxal_ach",
                    "true_luo": 0,
                    "ref": ref,
                    "text_luo": tl,
                    "conf_luo": cl,
                    "text_ach": ta,
                    "conf_ach": ca,
                    "delta": cl - ca,
                }
            )
            taken += 1
    except Exception as e:
        logger.warning("Ach calib failed: %s", e)

    cdf = pd.DataFrame(rows)
    if cdf.empty:
        return 0.02, {"error": "empty calib"}, cdf

    # Grid margin: maximize TPR@FPR constraints; prefer low FPR on ach
    best = None
    grid = []
    for m in np.linspace(-0.05, 0.15, 41):
        pred = (cdf["delta"] >= m).astype(int)
        luo = cdf[cdf.true_luo == 1]
        ach_ = cdf[cdf.true_luo == 0]
        tpr = float(pred[cdf.true_luo == 1].mean()) if len(luo) else 0.0
        fpr = float(pred[cdf.true_luo == 0].mean()) if len(ach_) else 0.0
        # zindi if we pick luo text when pred else ach text
        hyps = [
            r.text_luo if p else r.text_ach
            for r, p in zip(cdf.itertuples(), pred)
        ]
        refs = cdf["ref"].tolist()
        # only score rows with non-empty ref
        mask = [bool(x) for x in refs]
        if sum(mask) >= 10:
            z, w, c = zindi([r for r, msk in zip(refs, mask) if msk], [h for h, msk in zip(hyps, mask) if msk])
        else:
            z, w, c = 0.0, 1.0, 1.0
        rec = {"margin": float(m), "tpr": tpr, "fpr": fpr, "zindi": z, "wer": w, "cer": c, "n_accept": int(pred.sum())}
        grid.append(rec)
        # primary: fpr <= 0.10, max tpr; secondary max zindi
        if fpr <= 0.10:
            key = (tpr, z)
            if best is None or key > best[0]:
                best = (key, rec)
    if best is None:
        # fall back lowest fpr then highest tpr
        grid_sorted = sorted(grid, key=lambda r: (r["fpr"], -r["tpr"], -r["zindi"]))
        rec = grid_sorted[0]
    else:
        rec = best[1]

    report = {
        "n_luo": int((cdf.true_luo == 1).sum()),
        "n_ach": int((cdf.true_luo == 0).sum()),
        "delta_luo_mean": float(cdf.loc[cdf.true_luo == 1, "delta"].mean()) if (cdf.true_luo == 1).any() else None,
        "delta_ach_mean": float(cdf.loc[cdf.true_luo == 0, "delta"].mean()) if (cdf.true_luo == 0).any() else None,
        "chosen": rec,
        "grid_top": sorted(grid, key=lambda r: (-r["tpr"], r["fpr"]))[:8],
    }
    return float(rec["margin"]), report, cdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=None, help="If set, skip calib grid and use this")
    ap.add_argument("--calib-only", action="store_true")
    ap.add_argument("--max-calib-luo", type=int, default=60)
    ap.add_argument("--max-calib-ach", type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--cache",
        type=Path,
        default=OUTPUT_DIR / "phase2_luo_conf_gate_scores.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "submission_phase2_selective_v3_dual15_luoconf.csv",
    )
    ap.add_argument(
        "--detail",
        type=Path,
        default=OUTPUT_DIR / "phase2_luo_conf_gate_detail.csv",
    )
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    model, processor, device = load_mms(device=device)

    margin = args.margin
    calib_report = None
    if margin is None:
        margin, calib_report, cdf = calibrate_margin(
            model, processor, device, max_luo=args.max_calib_luo, max_ach=args.max_calib_ach
        )
        cdf.to_csv(OUTPUT_DIR / "phase2_luo_conf_gate_calib.csv", index=False)
        (OUTPUT_DIR / "phase2_luo_conf_gate_calib.json").write_text(
            json.dumps(calib_report, indent=2, default=str)
        )
        logger.info("calibrated margin=%.4f report=%s", margin, calib_report.get("chosen"))
    if args.calib_only:
        print(json.dumps({"margin": margin, "calib": calib_report}, indent=2, default=str))
        return

    floor = pd.read_csv(FLOOR)
    v3 = pd.read_csv(SEL_V3)
    det = pd.read_csv(DETAIL)
    dual_ids = set(floor.loc[floor.Target.values != v3.Target.values, "ID"].astype(str))
    frozen_lug = set(det.loc[det.decode_lang == "lug", "ID"].astype(str))

    pool = det[
        (det.lid_lang == "luo")
        & (det.decode_lang == "ach")
        & (~det.ID.astype(str).isin(dual_ids))
    ].copy()
    logger.info("residual pool n=%d (ach-route luo, not dual)", len(pool))

    # score cache — batch by adapter (set_lang once each) to avoid per-clip adapter downloads
    if args.cache.exists():
        scores = pd.read_csv(args.cache)
        logger.info("loaded score cache %s n=%d", args.cache, len(scores))
        have = set(scores.ID.astype(str))
    else:
        scores = pd.DataFrame(columns=["ID", "text_luo", "conf_luo", "text_ach", "conf_ach", "delta"])
        have = set()

    todo = [u for u in pool.ID.astype(str).tolist() if u not in have]
    logger.info("need scores for %d residual clips", len(todo))
    if todo:
        waves = {}
        for uid in tqdm(todo, desc="load-wav"):
            waves[uid] = load_wav(AUDIO / f"{uid}.wav")
        luo_map: dict[str, tuple[str, float]] = {}
        set_lang(model, processor, "luo")
        for uid in tqdm(todo, desc="score-luo"):
            arr, sr = waves[uid]
            text, conf = transcribe_waveform(
                model, processor, arr, sr, device=device, return_confidence=True
            )
            luo_map[uid] = (normalize_text(text) or ".", float(conf))
        ach_map: dict[str, tuple[str, float]] = {}
        set_lang(model, processor, "ach")
        for uid in tqdm(todo, desc="score-ach"):
            arr, sr = waves[uid]
            text, conf = transcribe_waveform(
                model, processor, arr, sr, device=device, return_confidence=True
            )
            ach_map[uid] = (normalize_text(text) or ".", float(conf))
        new_rows = []
        for uid in todo:
            tl, cl = luo_map[uid]
            ta, ca = ach_map[uid]
            new_rows.append(
                {
                    "ID": uid,
                    "text_luo": tl,
                    "conf_luo": cl,
                    "text_ach": ta,
                    "conf_ach": ca,
                    "delta": cl - ca,
                }
            )
        scores = pd.concat([scores, pd.DataFrame(new_rows)], ignore_index=True)
        scores.to_csv(args.cache, index=False)
    scores["ID"] = scores["ID"].astype(str)

    # build submission variants at several margins
    floor_map = floor.set_index("ID")["Target"].astype(str)
    smap = scores.set_index("ID")

    variants = {}
    for m in sorted({margin, 0.0, 0.01, 0.02, 0.03, 0.05, 0.08}):
        out_rows = []
        n_acc = 0
        for uid, ft in floor_map.items():
            src = "floor_keep"
            hyp = ft
            if uid in smap.index:
                r = smap.loc[uid]
                if float(r.delta) >= m:
                    hyp = str(r.text_luo)
                    src = "mms_luo_conf_gate"
                    n_acc += 1
            # never touch frozen lug
            if uid in frozen_lug and hyp != ft:
                raise RuntimeError(f"frozen lug touched {uid}")
            out_rows.append({"ID": uid, "Target": hyp, "source": src, "delta": float(smap.loc[uid].delta) if uid in smap.index else None})
        df = pd.DataFrame(out_rows)
        tag = f"m{str(m).replace('-', 'n').replace('.', 'p')}"
        path = PROJECT_ROOT / f"submission_phase2_selective_v3_dual15_luoconf_{tag}.csv"
        sub = df[["ID", "Target"]]
        sub.to_csv(path, index=False)
        rep = check_submission(path, SAMPLE)
        n_changed = int((sub.set_index("ID")["Target"] != floor.set_index("ID")["Target"]).sum())
        variants[tag] = {
            "path": str(path),
            "margin": m,
            "n_accept": n_acc,
            "n_changed_vs_floor": n_changed,
            "check_ok": rep.get("ok", True) if isinstance(rep, dict) else True,
        }
        logger.info("variant %s n_changed=%d", tag, n_changed)

    # primary out = calibrated margin
    primary = PROJECT_ROOT / "submission_phase2_selective_v3_dual15_luoconf.csv"
    # find matching variant
    best_tag = None
    for tag, v in variants.items():
        if abs(v["margin"] - margin) < 1e-9:
            best_tag = tag
            break
    if best_tag is None:
        best_tag = min(variants.keys(), key=lambda t: abs(variants[t]["margin"] - margin))
    primary.write_text(Path(variants[best_tag]["path"]).read_text())
    # detail
    det_out = floor[["ID"]].copy()
    det_out = det_out.merge(scores, on="ID", how="left")
    det_out["floor"] = floor["Target"].values
    det_out["accept"] = det_out["delta"].fillna(-1e9) >= margin
    det_out["prediction"] = np.where(det_out["accept"], det_out["text_luo"], det_out["floor"])
    det_out["source"] = np.where(det_out["accept"], "mms_luo_conf_gate", "floor_keep")
    det_out.to_csv(args.detail, index=False)

    summary = {
        "method": "floor + residual ach-route luo CTC conf gate (conf_luo - conf_ach >= margin)",
        "margin": margin,
        "calib": calib_report,
        "pool_n": int(len(pool)),
        "variants": variants,
        "primary": str(primary),
        "n_changed_primary": variants[best_tag]["n_changed_vs_floor"],
        "frozen_lug_touched": 0,
        "banned_levers_avoided": [
            "margin_reroute",
            "lug_rewrite",
            "dual_thr_gt_0.15",
            "blind_all_luo",
        ],
    }
    (OUTPUT_DIR / "phase2_luo_conf_gate_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(json.dumps(summary, indent=2, default=str))
    print("PRIMARY", primary, "n_changed", variants[best_tag]["n_changed_vs_floor"])


if __name__ == "__main__":
    main()
