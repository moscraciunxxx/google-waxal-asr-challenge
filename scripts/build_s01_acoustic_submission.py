#!/usr/bin/env python3
"""S01 end-to-end: calibrate acoustic mass thr on open val, score residual, pack CSV.

Uses MMS-LID-126 waveform mass_luo/mass_ach (scripts/luo_acoustic_router.AcousticRouter).
Open calib: FLEURS luo_ke validation + WAXAL ach_asr validation (no test gold).
Phase-2: residual lid=luo ∩ decode=ach \\ dual15; hyp from dual-pool mms / mms1b.
Floor-first pack; does NOT overwrite FINAL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORBIDDEN_TRAIN_SPLITS, SEED, TARGET_SR
from src.prize_pack import align_ids, apply_replace_set, diff_stats, is_banned_mass_rewrite, length_guard_ok
from src.s01_acoustic_gate import AcousticMass, S01Thresholds, calibrate_thresholds, s01_accept
from src.text_norm import normalize_text
from src.torch_env import pick_torch_device

# reuse router class
from scripts.luo_acoustic_router import AcousticRouter, load_wav, set_seed


def _array_from_hf(ex) -> tuple[np.ndarray, int]:
    """Decode HF audio without torchcodec (use soundfile/librosa on path or bytes)."""
    import io

    import soundfile as sf

    audio = ex["audio"]
    if isinstance(audio, dict) and audio.get("array") is not None:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate") or TARGET_SR)
    elif isinstance(audio, dict) and audio.get("bytes") is not None:
        # prefer embedded bytes (streaming parquet) over non-local path stubs
        arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        sr = int(sr)
    elif isinstance(audio, dict) and audio.get("path") and Path(str(audio["path"])).exists():
        arr, sr = sf.read(str(audio["path"]), dtype="float32", always_2d=False)
        sr = int(sr)
    else:
        raise ValueError(
            f"unsupported audio payload keys={list(audio) if isinstance(audio, dict) else type(audio)} "
            f"path_exists={Path(str(audio.get('path'))).exists() if isinstance(audio, dict) and audio.get('path') else None}"
        )
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    if peak > 0:
        arr = arr / peak
    return arr, sr


def score_open_calib(router: AcousticRouter, max_luo: int, max_ach: int, seed: int) -> tuple[list[AcousticMass], list[AcousticMass], dict]:
    """Stream open val only (no unlabeled bulk download). Decode audio via soundfile."""
    from datasets import Audio, load_dataset

    set_seed(seed)
    meta: dict = {
        "source_luo": "google/fleurs:luo_ke:validation:streaming",
        "source_ach": "google/WaxalNLP:ach_asr:validation:streaming",
    }
    # decode=False → path/bytes for soundfile (no torchcodec)
    luo_stream = load_dataset("google/fleurs", "luo_ke", split="validation", streaming=True)
    ach_stream = load_dataset("google/WaxalNLP", "ach_asr", split="validation", streaming=True)
    luo_stream = luo_stream.cast_column("audio", Audio(decode=False))
    ach_stream = ach_stream.cast_column("audio", Audio(decode=False))

    luo_masses: list[AcousticMass] = []
    for i, ex in enumerate(luo_stream):
        if i >= max_luo:
            break
        arr, sr = _array_from_hf(ex)
        r = router.route(arr, sr)
        luo_masses.append(
            AcousticMass(
                r["mass_luo"], r["mass_ach"], r["mass_bantu"], r.get("mass_other", 0.0), r["top1_lang"], r["top1_p"]
            )
        )
    ach_masses: list[AcousticMass] = []
    for i, ex in enumerate(ach_stream):
        if i >= max_ach:
            break
        arr, sr = _array_from_hf(ex)
        r = router.route(arr, sr)
        ach_masses.append(
            AcousticMass(
                r["mass_luo"], r["mass_ach"], r["mass_bantu"], r.get("mass_other", 0.0), r["top1_lang"], r["top1_p"]
            )
        )
    meta["n_luo_scored"] = len(luo_masses)
    meta["n_ach_scored"] = len(ach_masses)
    return luo_masses, ach_masses, meta


def residual_ids() -> list[str]:
    lid = {r["ID"]: r for r in csv.DictReader((ROOT / "outputs/phase2_lid126_full.csv").open())}
    det = {r["ID"]: r for r in csv.DictReader((ROOT / "outputs/phase2_openset_detail.csv").open())}
    pool = {r["ID"]: r for r in csv.DictReader((ROOT / "outputs/phase2_achluo_dual_pool_detail.csv").open())}
    dual = {
        sid
        for sid, r in pool.items()
        if str(r.get("already_dual", "")).lower() in ("1", "true", "yes")
    }
    out = []
    for sid, L in lid.items():
        if (L.get("lang1") or "").lower() != "luo":
            continue
        if (det[sid].get("decode_lang") or "").lower() != "ach":
            continue
        if sid in dual:
            continue
        out.append(sid)
    return out


def score_residual(
    router: AcousticRouter,
    ids: list[str],
    audio_dir: Path,
    thr: S01Thresholds,
    *,
    max_n: int,
) -> tuple[list[dict], list[dict]]:
    """Return (all scored rows, replace dicts for packing)."""
    floor = {
        r["ID"]: normalize_text(r["Target"])
        for r in csv.DictReader((ROOT / "submission_phase2_FINAL.csv").open())
    }
    pool = {r["ID"]: r for r in csv.DictReader((ROOT / "outputs/phase2_achluo_dual_pool_detail.csv").open())}
    mms1b = {r["ID"]: r for r in csv.DictReader((ROOT / "outputs/phase2_luo_mms1b_detail.csv").open())}

    scored: list[dict] = []
    accepts: list[tuple[float, dict]] = []
    for sid in ids:
        path = audio_dir / f"{sid}.wav"
        if not path.exists():
            continue
        arr, sr = load_wav(path)
        r = router.route(arr, sr)
        mass = AcousticMass(r["mass_luo"], r["mass_ach"], r["mass_bantu"], r.get("mass_other", 0.0), r["top1_lang"], r["top1_p"])
        ok, reason = s01_accept(mass, thr)
        hyp = normalize_text(pool.get(sid, {}).get("mms") or "") or normalize_text(
            mms1b.get(sid, {}).get("prediction") or ""
        )
        fl = floor[sid]
        row = {
            "ID": sid,
            "mass_luo": mass.mass_luo,
            "mass_ach": mass.mass_ach,
            "mass_bantu": mass.mass_bantu,
            "top1_lang": mass.top1_lang,
            "top1_p": mass.top1_p,
            "s01_accept": ok,
            "s01_reason": reason,
            "luo_hyp": hyp,
            "floor_hyp": fl,
        }
        scored.append(row)
        if not ok:
            continue
        if not hyp or hyp == fl or hyp == ".":
            continue
        if not length_guard_ok(fl, hyp):
            continue
        # score for ranking: margin then mass_luo
        score = (mass.mass_luo - mass.mass_ach) + 0.1 * mass.mass_luo
        accepts.append(
            (
                score,
                {
                    "ID": sid,
                    "own_hyp": hyp,
                    "floor_hyp": fl,
                    "reason": f"S01_acoustic:{reason}",
                    "signals": "S01",
                    "mass_luo": mass.mass_luo,
                    "mass_ach": mass.mass_ach,
                    "mass_bantu": mass.mass_bantu,
                    "score": score,
                },
            )
        )
    accepts.sort(key=lambda x: -x[0])
    replaces = [r for _, r in accepts[: max(0, max_n)]]
    return scored, replaces


def main() -> int:
    assert "test" in FORBIDDEN_TRAIN_SPLITS
    p = argparse.ArgumentParser()
    p.add_argument("--max-luo-calib", type=int, default=60)
    p.add_argument("--max-ach-calib", type=int, default=80)
    p.add_argument("--max-fpr", type=float, default=0.05)
    p.add_argument("--max-n", type=int, default=25)
    p.add_argument("--max-residual", type=int, default=0, help="0=all residual")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-calib", action="store_true", help="use fail-closed default thr (debug)")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/new_signals/submission_phase2_s01_acoustic.csv",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_torch_device()
    print(json.dumps({"device": str(device), "torch": torch.__version__}))
    router = AcousticRouter(device)

    calib_meta: dict = {}
    if args.skip_calib:
        # Debug only: use true zero-accept thr (not a soft thr that still fires residual)
        thr = S01Thresholds(1.01, -0.01, 1.01, -0.01)
        calib_meta = {
            "note": "skip_calib_zero_accept_thr",
            "thr": thr.as_dict(),
            "candidates_ok": 0,
            "zero_accept": True,
        }
    else:
        print("calibrating on open FLEURS luo + WAXAL ach...")
        luo_m, ach_m, cmeta = score_open_calib(router, args.max_luo_calib, args.max_ach_calib, args.seed)
        thr, stats = calibrate_thresholds(luo_m, ach_m, max_fpr=args.max_fpr)
        calib_meta = {**cmeta, **stats, "thr": thr.as_dict()}
        print(
            json.dumps(
                {
                    "calib": {
                        k: calib_meta[k]
                        for k in (
                            "tpr",
                            "fpr",
                            "n_luo",
                            "n_ach",
                            "note",
                            "thr",
                            "candidates_ok",
                            "zero_accept",
                        )
                        if k in calib_meta
                    }
                },
                indent=2,
            )
        )

    ids = residual_ids()
    if args.max_residual and args.max_residual > 0:
        ids = ids[: args.max_residual]
    audio_dir = ROOT / "data/phase2/audio"

    # Criterion 1: when open-val cannot meet max_fpr, fail closed → zero replaces (floor-only).
    zero_accept = bool(calib_meta.get("zero_accept")) or int(calib_meta.get("candidates_ok", -1)) == 0
    if zero_accept and not args.skip_calib:
        print("FAIL_CLOSED: open-val inseparable → packing floor-only (n_replace=0); residual not rescored for replace")
        # optional: score a tiny residual sample to prove thr rejects residual operating point
        probe_ids = ids[: min(5, len(ids))]
        scored, _ = score_residual(router, probe_ids, audio_dir, thr, max_n=0)
        replaces = []
        n_raw = sum(1 for r in scored if r["s01_accept"])
        print(f"s01_accept_raw_probe={n_raw}/{len(scored)} (expect 0) replaces=0")
        if n_raw != 0:
            raise RuntimeError("fail_closed thr accepted residual probe; thr is not zero-accept")
    else:
        print(f"scoring residual n={len(ids)} ...")
        scored, replaces = score_residual(router, ids, audio_dir, thr, max_n=args.max_n)
        print(f"s01_accept_raw={sum(1 for r in scored if r['s01_accept'])} replaces={len(replaces)}")

    floor_path = ROOT / "submission_phase2_FINAL.csv"
    fail_path = ROOT / "outputs/legit_system/phase2_multifamily_fusion.csv"
    sample = ROOT / "data/phase2/SampleSubmission.csv"
    floor = {r["ID"]: normalize_text(r["Target"]) for r in csv.DictReader(floor_path.open())}
    sample_ids = [r["ID"] for r in csv.DictReader(sample.open())]
    targets = apply_replace_set(floor, replaces)
    rows = align_ids(sample_ids, targets)
    st = diff_stats(floor, targets, sample_ids)
    fail = {r["ID"]: normalize_text(r["Target"]) for r in csv.DictReader(fail_path.open())}

    meta = {
        "out": str(args.out),
        "n_replace": st["n_diff"],
        "n_same_floor": st["n_same"],
        "n_same_failed_multifamily": sum(1 for i in sample_ids if targets[i] == fail[i]),
        "mass_rewrite": is_banned_mass_rewrite(st["n_diff"]),
        "floor_sha256": hashlib.sha256(floor_path.read_bytes()).hexdigest(),
        "failed_mf_sha256": hashlib.sha256(fail_path.read_bytes()).hexdigest(),
        "method": (
            "S01 acoustic MMS-LID-126 family mass gate calibrated on open FLEURS luo_ke val "
            "+ WAXAL ach_asr val (max FPR); residual lid=luo∩decode=ach exclude dual15; "
            "overlay dual-pool mms / mms1b luo hyp only when thr accepts; floor-first. "
            "If open-val cannot meet max_fpr → fail_closed zero-accept floor-equivalent CSV."
        ),
        "signals": ["S01"],
        "s02_deferred": True,
        "s02_reason": "token/time budget — S01 complete first",
        "calibration": calib_meta,
        "n_residual_scored": len(scored),
        "n_s01_accept_raw": sum(1 for r in scored if r["s01_accept"]),
        "zero_accept_mode": bool(calib_meta.get("zero_accept")),
        "ban_check": {
            "verdict": "FAIL_mass" if is_banned_mass_rewrite(st["n_diff"]) else "PASS_ban_check",
            "bans_respected": [
                "acoustic mass gate not residual CTC conf primary",
                "not dual thr>0.15 alone",
                "not mass multi-adapter",
                "not blind all-luo",
                "not wordmerge/sna/corrector/S08-only rehash as sole lever",
                "no test gold train",
                "floor_default_except_replace_set",
                "fail_closed_zero_accept_when_open_val_inseparable",
            ],
            "n_replace": st["n_diff"],
        },
        "status": (
            "floor_equivalent_zero_accept_open_val_inseparable"
            if st["n_diff"] == 0 and calib_meta.get("zero_accept")
            else "candidate_ready_for_upload_awaiting_public_score"
        ),
        "win_condition": "public score > 0.560605696",
        "floor_public": 0.560605696,
        "public_win_claimed": False,
        "device": str(device),
    }

    if args.dry_run:
        print(json.dumps({**meta, "sample_replaces": replaces[:5]}, indent=2))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "Target"])
        w.writeheader()
        w.writerows(rows)
    meta["sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    meta["byte_identical_failed_mf"] = (
        hashlib.sha256(args.out.read_bytes()).digest() == hashlib.sha256(fail_path.read_bytes()).digest()
    )
    meta["byte_identical_floor"] = (
        hashlib.sha256(args.out.read_bytes()).digest() == hashlib.sha256(floor_path.read_bytes()).digest()
    )
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    args.out.with_name(args.out.stem + "_replaces.json").write_text(json.dumps(replaces, indent=2))
    # full residual scores for audit
    score_path = args.out.with_name(args.out.stem + "_residual_scores.csv")
    if scored:
        with score_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
            w.writeheader()
            w.writerows(scored)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
