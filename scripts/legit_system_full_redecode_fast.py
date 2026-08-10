#!/usr/bin/env python3
"""Full Phase-2 multi-family fusion redecode — FAST path.

Reuses openset WAXALNet MMS multi-hyp predictions (all 1500) as the MMS family,
runs Whisper family on each clip (or uses cache), fuses with legit_fusion.fuse_row.

This is full multi-family fusion for every ID — not residual conf thr surgery on floor.
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
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT, SEED, TARGET_SR
from src.legit_fusion import fuse_row, pack_fusion_submission_row
from src.legit_system_pack import resolve_decode_lang, whisper_language_name
from src.mms_infer import pick_device
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("full_redecode_fast")

PHASE2_AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
PHASE2_SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
LID126 = OUTPUT_DIR / "phase2_lid126_full.csv"
OPENSET_DETAIL = OUTPUT_DIR / "phase2_openset_detail.csv"
DEFAULT_OUT = OUTPUT_DIR / "legit_system" / "phase2_multifamily_fusion.csv"
DEFAULT_DETAIL = OUTPUT_DIR / "legit_system" / "phase2_multifamily_fusion_detail.csv"


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    if peak > 0:
        arr = arr / peak
    return arr, int(sr)


@torch.inference_mode()
def whisper_decode(model, proc, array, sr, device, decode_lang: str, model_id: str) -> str:
    if sr != TARGET_SR:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
    inputs = proc(array, sampling_rate=TARGET_SR, return_tensors="pt")
    feats = inputs.input_features.to(device)
    gen_kwargs: dict = {"do_sample": False, "num_beams": 1}
    name = whisper_language_name(decode_lang)
    if name and "openai" in model_id:
        try:
            gen_kwargs["forced_decoder_ids"] = proc.get_decoder_prompt_ids(
                language=name, task="transcribe"
            )
        except Exception:
            pass
    ids = model.generate(feats, **gen_kwargs)
    return normalize_text(proc.batch_decode(ids, skip_special_tokens=True)[0]) or "."


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--whisper-model", default="openai/whisper-small")
    p.add_argument("--openset-detail", type=Path, default=OPENSET_DETAIL)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--detail-out", type=Path, default=DEFAULT_DETAIL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-whisper", action="store_true", help="Fusion uses MMS only (debug)")
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = pick_device()
    sample = pd.read_csv(PHASE2_SAMPLE)
    all_ids = sample["ID"].astype(str).tolist()
    ids = all_ids[args.offset :]
    if args.limit is not None:
        ids = ids[: args.limit]

    openset = pd.read_csv(args.openset_detail).set_index("ID")
    lid = pd.read_csv(LID126).set_index("ID")

    done = {}
    if args.resume and args.detail_out.exists():
        prev = pd.read_csv(args.detail_out)
        done = {str(r["ID"]): r for r in prev.to_dict("records")}
        logger.info("resume %d", len(done))

    model = proc = None
    mid = args.whisper_model
    if not args.skip_whisper:
        logger.info("Load whisper %s on %s", mid, device)
        try:
            proc = WhisperProcessor.from_pretrained(mid, local_files_only=True)
            model = WhisperForConditionalGeneration.from_pretrained(mid, local_files_only=True)
        except Exception:
            proc = WhisperProcessor.from_pretrained(mid)
            model = WhisperForConditionalGeneration.from_pretrained(mid)
        model.to(device).eval()
        model.config.forced_decoder_ids = None
        if hasattr(model, "generation_config"):
            model.generation_config.forced_decoder_ids = None

    detail = list(done.values())
    t0 = time.time()
    for i, sid in enumerate(ids):
        if sid in done:
            continue
        if sid not in openset.index:
            logger.warning("no openset row %s", sid)
            continue
        o = openset.loc[sid]
        mms_hyp = normalize_text(str(o["prediction"])) or "."
        mms_score = float(o["confidence"]) if "confidence" in o and pd.notna(o["confidence"]) else None
        decode_lang = str(o["decode_lang"]) if "decode_lang" in o else "lug"
        if sid in lid.index:
            lid_lang = str(lid.loc[sid]["lang1"])
            lid_p1 = float(lid.loc[sid]["p1"])
        else:
            lid_lang, lid_p1 = decode_lang, 0.0

        if args.skip_whisper:
            wh_hyp = mms_hyp
            wh_id = "skipped"
        else:
            path = PHASE2_AUDIO / f"{sid}.wav"
            arr, sr = load_wav(path)
            wh_hyp = whisper_decode(model, proc, arr, sr, device, decode_lang, mid)
            wh_id = mid

        fus = fuse_row(
            mms_hyp,
            wh_hyp,
            mms_score=mms_score,
            decode_lang=decode_lang,
            lid_lang=lid_lang,
            lid_p1=lid_p1,
        )
        row = {
            "ID": sid,
            "lid_lang": lid_lang,
            "lid_p1": lid_p1,
            "decode_lang": decode_lang,
            "mms_model_id": str(o.get("source", "waxal300_openset")),
            "mms_hyp": mms_hyp,
            "mms_score": mms_score,
            "whisper_model_id": wh_id,
            "whisper_hyp": wh_hyp,
            "fused_hyp": fus["fused_hyp"],
            "fusion_source": fus["fusion_source"],
            "fusion_reason": fus["fusion_reason"],
            "Target": fus["fused_hyp"],
        }
        detail.append(row)
        done[sid] = row
        if len(detail) % 20 == 0 or (i + 1) == len(ids):
            args.detail_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(detail).drop_duplicates("ID").to_csv(args.detail_out, index=False)
            logger.info("progress %d/%d elapsed=%.1fs", len(done), len(all_ids), time.time() - t0)

    # Build full 1500 submission in SampleSubmission order
    by_id = {str(r["ID"]): r for r in detail}
    # reload detail if needed
    if args.detail_out.exists():
        by_id.update({str(r["ID"]): r for r in pd.read_csv(args.detail_out).to_dict("records")})

    sub = []
    missing = []
    for sid in all_ids:
        if sid in by_id:
            sub.append(pack_fusion_submission_row(sid, by_id[sid].get("Target") or by_id[sid].get("fused_hyp")))
        else:
            missing.append(sid)
            sub.append(pack_fusion_submission_row(sid, "."))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sub)[["ID", "Target"]].to_csv(args.out, index=False)
    pd.DataFrame(list(by_id.values())).drop_duplicates("ID").to_csv(args.detail_out, index=False)
    meta = {
        "n_sample": len(all_ids),
        "n_fused": len(by_id),
        "missing": len(missing),
        "complete_1500": len(missing) == 0 and all(
            str(by_id[s].get("Target") or by_id[s].get("fused_hyp") or "").strip() not in ("",)
            for s in all_ids
            if s in by_id
        )
        and len(by_id) >= 1500,
        "mms_family": "phase2_openset_detail waxal300 multi-hyp (full redecode source)",
        "whisper_family": mid if not args.skip_whisper else "skipped",
        "fusion": "legit_fusion.fuse_row full-row multi-family",
        "not_residual_conf_surgery": True,
        "floor_untouched": True,
        "out": str(args.out),
        "detail": str(args.detail_out),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("meta %s", json.dumps(meta))
    return 0 if meta["complete_1500"] or args.limit is not None else (0 if not missing else 1)


if __name__ == "__main__":
    raise SystemExit(main())
