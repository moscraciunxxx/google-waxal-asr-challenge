#!/usr/bin/env python3
"""Build sna-only ship candidate: v2_full with new sna (decode_lang==sna) redecoded by mubarak whisper.

Requires matched gate_pass for mubarak_whisper in
outputs/goal_2026_08_06/sna_matched_external_ab.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR
from src.text_norm import normalize_text
from scripts.run_phase2_openset import load_wav

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sna_mubarak_ship")

MUBARAK = "Mubarak127/waxal-whisper-large-v3-sna_asr"
GATE_JSON = ROOT / "outputs" / "goal_2026_08_06" / "sna_matched_external_ab.json"
NEW_ROUTES = ROOT / "outputs" / "next_iter" / "new_routes.csv"
NEW_AUDIO = ROOT / "newaudios"
V2 = ROOT / "submission_phase2_v2_full.csv"
DEFAULT_OUT = ROOT / "submission_phase2_sna_mubarak_gatepass.csv"
HYPS_OUT = ROOT / "outputs" / "goal_2026_08_06" / "hyps_sna_mubarak_whisper.csv"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def decode_one(model, proc, arr: np.ndarray, device, language: str = "sn") -> str:
    inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt")
    feats = inputs.input_features.to(device)
    try:
        forced = proc.get_decoder_prompt_ids(language=language, task="transcribe")
        ids = model.generate(feats, forced_decoder_ids=forced, max_new_tokens=128)
    except Exception:
        ids = model.generate(feats, max_new_tokens=128)
    return normalize_text(proc.batch_decode(ids, skip_special_tokens=True)[0]) or "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--hyps", type=Path, default=HYPS_OUT)
    ap.add_argument("--force", action="store_true", help="Ignore gate_pass false")
    ap.add_argument("--reuse-hyps", action="store_true", help="Reuse existing hyps CSV if complete")
    args = ap.parse_args()

    gate = json.loads(GATE_JSON.read_text())
    mub = gate.get("candidates", {}).get("mubarak_whisper", {})
    if not gate.get("gate_pass") or not mub.get("gate_pass"):
        if not args.force:
            raise SystemExit(
                f"Refuse ship: gate_pass={gate.get('gate_pass')} mubarak={mub}. Use --force to override."
            )
        logger.warning("FORCE ship despite gate_pass=false")

    routes = list(csv.DictReader(NEW_ROUTES.open()))
    sna_ids = [r["ID"] for r in routes if r["decode_lang"] == "sna"]
    logger.info("new sna ids=%d", len(sna_ids))

    hyps: dict[str, str] = {}
    if args.reuse_hyps and args.hyps.exists():
        for row in csv.DictReader(args.hyps.open()):
            hyps[row["ID"]] = normalize_text(row.get("Target") or row.get("hyp") or "") or "."
        logger.info("reused hyps %d", len(hyps))

    missing = [uid for uid in sna_ids if uid not in hyps]
    if missing:
        device = pick_device()
        logger.info("decoding %d sna with mubarak on %s", len(missing), device)
        proc = WhisperProcessor.from_pretrained(MUBARAK)
        model = WhisperForConditionalGeneration.from_pretrained(MUBARAK).to(device).eval()
        t0 = time.time()
        for k, uid in enumerate(missing):
            path = NEW_AUDIO / f"{uid}.wav"
            arr, sr = load_wav(path)
            arr = np.asarray(arr, dtype=np.float32)
            if sr != TARGET_SR:
                import librosa

                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
            peak = float(np.max(np.abs(arr)) + 1e-9)
            arr = arr / peak
            hyps[uid] = decode_one(model, proc, arr, device)
            if (k + 1) % 25 == 0 or (k + 1) == len(missing):
                logger.info(
                    "mubarak ship %d/%d %.1fs last=%s",
                    k + 1,
                    len(missing),
                    time.time() - t0,
                    hyps[uid][:60],
                )
            # checkpoint hyps every 50
            if (k + 1) % 50 == 0:
                args.hyps.parent.mkdir(parents=True, exist_ok=True)
                with args.hyps.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["ID", "Target"])
                    w.writeheader()
                    for i in sna_ids:
                        if i in hyps:
                            w.writerow({"ID": i, "Target": hyps[i]})
        del model
        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    # write full hyps
    args.hyps.parent.mkdir(parents=True, exist_ok=True)
    with args.hyps.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "Target"])
        w.writeheader()
        for uid in sna_ids:
            w.writerow({"ID": uid, "Target": hyps[uid]})

    # merge into v2
    base = list(csv.DictReader(V2.open()))
    assert len(base) == 2392
    n_changed = 0
    out_rows = []
    for row in base:
        uid = row["ID"]
        if uid in hyps:
            new_t = hyps[uid]
            old_t = normalize_text(row["Target"]) or "."
            if new_t != old_t:
                n_changed += 1
            out_rows.append({"ID": uid, "Target": new_t})
        else:
            out_rows.append({"ID": uid, "Target": normalize_text(row["Target"]) or "."})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "Target"])
        w.writeheader()
        w.writerows(out_rows)

    meta = {
        "base": str(V2),
        "out": str(args.out),
        "hyps": str(args.hyps),
        "model": MUBARAK,
        "n_sna_decoded": len(sna_ids),
        "n_changed_vs_v2": n_changed,
        "rows": len(out_rows),
        "gate": {
            "gate_pass": gate.get("gate_pass"),
            "ship_tag": gate.get("ship_tag"),
            "mubarak": mub,
            "delta_zindi_vs_floor": gate.get("candidates", {})
            .get("mubarak_whisper", {})
            .get("delta_zindi_vs_floor"),
        },
        "scope": "newclips decode_lang==sna only (445); old clips unchanged",
    }
    meta_path = ROOT / "outputs" / "goal_2026_08_06" / "sna_mubarak_ship_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s changed=%d meta=%s", args.out, n_changed, meta_path)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
