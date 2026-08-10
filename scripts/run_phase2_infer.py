#!/usr/bin/env python3
"""Phase-2 audio-only inference: conf-route lin/sna/lug, optional FT re-decode.

Portal scoring now expects IDs like ID_TBDTM (1500 files in data/phase2/audio).
Phase-1 CSVs (lin_*/sna_*/lug_*) fail with: Missing entries for IDs ID_...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Prefer local HF cache — load_adapter must not hit the network per file.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, LANGUAGES, OUTPUT_DIR, PROJECT_ROOT, TARGET_SR
from src.mms_infer import load_mms, pick_device, set_lang, transcribe_waveform
from src.submission import build_submission, check_submission
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase2_infer")

PHASE2_DIR = PROJECT_ROOT / "data" / "phase2"
PHASE2_AUDIO = PHASE2_DIR / "audio"
PHASE2_SAMPLE = PHASE2_DIR / "SampleSubmission.csv"
DEFAULT_OUT = PROJECT_ROOT / "submission_phase2.csv"
DEFAULT_DETAIL = OUTPUT_DIR / "phase2_predictions_detail.csv"


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=-1)
    return np.asarray(arr, dtype=np.float32), int(sr)


def load_ft(lang: str, device: torch.device, ckpt_suffix: str = "ft-v2"):
    candidates = [
        CHECKPOINT_DIR / f"mms-{lang}-{ckpt_suffix}",
        CHECKPOINT_DIR / f"mms-{lang}-ft-v2",
        CHECKPOINT_DIR / f"mms-{lang}-ft",
    ]
    ckpt = next(
        (
            c
            for c in candidates
            if (c / "model.safetensors").exists() or (c / "pytorch_model.bin").exists()
        ),
        None,
    )
    if ckpt is None:
        return None, None
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    logger.info("Loading FT %s from %s", lang, ckpt)
    processor = AutoProcessor.from_pretrained(str(ckpt), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(ckpt), local_files_only=True)
    try:
        from scripts.mms_adapter_ft import fix_mms_tokenizer

        fix_mms_tokenizer(processor, lang)
    except Exception as e:
        logger.warning("fix_mms_tokenizer(%s): %s", lang, e)
    model.to(device)
    model.eval()
    return model, processor


def route_and_decode_zs(
    model,
    processor,
    array: np.ndarray,
    sr: int,
    device: torch.device,
    langs: tuple[str, ...] = LANGUAGES,
    _state: dict | None = None,
) -> tuple[str, str, float, dict[str, float]]:
    """Try each adapter; return best text, chosen lang, conf, per-lang confs."""
    best_text, best_lang, best_conf = ".", langs[0], -1e9
    confs: dict[str, float] = {}
    cur = (_state or {}).get("lang")
    for lang in langs:
        if cur != lang:
            set_lang(model, processor, lang)
            cur = lang
            if _state is not None:
                _state["lang"] = lang
        text, conf = transcribe_waveform(
            model, processor, array, sr, device=device, return_confidence=True
        )
        confs[lang] = conf
        if conf > best_conf:
            best_text, best_lang, best_conf = text, lang, conf
    return best_text, best_lang, best_conf, confs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Phase-2 audio-only MMS inference")
    p.add_argument("--audio-dir", type=Path, default=PHASE2_AUDIO)
    p.add_argument("--sample", type=Path, default=PHASE2_SAMPLE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--detail-out", type=Path, default=DEFAULT_DETAIL)
    p.add_argument("--max-files", type=int, default=None, help="smoke limit")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--ft-langs",
        nargs="*",
        default=["lug", "sna"],
        help="Re-decode with FT-v2 for these routed languages (lin usually ZS)",
    )
    p.add_argument("--ckpt-suffix", default="ft-v2")
    p.add_argument("--no-ft", action="store_true", help="ZS conf-routing only")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip IDs already in detail-out",
    )
    p.add_argument("--shard-every", type=int, default=50)
    args = p.parse_args(argv)

    audio_dir = Path(args.audio_dir)
    sample_path = Path(args.sample)
    if not audio_dir.is_dir():
        raise SystemExit(f"Missing audio dir: {audio_dir}")
    if not sample_path.is_file():
        raise SystemExit(f"Missing sample: {sample_path}")

    sample = pd.read_csv(sample_path)
    if "ID" not in sample.columns:
        raise SystemExit(f"{sample_path} missing ID column")
    want_ids = [str(x) for x in sample["ID"].tolist()]
    logger.info("SampleSubmission rows=%d", len(want_ids))

    wavs = {p.stem: p for p in sorted(audio_dir.glob("*.wav"))}
    missing_audio = [i for i in want_ids if i not in wavs]
    if missing_audio:
        raise SystemExit(
            f"{len(missing_audio)} sample IDs lack audio, e.g. {missing_audio[:5]}"
        )
    extra = set(wavs) - set(want_ids)
    if extra:
        logger.warning("%d extra wavs not in sample (ignored)", len(extra))

    ids = want_ids[: args.max_files] if args.max_files else want_ids
    device = torch.device(args.device) if args.device else pick_device()
    logger.info("device=%s n_files=%d", device, len(ids))

    detail_path = Path(args.detail_out)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if args.resume and detail_path.exists():
        prev = pd.read_csv(detail_path)
        for _, r in prev.iterrows():
            done[str(r["ID"])] = r.to_dict()
        logger.info("Resume: %d already done", len(done))

    model, processor, device = load_mms(device=device)
    # Warm-load all language adapters once (offline cache).
    for lang in LANGUAGES:
        set_lang(model, processor, lang)
    lang_state: dict = {"lang": LANGUAGES[-1]}
    rows: list[dict] = list(done.values()) if done else []
    t0 = time.time()
    todo = [i for i in ids if i not in done]
    logger.info("Decoding %d remaining files (ZS conf-routing)", len(todo))

    for n, uid in enumerate(tqdm(todo, desc="phase2-zs-route"), start=1):
        path = wavs[uid]
        try:
            arr, sr = load_wav(path)
            text, lang, conf, confs = route_and_decode_zs(
                model, processor, arr, sr, device, _state=lang_state
            )
            row = {
                "ID": uid,
                "prediction": text,
                "chosen_lang": lang,
                "confidence": conf,
                "conf_lin": confs.get("lin"),
                "conf_sna": confs.get("sna"),
                "conf_lug": confs.get("lug"),
                "source": "zs_route",
            }
        except Exception as e:
            logger.exception("Fail %s: %s", uid, e)
            row = {
                "ID": uid,
                "prediction": ".",
                "chosen_lang": "lin",
                "confidence": -1e9,
                "conf_lin": None,
                "conf_sna": None,
                "conf_lug": None,
                "source": f"error:{type(e).__name__}",
            }
        rows.append(row)
        done[uid] = row

        if n % args.shard_every == 0 or n == len(todo):
            pd.DataFrame(rows).to_csv(detail_path, index=False)
            elapsed = time.time() - t0
            rate = n / max(elapsed, 1e-6)
            logger.info(
                "checkpoint n=%d/%.0fs rate=%.2f/s eta=%.0fs",
                n,
                elapsed,
                rate,
                (len(todo) - n) / max(rate, 1e-6),
            )

    # Free ZS model before FT re-decode
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    detail = pd.DataFrame(rows)
    if not args.no_ft and args.ft_langs:
        ft_langs = [x for x in args.ft_langs if x in LANGUAGES]
        for lang in ft_langs:
            mask = detail["chosen_lang"].astype(str) == lang
            subset_ids = detail.loc[mask, "ID"].astype(str).tolist()
            if not subset_ids:
                logger.info("No files routed to %s — skip FT", lang)
                continue
            ft_model, ft_proc = load_ft(lang, device, ckpt_suffix=args.ckpt_suffix)
            if ft_model is None:
                logger.warning("No FT checkpoint for %s — keep ZS", lang)
                continue
            logger.info("FT re-decode %s n=%d", lang, len(subset_ids))
            for uid in tqdm(subset_ids, desc=f"ft-{lang}"):
                arr, sr = load_wav(wavs[uid])
                hyp = transcribe_waveform(ft_model, ft_proc, arr, sr, device=device)
                hyp = normalize_text(hyp) or "."
                idx = detail.index[detail["ID"].astype(str) == uid]
                detail.loc[idx, "prediction"] = hyp
                detail.loc[idx, "source"] = f"ft_{args.ckpt_suffix}"
            del ft_model
            if device.type == "mps":
                torch.mps.empty_cache()

    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)
    logger.info("Wrote detail %s", detail_path)

    # Build submission aligned to full sample (all 1500 even if --max-files)
    out = Path(args.out)
    preds = detail[["ID", "prediction"]].copy()
    # If smoke run, fill missing sample IDs with placeholder only for check of produced set
    if args.max_files:
        logger.warning(
            "Smoke mode: submission only covers %d IDs; full sample needs full run",
            len(preds),
        )
        sub = preds.rename(columns={"prediction": "Target"})
        sub["Target"] = sub["Target"].fillna(".").astype(str).replace("", ".")
        sub.to_csv(out, index=False)
        report = {
            "ok": False,
            "smoke": True,
            "n": len(sub),
            "note": "partial; re-run without --max-files for portal",
        }
    else:
        sub = build_submission(
            preds,
            sample_path=sample_path,
            out_path=out,
        )
        report = check_submission(out, sample_path)

    report_path = OUTPUT_DIR / "phase2_submission_check.json"
    # enrich report
    if isinstance(report, dict):
        report["chosen_lang_counts"] = (
            detail["chosen_lang"].value_counts().to_dict() if len(detail) else {}
        )
        report["source_counts"] = (
            detail["source"].value_counts().to_dict() if len(detail) else {}
        )
        report["out"] = str(out)
        report["n_detail"] = len(detail)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("submission check: %s", report)
    if not report.get("ok") and not args.max_files:
        raise SystemExit(f"Submission check failed: {report}")
    print(json.dumps(report, indent=2))
    print(f"UPLOAD: {out}")


if __name__ == "__main__":
    main()
