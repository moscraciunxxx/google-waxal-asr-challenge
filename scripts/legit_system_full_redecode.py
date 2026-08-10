#!/usr/bin/env python3
"""Full Phase-2 multi-family redecode + fusion (all SampleSubmission IDs).

Families: WAXALNet MMS-300m specialist (via LID→decode_lang) + Whisper (own FT or base).
Fusion: src.legit_fusion.fuse_row — full-row multi-family selection, NOT residual conf surgery.

Does not overwrite submission_phase2_FINAL.csv / floor KEEP.
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
from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT, SEED, TARGET_SR
from src.legit_fusion import fuse_row, pack_fusion_submission_row
from src.legit_system_pack import (
    WAXAL300_LANGS,
    resolve_decode_lang,
    whisper_language_name,
)
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("full_redecode")

PHASE2_AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
PHASE2_SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
LID126_CSV = OUTPUT_DIR / "phase2_lid126_full.csv"
DEFAULT_OUT = OUTPUT_DIR / "legit_system" / "phase2_multifamily_fusion.csv"
DEFAULT_DETAIL = OUTPUT_DIR / "legit_system" / "phase2_multifamily_fusion_detail.csv"

WAXAL300 = {lang: f"waxal-benchmarking/mms-300m-waxal-{lang}" for lang in sorted(WAXAL300_LANGS)}
WHISPER_WAXAL = {
    "lug": "waxal-benchmarking/whisper-small-waxal-lug",
    "lin": "waxal-benchmarking/whisper-small-waxal-lin",
    "sna": "waxal-benchmarking/whisper-small-waxal-sna",
}


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    if peak > 0:
        arr = arr / peak
    return arr, int(sr)


def load_lid_table(path: Path) -> dict[str, tuple[str, float]]:
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        out[str(row["ID"])] = (str(row["lang1"]), float(row["p1"]))
    return out


class MmsCache:
    def __init__(self, device: torch.device):
        self.device = device
        self.cache: dict[str, tuple[str, object, object]] = {}

    def get(self, lang: str):
        lang = resolve_decode_lang(lang)
        if lang in self.cache:
            return self.cache[lang]
        mid = WAXAL300.get(lang, WAXAL300["lug"])
        logger.info("Load MMS %s", mid)
        try:
            proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
            model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
        except Exception:
            proc = AutoProcessor.from_pretrained(mid)
            model = Wav2Vec2ForCTC.from_pretrained(mid)
        model.to(self.device).eval()
        self.cache[lang] = (mid, model, proc)
        return self.cache[lang]


class WhisperFamily:
    def __init__(self, device: torch.device, model_id: str):
        self.device = device
        self.default_id = model_id
        self._models: dict[str, tuple[str, object, object]] = {}

    def _load(self, mid: str):
        if mid in self._models:
            return self._models[mid]
        logger.info("Load Whisper %s", mid)
        try:
            proc = WhisperProcessor.from_pretrained(mid, local_files_only=True)
            model = WhisperForConditionalGeneration.from_pretrained(mid, local_files_only=True)
        except Exception:
            proc = WhisperProcessor.from_pretrained(mid)
            model = WhisperForConditionalGeneration.from_pretrained(mid)
        model.to(self.device).eval()
        model.config.forced_decoder_ids = None
        if hasattr(model, "generation_config"):
            model.generation_config.forced_decoder_ids = None
        self._models[mid] = (mid, model, proc)
        return self._models[mid]

    def resolve_id(self, decode_lang: str) -> str:
        # Prefer own FT checkpoint if provided as default and exists
        if Path(self.default_id).exists() or self.default_id.startswith("checkpoints/"):
            return self.default_id
        mid = WHISPER_WAXAL.get(decode_lang)
        if mid:
            try:
                WhisperProcessor.from_pretrained(mid, local_files_only=True)
                return mid
            except Exception:
                pass
        return self.default_id if not self.default_id.startswith("checkpoints/") else "openai/whisper-small"

    @torch.inference_mode()
    def transcribe(self, array: np.ndarray, sr: int, decode_lang: str) -> tuple[str, str]:
        if sr != TARGET_SR:
            import librosa

            array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
        mid = self.resolve_id(decode_lang)
        mid, model, proc = self._load(mid)
        inputs = proc(array, sampling_rate=TARGET_SR, return_tensors="pt")
        feats = inputs.input_features.to(self.device)
        gen_kwargs: dict = {"do_sample": False, "num_beams": 1}
        lang_name = whisper_language_name(decode_lang)
        if lang_name and "openai" in mid:
            try:
                gen_kwargs["forced_decoder_ids"] = proc.get_decoder_prompt_ids(
                    language=lang_name, task="transcribe"
                )
            except Exception:
                pass
        ids = model.generate(feats, **gen_kwargs)
        text = normalize_text(proc.batch_decode(ids, skip_special_tokens=True)[0]) or "."
        return text, mid


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", type=Path, default=PHASE2_AUDIO)
    p.add_argument("--sample-csv", type=Path, default=PHASE2_SAMPLE)
    p.add_argument("--lid-csv", type=Path, default=LID126_CSV)
    p.add_argument("--whisper-model", default="checkpoints/whisper-waxal-legit-p2/best")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--detail-out", type=Path, default=DEFAULT_DETAIL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    logger.info("device=%s whisper=%s offset=%s limit=%s", device, args.whisper_model, args.offset, args.limit)

    sample = pd.read_csv(args.sample_csv)
    ids = sample["ID"].astype(str).tolist()
    if args.limit is not None:
        ids = ids[args.offset : args.offset + args.limit]
    else:
        ids = ids[args.offset :]

    lid_table = load_lid_table(args.lid_csv)
    mms_cache = MmsCache(device)
    whisper = WhisperFamily(device, args.whisper_model)

    done: set[str] = set()
    detail_rows: list[dict] = []
    if args.resume and args.detail_out.exists():
        prev = pd.read_csv(args.detail_out)
        detail_rows = prev.to_dict("records")
        done = set(str(r["ID"]) for r in detail_rows)
        logger.info("Resume: %d rows already done", len(done))

    t0 = time.time()
    for i, sid in enumerate(ids):
        if sid in done:
            continue
        path = args.audio_dir / f"{sid}.wav"
        if not path.exists():
            logger.warning("missing audio %s", path)
            continue
        arr, sr = load_wav(path)
        lid_lang, lid_p1 = lid_table.get(sid, ("lug", 0.0))
        decode_lang = resolve_decode_lang(lid_lang)
        mms_id, mms_model, mms_proc = mms_cache.get(decode_lang)
        mms_hyp, mms_score = transcribe_waveform(
            mms_model, mms_proc, arr, sr, device=device, return_confidence=True
        )
        wh_hyp, wh_id = whisper.transcribe(arr, sr, decode_lang)
        fus = fuse_row(
            mms_hyp,
            wh_hyp,
            mms_score=mms_score,
            decode_lang=decode_lang,
            lid_lang=lid_lang,
            lid_p1=lid_p1,
        )
        detail_rows.append(
            {
                "ID": sid,
                "lid_lang": lid_lang,
                "lid_p1": lid_p1,
                "decode_lang": decode_lang,
                "mms_model_id": mms_id,
                "mms_hyp": normalize_text(mms_hyp) or ".",
                "mms_score": mms_score,
                "whisper_model_id": wh_id,
                "whisper_hyp": normalize_text(wh_hyp) or ".",
                "fused_hyp": fus["fused_hyp"],
                "fusion_source": fus["fusion_source"],
                "fusion_reason": fus["fusion_reason"],
                "Target": fus["fused_hyp"],
            }
        )
        if (len(detail_rows) % 25 == 0) or (i + 1 == len(ids)):
            args.detail_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(detail_rows).to_csv(args.detail_out, index=False)
            logger.info(
                "progress %d/%d elapsed=%.1fs last=%s src=%s",
                len(detail_rows),
                len(ids) if not args.resume else 1500,
                time.time() - t0,
                sid,
                fus["fusion_source"],
            )

    # Full table against SampleSubmission order
    all_ids = sample["ID"].astype(str).tolist()
    by_id = {str(r["ID"]): r for r in detail_rows}
    # If partial run, merge with existing detail for full 1500 only when complete
    if args.detail_out.exists():
        by_id.update({str(r["ID"]): r for r in pd.read_csv(args.detail_out).to_dict("records")})

    missing = [i for i in all_ids if i not in by_id]
    sub_rows = []
    for sid in all_ids:
        if sid in by_id:
            r = by_id[sid]
            sub_rows.append(pack_fusion_submission_row(sid, r.get("fused_hyp") or r.get("Target") or "."))
        else:
            sub_rows.append(pack_fusion_submission_row(sid, "."))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows if detail_rows else list(by_id.values())).drop_duplicates("ID").to_csv(
        args.detail_out, index=False
    )
    pd.DataFrame(sub_rows)[["ID", "Target"]].to_csv(args.out, index=False)

    n_done = sum(1 for sid in all_ids if sid in by_id and str(by_id[sid].get("Target") or by_id[sid].get("fused_hyp") or "").strip() not in ("",))
    meta = {
        "n_sample": len(all_ids),
        "n_detail_unique": len(by_id),
        "n_with_hyp": n_done,
        "missing": len(missing),
        "out": str(args.out),
        "detail": str(args.detail_out),
        "whisper_model": args.whisper_model,
        "fusion": "legit_fusion.fuse_row multi-family full-row (not residual conf thr)",
        "omnilingual": "deferred",
        "floor_untouched": True,
        "complete_1500": len(missing) == 0 and n_done >= 1500,
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("meta %s", json.dumps(meta))
    if missing and args.limit is None and args.offset == 0:
        logger.warning("Incomplete: missing %d ids", len(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
