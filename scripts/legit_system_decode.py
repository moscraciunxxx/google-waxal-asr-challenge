#!/usr/bin/env python3
"""P1 multi-family decode skeleton: LID + WAXALNet MMS-300m + Whisper.

Whisper-first own-FT is the P2 path; Omnilingual is deferred (not loaded here).
Does not touch submission_phase2_FINAL.csv / floor KEEP.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoProcessor,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT, SEED, TARGET_SR
from src.legit_system_pack import (
    WAXAL300_LANGS,
    pack_hyp_row,
    resolve_decode_lang,
    rows_to_records,
    whisper_language_name,
)
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("legit_system_decode")

PHASE2_AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
LID126_CSV = OUTPUT_DIR / "phase2_lid126_full.csv"
DEFAULT_OUT = OUTPUT_DIR / "legit_system" / "decode_skeleton_hyps.csv"

WAXAL300 = {
    lang: f"waxal-benchmarking/mms-300m-waxal-{lang}" for lang in sorted(WAXAL300_LANGS)
}

WHISPER_WAXAL = {
    "lug": "waxal-benchmarking/whisper-small-waxal-lug",
    "lin": "waxal-benchmarking/whisper-small-waxal-lin",
    "sna": "waxal-benchmarking/whisper-small-waxal-sna",
}


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    if peak > 0:
        arr = arr / peak
    return arr, int(sr)


def load_lid_table(path: Path | None) -> dict[str, tuple[str, float]]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, tuple[str, float]] = {}
    for _, row in df.iterrows():
        sid = str(row["ID"])
        lang = str(row.get("lang1", row.get("lid_lang", "lug")))
        p1 = float(row.get("p1", row.get("lid_p1", 0.0)))
        out[sid] = (lang, p1)
    return out


class LiveLID:
    def __init__(self, device: torch.device, model_id: str = "facebook/mms-lid-126"):
        self.device = device
        logger.info("Loading live LID %s", model_id)
        try:
            self.fe = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=True)
            self.model = AutoModelForAudioClassification.from_pretrained(
                model_id, local_files_only=True
            )
        except Exception:
            self.fe = AutoFeatureExtractor.from_pretrained(model_id)
            self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.to(device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.inference_mode()
    def predict(self, array: np.ndarray, sr: int) -> tuple[str, float]:
        if sr != TARGET_SR:
            import librosa

            array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
        inputs = self.fe(array, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        idx = int(probs.argmax().item())
        return self.id2label[idx], float(probs[idx].item())


class MmsCache:
    def __init__(self, device: torch.device):
        self.device = device
        self.cache: dict[str, tuple[str, object, object]] = {}

    def get(self, lang: str):
        lang = resolve_decode_lang(lang)
        if lang in self.cache:
            return self.cache[lang]
        mid = WAXAL300.get(lang, WAXAL300["lug"])
        logger.info("Loading MMS specialist %s", mid)
        try:
            processor = AutoProcessor.from_pretrained(mid, local_files_only=True)
            model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(mid)
            model = Wav2Vec2ForCTC.from_pretrained(mid)
        model.to(self.device).eval()
        self.cache[lang] = (mid, model, processor)
        return self.cache[lang]


class WhisperFamily:
    """Whisper path: prefer WAXALNet specialist if cached, else openai/whisper-small."""

    def __init__(self, device: torch.device, model_id: str | None = None):
        self.device = device
        self.model_id = model_id or "openai/whisper-small"
        self._models: dict[str, tuple[str, object, object]] = {}

    def _load(self, mid: str):
        if mid in self._models:
            return self._models[mid]
        logger.info("Loading Whisper family %s", mid)
        try:
            processor = WhisperProcessor.from_pretrained(mid, local_files_only=True)
            model = WhisperForConditionalGeneration.from_pretrained(mid, local_files_only=True)
        except Exception:
            processor = WhisperProcessor.from_pretrained(mid)
            model = WhisperForConditionalGeneration.from_pretrained(mid)
        model.to(self.device).eval()
        self._models[mid] = (mid, model, processor)
        return self._models[mid]

    def resolve_id(self, decode_lang: str) -> str:
        if self.model_id != "openai/whisper-small":
            return self.model_id
        mid = WHISPER_WAXAL.get(decode_lang)
        if mid:
            try:
                WhisperProcessor.from_pretrained(mid, local_files_only=True)
                return mid
            except Exception:
                pass
        return "openai/whisper-small"

    @torch.inference_mode()
    def transcribe(
        self, array: np.ndarray, sr: int, decode_lang: str
    ) -> tuple[str, float | None, str]:
        if sr != TARGET_SR:
            import librosa

            array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
        mid = self.resolve_id(decode_lang)
        mid, model, processor = self._load(mid)
        inputs = processor(array, sampling_rate=TARGET_SR, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        gen_kwargs: dict = {"do_sample": False, "num_beams": 1}
        lang_name = whisper_language_name(decode_lang)
        if lang_name and mid.startswith("openai/"):
            try:
                forced = processor.get_decoder_prompt_ids(language=lang_name, task="transcribe")
                gen_kwargs["forced_decoder_ids"] = forced
            except Exception as e:
                logger.warning("forced_decoder_ids failed for %s: %s", lang_name, e)
        out_ids = model.generate(input_features, **gen_kwargs)
        text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        text = normalize_text(text) or "."
        # Sequence log-prob proxy not always available; leave score None for AR path
        return text, None, mid


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Legit multi-family decode skeleton (P1)")
    p.add_argument("--audio-dir", type=Path, default=PHASE2_AUDIO)
    p.add_argument("--ids", nargs="*", default=None, help="Sample IDs (without .wav)")
    p.add_argument("--max-files", type=int, default=2)
    p.add_argument("--lid-csv", type=Path, default=LID126_CSV)
    p.add_argument("--live-lid", action="store_true", help="Run mms-lid-126 live instead of CSV")
    p.add_argument("--whisper-model", default="openai/whisper-small")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    logger.info(
        "device=%s seed=%d omnilingual=deferred whisper-first-own-ft=P2",
        device,
        args.seed,
    )

    audio_dir = Path(args.audio_dir)
    if args.ids:
        paths = [audio_dir / f"{i}.wav" if not str(i).endswith(".wav") else audio_dir / i for i in args.ids]
    else:
        paths = sorted(audio_dir.glob("*.wav"))[: args.max_files]
    if not paths:
        raise SystemExit(f"No wavs under {audio_dir}")

    lid_table = {} if args.live_lid else load_lid_table(args.lid_csv)
    live_lid = LiveLID(device) if args.live_lid else None
    mms_cache = MmsCache(device)
    whisper = WhisperFamily(device, model_id=args.whisper_model)

    rows: list[dict] = []
    for path in paths:
        sid = path.stem
        if not path.exists():
            logger.warning("missing %s", path)
            continue
        arr, sr = load_wav(path)
        if live_lid is not None:
            lid_lang, lid_p1 = live_lid.predict(arr, sr)
        elif sid in lid_table:
            lid_lang, lid_p1 = lid_table[sid]
        else:
            lid_lang, lid_p1 = "lug", 0.0
            logger.warning("No LID for %s — default lug", sid)

        decode_lang = resolve_decode_lang(lid_lang)
        mms_id, mms_model, mms_proc = mms_cache.get(decode_lang)
        mms_hyp, mms_score = transcribe_waveform(
            mms_model, mms_proc, arr, sr, device=device, return_confidence=True
        )
        wh_hyp, wh_score, wh_id = whisper.transcribe(arr, sr, decode_lang)

        row = pack_hyp_row(
            sid,
            lid_lang=lid_lang,
            lid_p1=lid_p1,
            decode_lang=decode_lang,
            mms_hyp=mms_hyp,
            mms_score=mms_score,
            whisper_hyp=wh_hyp,
            whisper_score=wh_score,
            mms_model_id=mms_id,
            whisper_model_id=wh_id,
            seed=args.seed,
        )
        rows.append(row)
        logger.info(
            "%s lid=%s/%.3f decode=%s mms='%s' whisper='%s'",
            sid,
            lid_lang,
            lid_p1,
            decode_lang,
            (mms_hyp or "")[:60],
            (wh_hyp or "")[:60],
        )

    records = rows_to_records(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    meta = {
        "n": len(records),
        "out": str(out),
        "seed": args.seed,
        "device": str(device),
        "omnilingual": "deferred",
        "own_ft_priority": "whisper-first",
        "decode_spine": "waxalnet-mms-300m + lid",
        "ids": [r["ID"] for r in records],
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s (%d rows) meta=%s", out, len(records), meta_path)

    # Hard fail if either family empty
    for r in records:
        if not r["mms_hyp"] and not r["whisper_hyp"]:
            raise SystemExit(f"Both hyps empty for {r['ID']}")
        if not r["mms_hyp"]:
            raise SystemExit(f"mms_hyp empty for {r['ID']}")
        if not r["whisper_hyp"]:
            raise SystemExit(f"whisper_hyp empty for {r['ID']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
