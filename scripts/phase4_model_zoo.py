#!/usr/bin/env python3
"""Phase-4 model zoo: open ASR systems on proxy_val_index (seed 42).

Systems (oracle true-lang where applicable; no cross-family conf mix scores):
  1. waxal-300m true-lang greedy
  2. mms-lug-ft-v2 on lug only
  3. facebook/mms-1b-all true-lang adapters (per-lang only)
  4. waxal-300m + KenLM beam (matched ARPA) with length-guard vs greedy

Outputs:
  outputs/phase4_model_zoo.json
  outputs/phase4_model_zoo.md
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Prefer local caches; allow hub for missing MMS adapters (nyn/xog).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Do NOT force HF offline — adapter.nyn / adapter.xog may need one-time download.

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINT_DIR, DATA_DIR, OUTPUT_DIR, TARGET_SR
from src.dataset import _decode_audio_item
from src.metrics import score_pairs
from src.mms_infer import load_mms, pick_device, set_lang, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase4_model_zoo")

SEED = 42
PROXY_CSV = DATA_DIR / "proxy_val_index.csv"
WAXAL300 = {
    lang: f"waxal-benchmarking/mms-300m-waxal-{lang}"
    for lang in ("ach", "nyn", "lug", "sog", "mas", "lin", "sna")
}
MMS_1B = "facebook/mms-1b-all"
# ISO remap for MMS-1b-all adapters (proxy "sog" == ISO xog; mas has no adapter)
MMS_ADAPTER_LANG = {
    "sog": "xog",  # Soga
    # "mas": None — no mms-1b adapter
}
FT_LUG = CHECKPOINT_DIR / "mms-lug-ft-v2"
LM_DIR = DATA_DIR / "lms"

# Length-guard: accept beam hyp only if word-count ratio vs greedy is in band
LEN_GUARD_LO = 0.5
LEN_GUARD_HI = 2.0


def free_mem(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    if device.type == "cuda":
        torch.cuda.empty_cache()


def free_model(model, device: torch.device) -> None:
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    free_mem(device)


def _resolve_val_parquets(lang: str) -> list[str]:
    hub_ds = "google--WaxalNLP"
    cache_root = Path(
        os.environ.get("HF_HUB_CACHE") or (Path.home() / ".cache/huggingface/hub")
    )
    snap_root = cache_root / f"datasets--{hub_ds}" / "snapshots"
    needle = f"{lang}-validation-"
    if not snap_root.is_dir():
        return []
    for snap in sorted(snap_root.iterdir(), reverse=True):
        asr_dir = snap / "data" / "ASR" / lang
        if not asr_dir.is_dir():
            continue
        found = sorted(
            p for p in asr_dir.glob("*.parquet") if needle in p.name and p.is_file()
        )
        if found:
            return [str(p.resolve()) for p in found]
    return []


def load_proxy_samples(
    index_csv: Path,
    max_per_lang: int | None = None,
    langs: list[str] | None = None,
    seed: int = SEED,
) -> list[dict]:
    """ID-matched load from HF validation parquet (soundfile path)."""
    from datasets import Audio, load_dataset

    idx = pd.read_csv(index_csv)
    if langs:
        idx = idx[idx["language"].isin(langs)]
    want_ids: dict[str, set[str]] = {}
    for lang, g in idx.groupby("language"):
        ids = g["id"].astype(str).tolist()
        if max_per_lang is not None and len(ids) > max_per_lang:
            rng = np.random.default_rng(seed)
            ids = list(rng.choice(ids, size=max_per_lang, replace=False))
        want_ids[str(lang)] = set(ids)

    samples: list[dict] = []
    for lang, id_set in sorted(want_ids.items()):
        logger.info("Loading %s validation for %d proxy ids", lang, len(id_set))
        files = _resolve_val_parquets(lang)
        if not files:
            from src.dataset import load_hf_asr_split

            logger.warning("%s: no cached parquet; load_hf_asr_split fallback", lang)
            ds = load_hf_asr_split(lang, "validation", max_samples=None)
        else:
            ds = load_dataset("parquet", data_files={"validation": files}, split="validation")
        all_ids = [str(x) for x in ds["id"]]
        positions = [i for i, uid in enumerate(all_ids) if uid in id_set]
        if not positions:
            logger.warning("%s: no matching proxy ids", lang)
            continue
        sub = ds.select(positions)
        try:
            sub = sub.cast_column("audio", Audio(decode=False))
        except Exception:
            pass
        for i in tqdm(range(len(sub)), desc=f"decode-audio-{lang}"):
            ex = sub[i]
            uid = str(ex["id"])
            audio = _decode_audio_item(ex["audio"], TARGET_SR)
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            ref = normalize_text(ex.get("transcription") or ex.get("text") or "")
            if not ref:
                # fall back to index transcription
                row = idx[idx["id"].astype(str) == uid]
                if len(row):
                    ref = normalize_text(str(row.iloc[0].get("transcription") or ""))
            if not ref:
                logger.warning("empty ref %s — skip", uid)
                continue
            samples.append(
                {"id": uid, "true_lang": lang, "ref": ref, "arr": arr, "sr": sr}
            )
        found = sum(1 for s in samples if s["true_lang"] == lang)
        logger.info("%s: loaded %d/%d", lang, found, len(id_set))
    return samples


def load_local_or_hub(model_id: str | Path, device: torch.device):
    mid = str(model_id)
    logger.info("Loading %s on %s", mid, device)
    try:
        processor = AutoProcessor.from_pretrained(mid, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
    except Exception as e:
        logger.warning("local_files_only failed for %s (%s); online retry", mid, e)
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        processor = AutoProcessor.from_pretrained(mid)
        model = Wav2Vec2ForCTC.from_pretrained(mid)
    model.to(device).eval()
    return model, processor


def metrics_block(refs: list[str], hyps: list[str]) -> dict:
    sc = score_pairs(refs, hyps)
    return {
        "n": int(sc["n"]),
        "wer": float(sc["wer"]),
        "cer": float(sc["cer"]),
        "error": float(sc["score"]),
        "zindi_est": float(1.0 - sc["score"]),
    }


def summarize_rows(rows: list[dict]) -> dict:
    overall = metrics_block([r["ref"] for r in rows], [r["hyp"] for r in rows])
    per: dict[str, dict] = {}
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["true_lang"]].append(r)
    for lang, group in sorted(by.items()):
        per[lang] = metrics_block([g["ref"] for g in group], [g["hyp"] for g in group])
    return {"overall": overall, "per_lang": per}


def word_count(text: str) -> int:
    t = (text or "").strip()
    if not t or t == ".":
        return 0
    return len(t.split())


def length_guard_ok(greedy: str, beam: str) -> bool:
    """Reject beam if space collapse / expansion vs greedy."""
    wg, wb = word_count(greedy), word_count(beam)
    if wg == 0 and wb == 0:
        return True
    if wg == 0:
        return wb <= 3  # avoid inventing long text from empty greedy
    ratio = wb / max(wg, 1)
    if ratio < LEN_GUARD_LO or ratio > LEN_GUARD_HI:
        return False
    # also reject total space collapse (single blob)
    if wb <= 1 and wg >= 4:
        return False
    return True


# ---------------------------------------------------------------------------
# System 1: waxal-300m true-lang greedy
# ---------------------------------------------------------------------------
@torch.inference_mode()
def eval_waxal_true_lang_greedy(samples: list[dict], device: torch.device) -> dict:
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_lang[s["true_lang"]].append(s)

    rows: list[dict] = []
    for lang in sorted(by_lang.keys()):
        if lang not in WAXAL300:
            logger.error("No waxal-300m for %s", lang)
            continue
        model, processor = load_local_or_hub(WAXAL300[lang], device)
        for s in tqdm(by_lang[lang], desc=f"waxal-greedy-{lang}"):
            hyp, conf = transcribe_waveform(
                model, processor, s["arr"], s["sr"], device=device, return_confidence=True
            )
            rows.append(
                {
                    "id": s["id"],
                    "true_lang": s["true_lang"],
                    "decode_lang": lang,
                    "hyp": hyp,
                    "ref": s["ref"],
                    "conf": float(conf),
                }
            )
        free_model(model, device)
        del processor
        free_mem(device)

    summary = summarize_rows(rows)
    return {
        "system": "waxal-300m-true-lang-greedy",
        "family": "waxal-300m",
        "description": "True-language waxal-300m greedy CTC (oracle lang)",
        "models": {lang: WAXAL300[lang] for lang in by_lang if lang in WAXAL300},
        "metrics": summary,
        "n_rows": len(rows),
        "detail": rows,
    }


# ---------------------------------------------------------------------------
# System 2: mms-lug-ft-v2 on lug only
# ---------------------------------------------------------------------------
@torch.inference_mode()
def eval_mms_lug_ft(samples: list[dict], device: torch.device) -> dict:
    lug = [s for s in samples if s["true_lang"] == "lug"]
    if not lug:
        return {
            "system": "mms-lug-ft-v2",
            "family": "mms-1b-ft",
            "error": "no lug samples",
            "metrics": {},
        }
    if not (FT_LUG / "model.safetensors").exists():
        raise FileNotFoundError(FT_LUG)

    model, processor = load_local_or_hub(FT_LUG, device)
    # FT full ckpt may still benefit from tokenizer lang set
    try:
        set_lang(model, processor, "lug")
    except Exception as e:
        logger.warning("set_lang lug on FT ckpt: %s", e)

    rows = []
    for s in tqdm(lug, desc="mms-lug-ft-v2"):
        hyp, conf = transcribe_waveform(
            model, processor, s["arr"], s["sr"], device=device, return_confidence=True
        )
        rows.append(
            {
                "id": s["id"],
                "true_lang": "lug",
                "decode_lang": "lug",
                "hyp": hyp,
                "ref": s["ref"],
                "conf": float(conf),
            }
        )
    free_model(model, device)
    del processor
    free_mem(device)

    summary = summarize_rows(rows)
    return {
        "system": "mms-lug-ft-v2",
        "family": "mms-1b-ft",
        "description": "Local fine-tuned MMS-1B on lug proxy only (oracle lang)",
        "checkpoint": str(FT_LUG),
        "metrics": summary,
        "n_rows": len(rows),
        "scope": "lug_only",
        "detail": rows,
    }


# ---------------------------------------------------------------------------
# System 3: mms-1b-all true-lang adapters (per-lang only; never conf-mix score)
# ---------------------------------------------------------------------------
def _try_load_mms_adapter(model, processor, lang: str) -> tuple[bool, str]:
    """Load true-lang adapter; return (ok, status). Do not decode on failure."""
    code = MMS_ADAPTER_LANG.get(lang, lang)
    if code is None:
        return False, "no_adapter_mapping"
    # Prefer local; fall back to hub for missing adapter files
    try:
        processor.tokenizer.set_target_lang(code)
    except Exception as e:
        return False, f"set_target_lang_fail:{e}"
    # word delimiter fix (same as set_lang)
    tok = processor.tokenizer
    try:
        pipe_id = None
        for i in range(len(tok)):
            if tok.convert_ids_to_tokens(i) == "|":
                pipe_id = i
                break
        if pipe_id is not None:
            tok.word_delimiter_token = "|"
            tok.word_delimiter_token_id = int(pipe_id)
            if hasattr(tok, "_word_delimiter_token"):
                tok._word_delimiter_token = "|"
    except Exception:
        pass
    try:
        model.load_adapter(code, local_files_only=True)
        return True, f"ok_local:{code}"
    except Exception:
        pass
    try:
        model.load_adapter(code)
        return True, f"ok_hub:{code}"
    except Exception as e:
        return False, f"load_adapter_fail:{code}:{e}"


@torch.inference_mode()
def eval_mms_1b_true_lang(samples: list[dict], device: torch.device) -> dict:
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_lang[s["true_lang"]].append(s)

    # Allow hub for adapters
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    model, processor, dev = load_mms(MMS_1B, device=device)
    device = dev

    rows: list[dict] = []
    adapter_status: dict[str, str] = {}
    for lang in sorted(by_lang.keys()):
        ok, status = _try_load_mms_adapter(model, processor, lang)
        adapter_status[lang] = status
        if not ok:
            logger.warning("SKIP mms-1b lang=%s status=%s", lang, status)
            continue
        for s in tqdm(by_lang[lang], desc=f"mms-1b-{lang}"):
            hyp, conf = transcribe_waveform(
                model, processor, s["arr"], s["sr"], device=device, return_confidence=True
            )
            rows.append(
                {
                    "id": s["id"],
                    "true_lang": lang,
                    "decode_lang": lang,
                    "adapter_code": MMS_ADAPTER_LANG.get(lang, lang),
                    "hyp": hyp,
                    "ref": s["ref"],
                    "conf": float(conf),
                }
            )

    free_model(model, device)
    del processor
    free_mem(device)

    summary = summarize_rows(rows) if rows else {"overall": {}, "per_lang": {}}
    return {
        "system": "mms-1b-all-true-lang-adapter",
        "family": "mms-1b-all",
        "description": (
            "facebook/mms-1b-all with true-lang adapter via set_lang+load_adapter; "
            "per-lang rows only for selection — do not conf-mix with waxal for upload. "
            "sog→xog remap; mas skipped (no adapter)."
        ),
        "model_id": MMS_1B,
        "adapter_lang_map": MMS_ADAPTER_LANG,
        "adapter_status": adapter_status,
        "metrics": summary,
        "n_rows": len(rows),
        "scope": "per_lang_oracle",
        "detail": rows,
    }


# ---------------------------------------------------------------------------
# System 4: waxal + KenLM beam with length-guard
# ---------------------------------------------------------------------------
def build_ctc_decoder(processor, lang: str, alpha: float, beta: float):
    from pyctcdecode import build_ctcdecoder

    vocab = processor.tokenizer.get_vocab()
    id2 = {i: t for t, i in vocab.items()}
    labels = [id2[i] for i in range(len(id2))]
    lm = LM_DIR / f"{lang}_2gram.arpa"
    uni_path = LM_DIR / f"{lang}_unigrams.txt"
    if not lm.is_file():
        raise FileNotFoundError(lm)
    unigrams = None
    if uni_path.is_file():
        unigrams = [u for u in uni_path.read_text().splitlines() if u.strip()]
    return build_ctcdecoder(
        labels,
        kenlm_model_path=str(lm),
        unigrams=unigrams,
        alpha=alpha,
        beta=beta,
    )


@torch.inference_mode()
def decode_beam(model, processor, decoder, array, sr, device, beam_width: int) -> str:
    array = np.asarray(array, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.max(np.abs(array)) + 1e-9)
    if peak > 0:
        array = array / peak
    inputs = processor(array, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()
    text = decoder.decode(logits, beam_width=beam_width)
    text = text.replace("|", " ")
    return normalize_text(text) or "."


@torch.inference_mode()
def eval_waxal_beam_lm(
    samples: list[dict],
    device: torch.device,
    *,
    alpha: float = 0.3,
    beta: float = 0.5,
    beam: int = 50,
    greedy_detail: list[dict] | None = None,
) -> dict:
    """Waxal true-lang beam + matched LM; length-guard vs greedy."""
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_lang[s["true_lang"]].append(s)

    greedy_map = {}
    if greedy_detail:
        greedy_map = {r["id"]: r["hyp"] for r in greedy_detail}

    rows: list[dict] = []
    guard_stats = {"accepted": 0, "rejected": 0, "no_lm": 0}
    for lang in sorted(by_lang.keys()):
        if lang not in WAXAL300:
            continue
        lm_path = LM_DIR / f"{lang}_2gram.arpa"
        if not lm_path.is_file():
            logger.warning("No LM for %s — skip beam, keep greedy", lang)
            for s in by_lang[lang]:
                g = greedy_map.get(s["id"])
                if g is None:
                    continue
                rows.append(
                    {
                        "id": s["id"],
                        "true_lang": lang,
                        "decode_lang": lang,
                        "hyp": g,
                        "ref": s["ref"],
                        "beam_hyp": g,
                        "greedy_hyp": g,
                        "used_beam": 0,
                        "guard_reject": 0,
                    }
                )
                guard_stats["no_lm"] += 1
            continue

        model, processor = load_local_or_hub(WAXAL300[lang], device)
        try:
            decoder = build_ctc_decoder(processor, lang, alpha, beta)
        except Exception as e:
            logger.error("decoder build failed for %s: %s", lang, e)
            free_model(model, device)
            continue

        for s in tqdm(by_lang[lang], desc=f"waxal-beam-{lang}"):
            greedy = greedy_map.get(s["id"])
            if greedy is None:
                greedy, _ = transcribe_waveform(
                    model, processor, s["arr"], s["sr"], device=device, return_confidence=True
                )
            try:
                beam_hyp = decode_beam(
                    model, processor, decoder, s["arr"], s["sr"], device, beam
                )
            except Exception as e:
                logger.warning("beam fail %s: %s", s["id"], e)
                beam_hyp = greedy

            accept = length_guard_ok(greedy, beam_hyp)
            if accept and beam_hyp != greedy:
                final = beam_hyp
                used = 1
                guard_stats["accepted"] += 1
            else:
                final = greedy
                used = 0
                if not accept:
                    guard_stats["rejected"] += 1
            rows.append(
                {
                    "id": s["id"],
                    "true_lang": lang,
                    "decode_lang": lang,
                    "hyp": final,
                    "ref": s["ref"],
                    "beam_hyp": beam_hyp,
                    "greedy_hyp": greedy,
                    "used_beam": used,
                    "guard_reject": int(not accept),
                }
            )

        free_model(model, device)
        del processor, decoder
        free_mem(device)

    # Also score raw beam without guard for diagnostics
    raw_rows = [
        {"true_lang": r["true_lang"], "ref": r["ref"], "hyp": r["beam_hyp"]} for r in rows
    ]
    guarded = summarize_rows(rows)
    raw = summarize_rows(raw_rows) if raw_rows else {}

    return {
        "system": "waxal-300m-true-lang-beam-lm",
        "family": "waxal-300m",
        "description": (
            f"True-lang waxal-300m + KenLM 2-gram (alpha={alpha}, beta={beta}, "
            f"beam={beam}) with length-guard vs greedy "
            f"(word ratio [{LEN_GUARD_LO},{LEN_GUARD_HI}])"
        ),
        "alpha": alpha,
        "beta": beta,
        "beam": beam,
        "length_guard": {"lo": LEN_GUARD_LO, "hi": LEN_GUARD_HI},
        "guard_stats": guard_stats,
        "metrics": guarded,
        "metrics_raw_beam_no_guard": raw,
        "n_rows": len(rows),
        "detail": rows,
    }



def save_checkpoint(systems: dict, path: Path, meta: dict) -> None:
    systems_out = {k: strip_detail(v) for k, v in systems.items()}
    payload = {**meta, "systems": systems_out}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("checkpoint %s systems=%s", path, list(systems.keys()))


def strip_detail(result: dict) -> dict:
    """JSON-safe result without full hyp detail (detail written separately if needed)."""
    out = {k: v for k, v in result.items() if k != "detail"}
    return out


def write_md(payload: dict, path: Path) -> None:
    systems = payload["systems"]
    lines = [
        "# Phase-4 Model Zoo (proxy)",
        "",
        f"Proxy: `{payload['proxy_csv']}` (seed={payload['seed']}, n={payload['n_samples']}).",
        f"Device: `{payload['device']}` · wall_s={payload['wall_s']:.1f}",
        "",
        "Score: `zindi_est = 1 - 0.5*WER - 0.5*CER` (higher better).",
        "Protocol: oracle true-lang where applicable; ID-matched HF validation audio.",
        "",
        "## Hard rules applied",
        "",
        "- No cross-family CTC conf pick recommended for upload.",
        "- KenLM/beam only if proxy shows clear gain without space collapse.",
        "- MMS-1B adapter scores reported **per-lang**; overall is diagnostic only.",
        "",
        "## Overall (where fair)",
        "",
        "| system | family | n | zindi_est | WER | CER | notes |",
        "|--------|--------|--:|----------:|----:|----:|-------|",
    ]

    note_map = {
        "waxal-300m-true-lang-greedy": "oracle-lang baseline",
        "mms-lug-ft-v2": "lug only",
        "mms-1b-all-true-lang-adapter": "oracle-lang multi-lang avg (diagnostic)",
        "waxal-300m-true-lang-beam-lm": "beam+LM+length-guard",
    }
    for key in (
        "waxal-300m-true-lang-greedy",
        "mms-1b-all-true-lang-adapter",
        "waxal-300m-true-lang-beam-lm",
        "mms-lug-ft-v2",
    ):
        s = systems.get(key)
        if not s or not s.get("metrics"):
            continue
        o = s["metrics"]["overall"]
        lines.append(
            f"| {s['system']} | {s.get('family','')} | {o['n']} | "
            f"{o['zindi_est']:.4f} | {o['wer']:.4f} | {o['cer']:.4f} | "
            f"{note_map.get(key, '')} |"
        )

    # Raw beam diagnostic
    beam = systems.get("waxal-300m-true-lang-beam-lm")
    if beam and beam.get("metrics_raw_beam_no_guard"):
        o = beam["metrics_raw_beam_no_guard"]["overall"]
        lines.append(
            f"| waxal-beam-raw (no guard) | waxal-300m | {o['n']} | "
            f"{o['zindi_est']:.4f} | {o['wer']:.4f} | {o['cer']:.4f} | diagnostic |"
        )

    # Per-lang table
    langs = payload.get("langs", [])
    lines += [
        "",
        "## Per-language zindi_est",
        "",
    ]
    # Build column list of systems that have per_lang
    sys_keys = []
    for key in (
        "waxal-300m-true-lang-greedy",
        "mms-1b-all-true-lang-adapter",
        "waxal-300m-true-lang-beam-lm",
        "mms-lug-ft-v2",
    ):
        if systems.get(key) and systems[key].get("metrics"):
            sys_keys.append(key)

    short = {
        "waxal-300m-true-lang-greedy": "waxal_greedy",
        "mms-1b-all-true-lang-adapter": "mms1b_adapter",
        "waxal-300m-true-lang-beam-lm": "waxal_beam_guard",
        "mms-lug-ft-v2": "mms_lug_ft",
    }
    header = "| lang | n |" + "".join(f" {short[k]} |" for k in sys_keys)
    sep = "|------|--:|" + "".join("----------:|" for _ in sys_keys)
    lines.append(header)
    lines.append(sep)

    # n from waxal greedy if present
    base = systems.get("waxal-300m-true-lang-greedy", {}).get("metrics", {}).get("per_lang", {})
    all_langs = sorted(
        set(langs)
        | set(base.keys())
        | set(
            systems.get("mms-1b-all-true-lang-adapter", {})
            .get("metrics", {})
            .get("per_lang", {})
            .keys()
        )
    )
    for lang in all_langs:
        n = base.get(lang, {}).get("n", "?")
        cells = []
        for k in sys_keys:
            pl = systems[k]["metrics"].get("per_lang", {})
            if lang in pl:
                cells.append(f"{pl[lang]['zindi_est']:.4f}")
            else:
                cells.append("—")
        lines.append(f"| {lang} | {n} |" + "".join(f" {c} |" for c in cells))

    # Detailed per-system sections
    lines += ["", "## System details", ""]
    for key in sys_keys:
        s = systems[key]
        lines.append(f"### `{s['system']}`")
        lines.append("")
        lines.append(s.get("description", ""))
        lines.append("")
        if "adapter_status" in s:
            lines.append(f"Adapter status: `{s['adapter_status']}`")
            lines.append("")
        if "guard_stats" in s:
            lines.append(f"Guard stats: `{s['guard_stats']}`")
            lines.append("")
            if s.get("metrics_raw_beam_no_guard"):
                ro = s["metrics_raw_beam_no_guard"]["overall"]
                lines.append(
                    f"Raw beam (no guard) overall zindi_est={ro['zindi_est']:.4f} "
                    f"WER={ro['wer']:.4f} CER={ro['cer']:.4f}"
                )
                lines.append("")
        o = s["metrics"]["overall"]
        lines.append(
            f"Overall: zindi_est={o['zindi_est']:.4f} WER={o['wer']:.4f} "
            f"CER={o['cer']:.4f} n={o['n']}"
        )
        lines.append("")
        lines.append("| lang | n | zindi_est | WER | CER |")
        lines.append("|------|--:|----------:|----:|----:|")
        for lang, p in sorted(s["metrics"]["per_lang"].items()):
            lines.append(
                f"| {lang} | {p['n']} | {p['zindi_est']:.4f} | {p['wer']:.4f} | {p['cer']:.4f} |"
            )
        lines.append("")

    # Deltas vs waxal greedy
    wax = systems.get("waxal-300m-true-lang-greedy", {}).get("metrics", {})
    if wax:
        lines += ["## Deltas vs waxal-300m true-lang greedy", ""]
        lines.append("| system | scope | Δzindi_est |")
        lines.append("|--------|-------|-----------:|")
        w_all = wax["overall"]["zindi_est"]
        for key in sys_keys:
            if key == "waxal-300m-true-lang-greedy":
                continue
            s = systems[key]
            if s.get("scope") == "lug_only":
                w_lug = wax.get("per_lang", {}).get("lug", {}).get("zindi_est")
                s_lug = s["metrics"]["per_lang"].get("lug", {}).get("zindi_est")
                if w_lug is not None and s_lug is not None:
                    lines.append(
                        f"| {s['system']} | lug | {s_lug - w_lug:+.4f} |"
                    )
            else:
                so = s["metrics"]["overall"]["zindi_est"]
                lines.append(f"| {s['system']} | overall | {so - w_all:+.4f} |")
                # per-lang for mms-1b
                if key == "mms-1b-all-true-lang-adapter":
                    for lang, p in sorted(s["metrics"]["per_lang"].items()):
                        w = wax.get("per_lang", {}).get(lang, {}).get("zindi_est")
                        if w is not None:
                            lines.append(
                                f"| {s['system']} | {lang} | {p['zindi_est'] - w:+.4f} |"
                            )
        lines.append("")

    rec = payload.get("recommendation", {})
    lines += [
        "## Phase-5 ensemble recommendation",
        "",
        rec.get("summary", ""),
        "",
    ]
    for item in rec.get("include", []):
        lines.append(f"- **INCLUDE**: {item}")
    for item in rec.get("exclude", []):
        lines.append(f"- **EXCLUDE**: {item}")
    for item in rec.get("notes", []):
        lines.append(f"- {item}")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload.get('out_json', 'outputs/phase4_model_zoo.json')}`",
        f"- MD: `{path}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_recommendation(systems: dict) -> dict:
    """Decide Phase-5 ensemble members from proxy zoo results."""
    include: list[str] = []
    exclude: list[str] = []
    notes: list[str] = []

    wax = systems.get("waxal-300m-true-lang-greedy")
    ft = systems.get("mms-lug-ft-v2")
    mms1b = systems.get("mms-1b-all-true-lang-adapter")
    beam = systems.get("waxal-300m-true-lang-beam-lm")

    if wax and wax.get("metrics"):
        include.append(
            f"waxal-300m true-lang (or openset multi-hyp same-family) as spine "
            f"[proxy overall zindi_est={wax['metrics']['overall']['zindi_est']:.4f}]"
        )

    if ft and ft.get("metrics"):
        ft_z = ft["metrics"]["per_lang"].get("lug", {}).get("zindi_est")
        wax_lug = (
            wax["metrics"]["per_lang"].get("lug", {}).get("zindi_est")
            if wax and wax.get("metrics")
            else None
        )
        if ft_z is not None and wax_lug is not None:
            if ft_z > wax_lug + 0.001:
                include.append(
                    f"mms-lug-ft-v2 for lug-routed utterances only "
                    f"(zindi_est={ft_z:.4f} vs waxal-lug {wax_lug:.4f}); "
                    f"route by LID/true-lang, never by cross-family CTC conf"
                )
            else:
                exclude.append(
                    f"mms-lug-ft-v2 (no clear proxy gain vs waxal-lug: "
                    f"{ft_z:.4f} vs {wax_lug:.4f})"
                )
                notes.append(
                    "If FT ≈ waxal on lug, keep waxal-only for simplicity unless public has lug mass."
                )

    if mms1b and mms1b.get("metrics"):
        beats = []
        loses = []
        for lang, p in mms1b["metrics"]["per_lang"].items():
            w = (
                wax["metrics"]["per_lang"].get(lang, {}).get("zindi_est")
                if wax and wax.get("metrics")
                else None
            )
            if w is None:
                continue
            if p["zindi_est"] > w + 0.01:
                beats.append(f"{lang}:{p['zindi_est']:.4f}>{w:.4f}")
            else:
                loses.append(f"{lang}:{p['zindi_est']:.4f}<={w:.4f}")
        if beats:
            include.append(
                "mms-1b-all adapters **only** for langs where proxy beats waxal "
                f"({', '.join(beats)}); language-routed, never conf-mix with waxal-300m"
            )
        if loses:
            exclude.append(
                "mms-1b-all for langs it loses on proxy "
                f"({', '.join(loses)}); keep waxal-300m"
            )
        notes.append(
            "HARD: do not recommend cross-family CTC confidence pick for upload "
            "(Phase-1 forensics: 1B conf-mix hurt public)."
        )

    if beam and beam.get("metrics"):
        bz = beam["metrics"]["overall"]["zindi_est"]
        wz = wax["metrics"]["overall"]["zindi_est"] if wax and wax.get("metrics") else None
        raw = beam.get("metrics_raw_beam_no_guard", {}).get("overall", {})
        raw_z = raw.get("zindi_est")
        gs = beam.get("guard_stats", {})
        if wz is not None:
            if bz > wz + 0.005:
                include.append(
                    f"waxal+KenLM beam with length-guard "
                    f"(zindi_est={bz:.4f} vs greedy {wz:.4f}, Δ={bz-wz:+.4f})"
                )
            else:
                exclude.append(
                    f"waxal+KenLM beam (proxy zindi_est={bz:.4f} vs greedy {wz:.4f}, "
                    f"Δ={bz-wz:+.4f}; Phase-1 KenLM was toxic on Phase-2 public)"
                )
                if raw_z is not None and raw_z < wz:
                    notes.append(
                        f"Raw beam without guard is worse ({raw_z:.4f}); space-collapse risk confirmed."
                    )
                notes.append(f"Guard stats: {gs}")

    notes.append(
        "Phase-5 ensemble should prefer same-family waxal multi-hyp + optional "
        "language-routed specialists (FT lug / strong 1B langs), not conf argmax across families."
    )

    summary = (
        "Use waxal-300m same-family multi-hyp as the Phase-5 spine; "
        "add language-routed specialists only where proxy shows clear gain; "
        "never cross-family conf-mix; beam/LM only if proxy-positive with length-guard."
    )
    return {
        "summary": summary,
        "include": include,
        "exclude": exclude,
        "notes": notes,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=PROXY_CSV)
    p.add_argument("--max-per-lang", type=int, default=None)
    p.add_argument("--langs", nargs="+", default=None, help="Subset e.g. lug ach")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-beam", action="store_true")
    p.add_argument("--skip-mms1b", action="store_true")
    p.add_argument("--skip-ft", action="store_true")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--beam", type=int, default=50)
    p.add_argument("--out-json", type=Path, default=OUTPUT_DIR / "phase4_model_zoo.json")
    p.add_argument("--out-md", type=Path, default=OUTPUT_DIR / "phase4_model_zoo.md")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    logger.info("device=%s seed=%s", device, args.seed)
    t0 = time.time()

    samples = load_proxy_samples(
        args.index, max_per_lang=args.max_per_lang, langs=args.langs, seed=args.seed
    )
    if not samples:
        raise SystemExit("No proxy samples loaded")
    lang_counts = dict(Counter(s["true_lang"] for s in samples))
    logger.info("samples=%d langs=%s", len(samples), lang_counts)

    systems: dict[str, dict] = {}

    # 1) waxal greedy oracle
    logger.info("=== System 1: waxal-300m true-lang greedy ===")
    r1 = eval_waxal_true_lang_greedy(samples, device)
    systems[r1["system"]] = r1
    logger.info(
        "waxal greedy overall zindi_est=%.4f", r1["metrics"]["overall"]["zindi_est"]
    )
    save_checkpoint(systems, args.out_json.with_suffix(".partial.json"), {
        "task": "phase4_model_zoo_partial", "device": str(device), "n_samples": len(samples),
    })

    # 2) mms-lug-ft-v2
    if not args.skip_ft:
        logger.info("=== System 2: mms-lug-ft-v2 (lug only) ===")
        r2 = eval_mms_lug_ft(samples, device)
        systems[r2["system"]] = r2
        if r2.get("metrics"):
            logger.info(
                "ft-lug zindi_est=%.4f",
                r2["metrics"]["overall"]["zindi_est"],
            )

    # 3) mms-1b-all true-lang adapters
    if not args.skip_mms1b:
        logger.info("=== System 3: mms-1b-all true-lang adapters ===")
        r3 = eval_mms_1b_true_lang(samples, device)
        systems[r3["system"]] = r3
        if r3.get("metrics"):
            logger.info(
                "mms-1b overall zindi_est=%.4f (diagnostic)",
                r3["metrics"]["overall"]["zindi_est"],
            )

    # 4) waxal beam + LM + length guard
    if not args.skip_beam:
        logger.info("=== System 4: waxal beam+LM+length-guard ===")
        greedy_detail = systems.get("waxal-300m-true-lang-greedy", {}).get("detail")
        r4 = eval_waxal_beam_lm(
            samples,
            device,
            alpha=args.alpha,
            beta=args.beta,
            beam=args.beam,
            greedy_detail=greedy_detail,
        )
        systems[r4["system"]] = r4
        if r4.get("metrics"):
            logger.info(
                "waxal beam+guard overall zindi_est=%.4f",
                r4["metrics"]["overall"]["zindi_est"],
            )

    recommendation = build_recommendation(systems)
    wall = time.time() - t0

    # Persist (strip large detail arrays from main JSON; keep compact metrics)
    systems_out = {k: strip_detail(v) for k, v in systems.items()}
    payload = {
        "task": "phase4_model_zoo",
        "seed": args.seed,
        "proxy_csv": str(args.index.resolve()),
        "n_samples": len(samples),
        "langs": sorted(lang_counts.keys()),
        "lang_counts": lang_counts,
        "device": str(device),
        "wall_s": wall,
        "beam_params": {
            "alpha": args.alpha,
            "beta": args.beta,
            "beam": args.beam,
            "length_guard_lo": LEN_GUARD_LO,
            "length_guard_hi": LEN_GUARD_HI,
        },
        "systems": systems_out,
        "recommendation": recommendation,
        "out_json": str(args.out_json),
        "hard_rules": [
            "Do not recommend cross-family CTC conf pick for upload",
            "Phase-1 KenLM was toxic on Phase-2 public; only recommend beam if proxy shows clear gain without collapse",
            "Report MMS-1B per-lang; never conf-mix with waxal for a single score without separate rows",
        ],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload, args.out_md)

    # Compact console summary
    print("\n=== PHASE-4 MODEL ZOO ===")
    for name, s in systems_out.items():
        if not s.get("metrics"):
            continue
        o = s["metrics"]["overall"]
        print(
            f"{name}: n={o['n']} zindi_est={o['zindi_est']:.4f} "
            f"wer={o['wer']:.4f} cer={o['cer']:.4f}"
        )
        for lang, p in sorted(s["metrics"]["per_lang"].items()):
            print(f"  {lang}: zindi_est={p['zindi_est']:.4f}")
    print("\nRECOMMENDATION:")
    print(recommendation["summary"])
    for x in recommendation["include"]:
        print("  INCLUDE:", x)
    for x in recommendation["exclude"]:
        print("  EXCLUDE:", x)
    print("WROTE", args.out_json)
    print("WROTE", args.out_md)
    print(f"wall_s={wall:.1f}")


if __name__ == "__main__":
    main()
