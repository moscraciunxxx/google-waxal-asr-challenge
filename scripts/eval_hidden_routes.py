#!/usr/bin/env python3
"""Audit and integrate conservative Phase-2 hidden-language routes.

This helper is deliberately self-contained.  It reads the existing validation
caches/checkpoints but writes only below outputs/goal_2026_08_08/hidden_routes.

The deployment question is stricter than "which model is best on its own
language?": old Phase-2 MMS-LID labels 785 clips as Luo even though the current
open-set decoder assigns most of them to Acholi/Luganda/Runyankole.  A Dholuo
replacement is therefore allowed only behind a cross-family agreement gate
whose false-fire rate is measured on exact WAXAL validation IDs from all nearby
non-Luo routes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2BertForCTC,
    Wav2Vec2ForCTC,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TARGET_SR  # noqa: E402
from src.metrics import score_pairs  # noqa: E402
from src.text_norm import normalize_text  # noqa: E402
from scripts.mms_adapter_ft import fix_mms_tokenizer  # noqa: E402

OUT = ROOT / "outputs" / "goal_2026_08_08" / "hidden_routes"
SPINE = (
    ROOT
    / "outputs"
    / "goal_2026_08_07"
    / "badrex_tiers"
    / "submission_phase2_badrex_sna_sim99_lug_splitjoin.csv"
)
ROUTES = ROOT / "outputs" / "phase2_selective_v3_detail.csv"
OPENSET = ROOT / "outputs" / "phase2_openset_detail.csv"
PUBLIC = ROOT / "outputs" / "beat075" / "public_visible_index.csv"
ORTHO = ROOT / "outputs" / "beat_k63" / "ortho_lid_luo_scores.csv"
PROXY = ROOT / "data" / "proxy_val_index.csv"

MMS_ID = "facebook/mms-1b-all"
CLEAR_ID = "CLEAR-Global/w2v-bert-2.0-luo_19_77h"
SUNBIRD_ID = "Sunbird/asr-whisper-51-african-languages"
WHISPER_SOG_ID = "waxal-benchmarking/whisper-small-waxal-sog"
SUNBIRD_SOG_TOKEN = 50310  # model-card token for xog/Lusoga

NEGATIVE_LANGS = ("ach", "lug", "nyn", "sog", "mas")
THRESHOLDS = (0.08, 0.10, 0.12, 0.15)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(refs: list[str], hyps: list[str]) -> dict[str, float]:
    score = score_pairs(refs, hyps)
    return {
        "n": len(refs),
        "wer": float(score["wer"]),
        "cer": float(score["cer"]),
        "zindi": float(1.0 - score["score"]),
    }


def pick_device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clean_audio(example: dict) -> np.ndarray:
    audio = example["audio"]
    if isinstance(audio, dict) and audio.get("array") is not None:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate") or TARGET_SR)
    elif isinstance(audio, dict) and audio.get("bytes") is not None:
        arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    elif isinstance(audio, dict) and audio.get("path"):
        arr, sr = sf.read(str(audio["path"]), dtype="float32", always_2d=False)
    else:
        raise ValueError(f"unsupported audio object for {example.get('ID') or example.get('id')}")
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    if arr.ndim > 1:
        arr = arr.mean(-1)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak


def proxy_examples(lang: str) -> list[dict]:
    index = pd.read_csv(PROXY)
    wanted = index.loc[index.language.eq(lang), "id"].astype(str).tolist()
    refs = dict(
        zip(
            index.loc[index.language.eq(lang), "id"].astype(str),
            index.loc[index.language.eq(lang), "transcription"].map(normalize_text),
        )
    )
    summary = json.loads((ROOT / "data" / "proxy_val_index.summary.json").read_text())
    paths = [Path(path) for path in summary["per_lang"][lang]["parquet_files"]]
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise RuntimeError(f"{lang}: cached validation parquet is unavailable")
    dataset = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    dataset["id"] = dataset.id.astype(str)
    by_id = {uid: i for i, uid in enumerate(dataset.id)}
    missing = [uid for uid in wanted if uid not in by_id]
    if missing:
        raise RuntimeError(f"{lang}: proxy IDs absent from validation: {missing[:5]}")
    return [
        {
            "ID": uid,
            "language": lang,
            "reference": refs[uid],
            "audio": clean_audio(dataset.iloc[by_id[uid]].to_dict()),
        }
        for uid in wanted
    ]


def release(model) -> None:
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


@torch.inference_mode()
def decode_ctc(model, processor, audio: np.ndarray, device: torch.device, bert: bool = False) -> str:
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    if bert:
        kwargs = {
            key: value.to(device)
            for key, value in inputs.items()
            if key in ("input_features", "attention_mask")
        }
        logits = model(**kwargs).logits
        ids = logits.argmax(-1)
        text = processor.batch_decode(ids)[0]
    else:
        logits = model(inputs.input_values.to(device)).logits
        ids = logits.argmax(-1)[0]
        text = processor.decode(ids)
    return normalize_text(text.replace("|", " ")) or "."


def resume_column(path: Path, column: str) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "ID" not in frame or column not in frame:
        return {}
    return dict(zip(frame.ID.astype(str), frame[column].fillna("").astype(str)))


def save_rows(rows: dict[str, dict], path: Path) -> None:
    pd.DataFrame(rows.values()).sort_values(["language", "ID"]).to_csv(path, index=False)


def evaluate_luo_detector_negatives(device: torch.device, limit: int | None = None) -> dict:
    """Measure MMS-Luo/CLEAR-Luo agreement false fires on exact proxy IDs."""
    path = OUT / "luo_detector_negative_controls.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows: dict[str, dict] = {}
    if not old.empty:
        rows = {str(row["ID"]): row.to_dict() for _, row in old.iterrows()}

    examples: list[dict] = []
    for lang in NEGATIVE_LANGS:
        current = proxy_examples(lang)
        if limit is not None:
            current = current[:limit]
        examples.extend(current)
        for ex in current:
            rows.setdefault(
                ex["ID"],
                {
                    "ID": ex["ID"],
                    "language": lang,
                    "reference": ex["reference"],
                },
            )

    # MMS-1B Luo is the proposed replacement text.
    missing = [ex for ex in examples if not str(rows[ex["ID"]].get("mms_luo", "")).strip()]
    if missing:
        proc = AutoProcessor.from_pretrained(MMS_ID, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(MMS_ID, local_files_only=True)
        fix_mms_tokenizer(proc, "luo")
        try:
            model.load_adapter("luo", local_files_only=True)
        except TypeError:
            model.load_adapter("luo")
        model.to(device).eval()
        started = time.time()
        for position, ex in enumerate(missing, 1):
            rows[ex["ID"]]["mms_luo"] = decode_ctc(model, proc, ex["audio"], device)
            if position % 10 == 0 or position == len(missing):
                save_rows(rows, path)
                print(f"negative controls MMS-Luo {position}/{len(missing)} {time.time()-started:.1f}s", flush=True)
        release(model)

    # CLEAR is architecturally independent, making agreement a useful detector.
    missing = [ex for ex in examples if not str(rows[ex["ID"]].get("clear_luo", "")).strip()]
    if missing:
        proc = AutoProcessor.from_pretrained(CLEAR_ID, local_files_only=True)
        model = Wav2Vec2BertForCTC.from_pretrained(CLEAR_ID, local_files_only=True).to(device).eval()
        started = time.time()
        for position, ex in enumerate(missing, 1):
            rows[ex["ID"]]["clear_luo"] = decode_ctc(
                model, proc, ex["audio"], device, bert=True
            )
            if position % 10 == 0 or position == len(missing):
                save_rows(rows, path)
                print(f"negative controls CLEAR-Luo {position}/{len(missing)} {time.time()-started:.1f}s", flush=True)
        release(model)

    frame = pd.DataFrame(rows.values()).sort_values(["language", "ID"])
    frame["agreement_cer"] = [
        float(jiwer.cer(str(m) or ".", str(c) or "."))
        for m, c in zip(frame.mms_luo, frame.clear_luo)
    ]
    frame.to_csv(path, index=False)
    result = {"n": len(frame), "by_threshold": {}, "by_language": {}}
    for threshold in THRESHOLDS:
        accepted = frame.agreement_cer.le(threshold)
        result["by_threshold"][str(threshold)] = {
            "false_accepts": int(accepted.sum()),
            "fpr": float(accepted.mean()),
        }
    for lang, group in frame.groupby("language"):
        result["by_language"][lang] = {
            "n": len(group),
            "min_agreement_cer": float(group.agreement_cer.min()),
            "p10_agreement_cer": float(group.agreement_cer.quantile(0.10)),
            "false_accepts_at_0.10": int(group.agreement_cer.le(0.10).sum()),
            "false_accepts_at_0.15": int(group.agreement_cer.le(0.15).sum()),
        }
    return result


@torch.inference_mode()
def decode_sunbird_sog(model, processor, audio: np.ndarray, device: torch.device) -> str:
    features = processor(
        audio,
        sampling_rate=TARGET_SR,
        do_normalize=True,
        return_tensors="pt",
    ).input_features.to(device)
    forced = [
        (1, SUNBIRD_SOG_TOKEN),
        (2, processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")),
        (3, processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")),
    ]
    ids = model.generate(
        features,
        forced_decoder_ids=forced,
        num_beams=1,
        do_sample=False,
        max_new_tokens=256,
    )
    text = processor.batch_decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return normalize_text(text) or "."


@torch.inference_mode()
def decode_whisper_sog(model, processor, audio: np.ndarray, device: torch.device) -> str:
    features = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(device)
    forced = processor.get_decoder_prompt_ids(language="sw", task="transcribe")
    ids = model.generate(features, forced_decoder_ids=forced, num_beams=1, max_new_tokens=256)
    return normalize_text(processor.batch_decode(ids, skip_special_tokens=True)[0]) or "."


def evaluate_sog(device: torch.device, limit: int | None = None) -> dict:
    """Exact-ID Lusoga A/B: incumbent WAXAL cache vs two local Whisper models."""
    out = OUT / "sog_same_id.csv"
    examples = proxy_examples("sog")
    if limit is not None:
        examples = examples[:limit]
    rows = {
        ex["ID"]: {
            "ID": ex["ID"],
            "language": "sog",
            "reference": ex["reference"],
        }
        for ex in examples
    }
    if out.exists():
        old = pd.read_csv(out)
        for _, row in old.iterrows():
            if str(row.ID) in rows:
                rows[str(row.ID)].update(row.dropna().to_dict())

    cache = json.loads((ROOT / "outputs" / "phase3_routing_hyps_cache.json").read_text())
    for ex in examples:
        cached = cache.get("sog", {}).get(ex["ID"])
        if not cached:
            raise RuntimeError(f"missing incumbent Sog cache for {ex['ID']}")
        rows[ex["ID"]]["waxal_sog"] = normalize_text(str(cached[0])) or "."

    missing = [ex for ex in examples if not str(rows[ex["ID"]].get("sunbird_sog", "")).strip()]
    if missing:
        proc = WhisperProcessor.from_pretrained(SUNBIRD_ID, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            SUNBIRD_ID, local_files_only=True, low_cpu_mem_usage=True
        ).to(device).eval()
        for position, ex in enumerate(missing, 1):
            rows[ex["ID"]]["sunbird_sog"] = decode_sunbird_sog(
                model, proc, ex["audio"], device
            )
            if position % 5 == 0 or position == len(missing):
                pd.DataFrame(rows.values()).to_csv(out, index=False)
                print(f"Sog Sunbird {position}/{len(missing)}", flush=True)
        release(model)

    missing = [ex for ex in examples if not str(rows[ex["ID"]].get("whisper_small_sog", "")).strip()]
    if missing:
        proc = WhisperProcessor.from_pretrained(WHISPER_SOG_ID, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_SOG_ID, local_files_only=True, low_cpu_mem_usage=True
        ).to(device).eval()
        for position, ex in enumerate(missing, 1):
            rows[ex["ID"]]["whisper_small_sog"] = decode_whisper_sog(
                model, proc, ex["audio"], device
            )
            if position % 5 == 0 or position == len(missing):
                pd.DataFrame(rows.values()).to_csv(out, index=False)
                print(f"Sog Whisper-small {position}/{len(missing)}", flush=True)
        release(model)

    frame = pd.DataFrame(rows.values()).sort_values("ID")
    frame.to_csv(out, index=False)
    refs = frame.reference.astype(str).tolist()
    scores = {
        name: metric(refs, frame[name].astype(str).tolist())
        for name in ("waxal_sog", "sunbird_sog", "whisper_small_sog")
    }
    base = scores["waxal_sog"]["zindi"]
    for name in ("sunbird_sog", "whisper_small_sog"):
        scores[name]["delta_vs_spine_model"] = scores[name]["zindi"] - base
    return {"same_ids": True, "sample": "WAXAL sog validation proxy", "scores": scores}


def cached_exact_id_evidence() -> dict:
    """Reconcile existing exact-protocol Luo and Acholi validation runs."""
    luo = json.loads((ROOT / "outputs" / "proxy_luo_fleurs_gate.json").read_text())
    luo_ft = json.loads((ROOT / "outputs" / "next_iter" / "luo_ft_gate.json").read_text())
    ach = json.loads((ROOT / "outputs" / "next_iter" / "ach_ab_v2.json").read_text())
    ach40 = json.loads((ROOT / "outputs" / "proxy_ach_beam_guard.json").read_text())

    luo_results = luo["results"]
    incumbent = luo_results["waxal_ach_beam"]["zindi"]
    luo_summary = {
        "same_ids": True,
        "dataset": luo["dataset"],
        "n": luo["n"],
        "spine_fallback_waxal_ach_beam": luo_results["waxal_ach_beam"],
        "mms1b_luo": luo_results["mms1b_luo"],
        "clear_luo": luo_results["clear_luo_77h"],
        "anv_ft_luo": luo_ft["ft_luo_on_fleurs"],
        "delta_mms1b_vs_spine_fallback": luo_results["mms1b_luo"]["zindi"] - incumbent,
        "delta_clear_vs_spine_fallback": luo_results["clear_luo_77h"]["zindi"] - incumbent,
        "delta_anv_ft_vs_spine_fallback": luo_ft["ft_luo_on_fleurs"]["zindi"] - incumbent,
        "delta_anv_ft_vs_mms1b": (
            luo_ft["ft_luo_on_fleurs"]["zindi"] - luo_results["mms1b_luo"]["zindi"]
        ),
    }

    ach_base = ach["waxal_ach_beam"]["zindi"]
    ach40_base = ach40["waxal_beam_guard"].get(
        "zindi", ach40["waxal_beam_guard"].get("zindi_est")
    )
    ach40_ft = ach40["achft_beam_guard"].get(
        "zindi", ach40["achft_beam_guard"].get("zindi_est")
    )
    ach_summary = {
        "same_ids": True,
        "dataset": "WAXAL ach validation; seed=42",
        "n": int(ach["waxal_ach_beam"]["n"]),
        "spine_waxal_ach_beam": ach["waxal_ach_beam"],
        "salt_mix_v2_beam": ach["waxal-ach-salt-mix-v2_beam"],
        "delta_salt_v2_vs_spine": ach["waxal-ach-salt-mix-v2_beam"]["zindi"] - ach_base,
        "n40_current_recipe": {
            "spine_waxal_ach_beam_guard": ach40["waxal_beam_guard"],
            "waxal_lmhead_ft_beam_guard": ach40["achft_beam_guard"],
            "delta_lmhead_vs_spine": ach40_ft - ach40_base,
        },
    }
    return {"luo": luo_summary, "ach": ach_summary}


def route_inventory() -> dict:
    spine = pd.read_csv(SPINE)
    routes = pd.read_csv(ROUTES)
    openset = pd.read_csv(OPENSET)
    public_ids = set(pd.read_csv(PUBLIC).ID.astype(str))
    old_ids = set(pd.read_csv(ROOT / "data" / "phase2" / "Test.csv").ID.astype(str))
    if len(spine) != 2392 or spine.ID.duplicated().any() or spine.Target.isna().any():
        raise RuntimeError("current submission spine failed expanded Phase-2 integrity checks")
    if set(routes.ID.astype(str)) != old_ids:
        raise RuntimeError("route detail is not the exact old Phase-2 ID set")
    merged = routes[["ID", "lid_lang", "decode_lang", "source"]].merge(
        openset[["ID", "candidates"]], on="ID", how="left", validate="one_to_one"
    )
    return {
        "spine": str(SPINE),
        "spine_sha256": sha256(SPINE),
        "expanded_rows": len(spine),
        "old_rows": len(old_ids),
        "lid_counts_old": merged.lid_lang.value_counts().to_dict(),
        "decode_counts_old": merged.decode_lang.value_counts().to_dict(),
        "source_counts_old": merged.source.value_counts().to_dict(),
        "lid_luo_rows": int(merged.lid_lang.eq("luo").sum()),
        "lid_luo_public_sensitive_intersection": int(
            merged.loc[merged.lid_lang.eq("luo"), "ID"].astype(str).isin(public_ids).sum()
        ),
        "warning": (
            "MMS-LID 'luo' is not a deployable true-language label: its 785 rows are "
            "decoded mainly as Acholi/Luganda/Runyankole by the old open-set system."
        ),
    }


def choose_safe_luo_threshold(negative: dict) -> float | None:
    safe = [
        threshold
        for threshold in THRESHOLDS
        if negative["by_threshold"][str(threshold)]["false_accepts"] == 0
    ]
    return max(safe) if safe else None


def build_private_safe_candidate(threshold: float | None) -> dict:
    spine = pd.read_csv(SPINE)
    spine["ID"] = spine.ID.astype(str)
    ortho = pd.read_csv(ORTHO)
    ortho["ID"] = ortho.ID.astype(str)
    public_ids = set(pd.read_csv(PUBLIC).ID.astype(str))

    if threshold is None:
        selected = ortho.iloc[0:0].copy()
    else:
        selected = ortho.loc[
            ortho.cer_mc.le(threshold)
            & ortho.mms.fillna("").astype(str).str.strip().ne("")
        ].copy()
    current = spine.set_index("ID").Target.fillna("").astype(str)
    selected["before"] = selected.ID.map(current)
    selected["after"] = selected.mms.map(lambda value: normalize_text(str(value)) or ".")
    selected["changed"] = selected.before.map(normalize_text) != selected.after.map(normalize_text)
    selected["public_sensitive"] = selected.ID.isin(public_ids)
    selected = selected.loc[selected.changed & ~selected.public_sensitive].copy()

    # The candidate is intentionally limited to zero-observed-FP Dholuo agreement
    # rows. Acholi SALT and Lusoga model changes are not mixed in unless their
    # exact-ID evidence and route identity both pass independently.
    candidate = spine.copy()
    replacements = dict(zip(selected.ID, selected.after))
    candidate["Target"] = [replacements.get(uid, text) for uid, text in zip(candidate.ID, candidate.Target)]
    out_csv = OUT / "submission_phase2_hidden_routes_private_safe.csv"
    candidate.to_csv(out_csv, index=False)
    selected[
        [
            "ID",
            "decode_lang",
            "p1",
            "cer_mc",
            "before",
            "after",
            "clr",
            "public_sensitive",
        ]
    ].to_csv(OUT / "private_safe_replacements.csv", index=False)

    reread = pd.read_csv(out_csv)
    integrity = {
        "rows": len(reread),
        "columns": reread.columns.tolist(),
        "unique_ids": int(reread.ID.nunique()),
        "empty_targets": int(reread.Target.isna().sum()),
        "same_id_order_as_spine": bool(reread.ID.astype(str).equals(spine.ID.astype(str))),
        "changed_vs_spine": int(
            (~reread.Target.fillna("").astype(str).eq(spine.Target.fillna("").astype(str))).sum()
        ),
    }
    if integrity != {
        "rows": 2392,
        "columns": ["ID", "Target"],
        "unique_ids": 2392,
        "empty_targets": 0,
        "same_id_order_as_spine": True,
        "changed_vs_spine": len(selected),
    }:
        raise RuntimeError(f"candidate integrity failure: {integrity}")
    return {
        "path": str(out_csv),
        "sha256": sha256(out_csv),
        "gate": f"lid=luo pool AND MMS-Luo/CLEAR-Luo CER <= {threshold}",
        "threshold": threshold,
        "n_changed": len(selected),
        "changed_ids": selected.ID.tolist(),
        "public_sensitive_changes": int(selected.public_sensitive.sum()),
        "integrity": integrity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug limit per validation language")
    parser.add_argument("--skip-negative-decode", action="store_true")
    parser.add_argument("--skip-sog-decode", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    report = {
        "task": "hidden/private route robustness",
        "device": str(device),
        "route_inventory": route_inventory(),
        "exact_id_evidence": cached_exact_id_evidence(),
    }

    negative_path = OUT / "luo_detector_negative_controls.csv"
    if args.skip_negative_decode:
        if not negative_path.exists():
            raise RuntimeError("--skip-negative-decode requires an existing negative-control cache")
        frame = pd.read_csv(negative_path)
        negative = {"n": len(frame), "by_threshold": {}, "by_language": {}}
        for threshold in THRESHOLDS:
            accepted = frame.agreement_cer.le(threshold)
            negative["by_threshold"][str(threshold)] = {
                "false_accepts": int(accepted.sum()),
                "fpr": float(accepted.mean()),
            }
        for lang, group in frame.groupby("language"):
            negative["by_language"][lang] = {
                "n": len(group),
                "min_agreement_cer": float(group.agreement_cer.min()),
                "false_accepts_at_0.10": int(group.agreement_cer.le(0.10).sum()),
                "false_accepts_at_0.15": int(group.agreement_cer.le(0.15).sum()),
            }
    else:
        negative = evaluate_luo_detector_negatives(device, args.limit)
    report["luo_negative_controls"] = negative

    if not args.skip_sog_decode:
        report["exact_id_evidence"]["sog"] = evaluate_sog(device, args.limit)
    elif (OUT / "sog_same_id.csv").exists():
        frame = pd.read_csv(OUT / "sog_same_id.csv")
        refs = frame.reference.astype(str).tolist()
        scores = {
            name: metric(refs, frame[name].astype(str).tolist())
            for name in ("waxal_sog", "sunbird_sog", "whisper_small_sog")
        }
        base = scores["waxal_sog"]["zindi"]
        for name in ("sunbird_sog", "whisper_small_sog"):
            scores[name]["delta_vs_spine_model"] = scores[name]["zindi"] - base
        report["exact_id_evidence"]["sog"] = {
            "same_ids": True,
            "sample": "WAXAL sog validation proxy",
            "scores": scores,
        }

    threshold = choose_safe_luo_threshold(negative)
    report["private_safe_candidate"] = build_private_safe_candidate(threshold)

    ach = report["exact_id_evidence"]["ach"]
    luo = report["exact_id_evidence"]["luo"]
    sog = report["exact_id_evidence"].get("sog", {}).get("scores", {})
    report["decisions"] = {
        "luo": {
            "decision": "keep MMS-1B Luo; reject ANV FT text; expand only zero-observed-FP agreement rows",
            "reason": (
                f"MMS-1B gains {luo['delta_mms1b_vs_spine_fallback']:+.6f} on the same 80 IDs; "
                f"ANV FT is {luo['delta_anv_ft_vs_mms1b']:+.6f} behind MMS-1B."
            ),
        },
        "ach": {
            "decision": "keep current WAXAL-Acholi beam",
            "reason": (
                f"SALT mix v2 is {ach['delta_salt_v2_vs_spine']:+.6f} and LM-head FT is "
                f"{ach['n40_current_recipe']['delta_lmhead_vs_spine']:+.6f} versus the incumbent."
            ),
        },
        "sog": {
            "decision": "validation winner reported, but no blind replacement from a one-row cross-language route",
            "reason": (
                "The sole old test row decoded as Sog has MMS-LID=luo; validation ASR quality does not "
                "prove its language identity."
            ),
            "scores": sog,
        },
    }
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
