#!/usr/bin/env python3
"""Forensic audit of WAXAL/Zindi text scoring and submission normalization.

This script is intentionally non-deploying: it reads validation/model artifacts and
writes only under outputs/goal_2026_08_08/scoring_forensics.  It never reads test
references, edits a prediction cache, builds a submission, or uploads anything.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "goal_2026_08_08" / "scoring_forensics"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", str(OUT / "hf_datasets_cache"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from rapidfuzz.distance import Levenshtein
from transformers import AutoProcessor, Wav2Vec2BertForCTC

from src.dataset import load_hf_asr_split


SOURCES = {
    "competition": "https://zindi.world/competitions/google-waxal-asr-challenge",
    "competition_data": "https://zindi.world/competitions/google-waxal-asr-challenge/data",
    "zindi_wer": "https://zindi.world/learn/zindi-error-metric-series-what-is-word-error-rate",
    "zindi_weighted": "https://zindi.world/learn/evaluating-language-generation-on-zindi-a-guide-to-weighted-wer-and-cer",
    "zindi_multimetric": "https://zindi.world/learn/introducing-multi-metric-evaluation-or-one-metric-to-rule-them-all",
    "lin_card": "https://huggingface.co/sulaimank/w2vbert-lingala-sd3",
    "sna_card": "https://huggingface.co/sulaimank/w2vbert-shona-sd2",
}

SPECS = {
    "lin": {
        "model": ROOT / "checkpoints" / "sulaimank-w2vbert-lingala-sd3",
        "normalized": ROOT / "outputs" / "goal_2026_08_08" / "sulaiman_public_descendants" / "validation_w2vbert-lingala-sd3.csv",
        "manifest": ROOT / "outputs" / "goal_2026_08_08" / "sulaiman_public_descendants" / "validation_manifest_lin.csv",
        "tag": "w2vbert-lingala-sd3",
    },
    "sna": {
        "model": ROOT / "checkpoints" / "sulaimank-w2vbert-shona-sd2",
        "normalized": ROOT / "outputs" / "goal_2026_08_08" / "shona_sd2_parallel" / "validation_w2vbert-shona-sd2.csv",
        "manifest": ROOT / "outputs" / "goal_2026_08_08" / "shona_sd2_parallel" / "validation_manifest_sna.csv",
        "tag": "w2vbert-shona-sd2",
    },
}

OFFICIAL_COMPONENT_EXAMPLES = [
    {"name": "user_best", "score": 0.689237520, "wer": 0.473331032, "cer": 0.148193926},
    {"name": "leader", "score": 0.767985072, "wer": 0.357945677, "cer": 0.106084176},
]

_WS = re.compile(r"\s+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def keep(text: Any) -> str:
    """Preserve case and punctuation; canonicalize Unicode/control/whitespace only."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = "".join(ch for ch in value if unicodedata.category(ch) not in {"Cc", "Cf"})
    return _WS.sub(" ", value).strip()


def strip_punctuation(text: Any) -> str:
    value = keep(text)
    # Match the repository cache transform exactly: preserve apostrophes and
    # map every other punctuation mark to a word boundary before collapsing
    # whitespace. This is intentionally audited, not assumed to be official.
    value = "".join(
        ch if (ch == "'" or not unicodedata.category(ch).startswith("P")) else " "
        for ch in value
    )
    return _WS.sub(" ", value).strip()


def lower_keep_punctuation(text: Any) -> str:
    return keep(text).lower()


def lower_strip(text: Any) -> str:
    return strip_punctuation(text).lower()


REGIMES: dict[str, Callable[[Any], str]] = {
    "keep": keep,
    "strip_preserve_case": strip_punctuation,
    "lower_preserve_punctuation": lower_keep_punctuation,
    "lower_strip": lower_strip,
}


def corpus_metric(refs: list[str], hyps: list[str]) -> dict[str, float]:
    word_edits = 0
    char_edits = 0
    ref_words = 0
    ref_chars = 0
    row_wer: list[float] = []
    row_cer: list[float] = []
    word_sqrt_num = word_sqrt_den = 0.0
    char_sqrt_num = char_sqrt_den = 0.0
    for ref, hyp in zip(refs, hyps):
        rw, hw = ref.split(), hyp.split()
        we = int(Levenshtein.distance(rw, hw))
        ce = int(Levenshtein.distance(ref, hyp))
        nw, nc = len(rw), len(ref)
        if nw == 0 or nc == 0:
            raise ValueError("empty validation reference after transform")
        word_edits += we
        char_edits += ce
        ref_words += nw
        ref_chars += nc
        row_wer.append(we / nw)
        row_cer.append(ce / nc)
        ww, cw = math.sqrt(nw), math.sqrt(nc)
        word_sqrt_num += (we / nw) * ww
        word_sqrt_den += ww
        char_sqrt_num += (ce / nc) * cw
        char_sqrt_den += cw
    wer = word_edits / ref_words
    cer = char_edits / ref_chars
    return {
        "wer": wer,
        "cer": cer,
        "zindi": 1.0 - 0.5 * wer - 0.5 * cer,
        "word_edits": word_edits,
        "char_edits": char_edits,
        "reference_words": ref_words,
        "reference_characters": ref_chars,
        "macro_wer": float(np.mean(row_wer)),
        "macro_cer": float(np.mean(row_cer)),
        "sqrt_weighted_mean_wer": word_sqrt_num / word_sqrt_den,
        "sqrt_weighted_mean_cer": char_sqrt_num / char_sqrt_den,
    }


def documented_evidence_metric(raw_refs: list[str], submitted_hyps: list[str]) -> dict[str, float]:
    """Conservative emulation of only the preprocessing Zindi documents.

    Zindi says punctuation is excluded from WER, so WER strips punctuation on
    both sides. No source found says CER strips punctuation or that either
    metric lowercases, so CER keeps raw characters and case.
    """
    word_part = corpus_metric(
        [strip_punctuation(x) for x in raw_refs],
        [strip_punctuation(x) for x in submitted_hyps],
    )
    char_part = corpus_metric([keep(x) for x in raw_refs], [keep(x) for x in submitted_hyps])
    wer, cer = word_part["wer"], char_part["cer"]
    return {
        "wer": wer,
        "cer": cer,
        "zindi": 1.0 - 0.5 * wer - 0.5 * cer,
        "word_edits": word_part["word_edits"],
        "char_edits": char_part["char_edits"],
        "reference_words": word_part["reference_words"],
        "reference_characters": char_part["reference_characters"],
        "macro_wer": word_part["macro_wer"],
        "macro_cer": char_part["macro_cer"],
        "sqrt_weighted_mean_wer": word_part["sqrt_weighted_mean_wer"],
        "sqrt_weighted_mean_cer": char_part["sqrt_weighted_mean_cer"],
    }


def raw_reference_map(lang: str) -> dict[str, str]:
    table = pq.read_table(
        ROOT / "data" / "hf_metadata" / f"{lang}_validation.parquet",
        columns=["ID", "Target"],
    ).to_pandas()
    if table.ID.duplicated().any() or table.Target.isna().any():
        raise RuntimeError(f"invalid raw-reference metadata for {lang}")
    return dict(zip(table.ID.astype(str), table.Target.astype(str)))


def normalize_audio(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim > 1:
        value = value.mean(axis=-1)
    return value / float(np.max(np.abs(value)) + 1e-9)


def exact_examples(lang: str, ids: list[str]) -> list[dict[str, Any]]:
    dataset = load_hf_asr_split(lang, "validation")
    dataset_ids = [str(value) for value in dataset["id"]]
    index = {uid: pos for pos, uid in enumerate(dataset_ids)}
    missing = [uid for uid in ids if uid not in index]
    if missing:
        raise RuntimeError(f"{lang}: missing validation IDs: {missing[:5]}")
    rows = []
    for uid in ids:
        ex = dict(dataset[index[uid]])
        audio = ex["audio"]
        if int(audio.get("sampling_rate") or 16000) != 16000:
            raise RuntimeError(f"{uid}: unexpected sample rate")
        rows.append({"ID": uid, "array": normalize_audio(audio["array"])})
    return rows


def existing_candidate(lang: str) -> pd.DataFrame:
    spec = SPECS[lang]
    detail = pd.read_csv(spec["normalized"], dtype={"ID": str})
    if "candidate" not in detail or len(detail) != 80 or detail.ID.nunique() != 80:
        raise RuntimeError(f"{lang}: expected complete 80-row normalized candidate cache")
    manifest = pd.read_csv(spec["manifest"], dtype={"ID": str})
    if detail.ID.tolist() != manifest.ID.tolist():
        raise RuntimeError(f"{lang}: candidate/manifest order mismatch")
    return detail[["ID", "candidate"]].rename(columns={"candidate": "cached_lower_strip"})


@torch.inference_mode()
def decode_raw(lang: str, batch_size: int, device: torch.device) -> pd.DataFrame:
    path = OUT / f"raw_hypotheses_{lang}.csv"
    existing = existing_candidate(lang)
    ids = existing.ID.tolist()
    done: dict[str, str] = {}
    if path.is_file():
        partial = pd.read_csv(path, dtype={"ID": str, "raw_hypothesis": str})
        if partial.ID.duplicated().any() or not set(partial.ID).issubset(set(ids)):
            raise RuntimeError(f"stale raw cache: {path}")
        done = dict(zip(partial.ID, partial.raw_hypothesis.fillna("")))
    todo_ids = [uid for uid in ids if uid not in done]
    if todo_ids:
        examples = exact_examples(lang, todo_ids)
        model_path = Path(SPECS[lang]["model"])
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        model = Wav2Vec2BertForCTC.from_pretrained(
            str(model_path), local_files_only=True, low_cpu_mem_usage=True
        ).to(device).eval()
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            inputs = processor(
                [row["array"] for row in chunk],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            kwargs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
            token_ids = torch.argmax(model(**kwargs).logits, dim=-1).detach().cpu()
            texts = processor.batch_decode(token_ids)
            for row, text in zip(chunk, texts):
                done[row["ID"]] = keep(text)
            pd.DataFrame(
                [{"ID": uid, "raw_hypothesis": done[uid]} for uid in ids if uid in done]
            ).to_csv(path, index=False)
            print(f"{lang}: raw decode {len(done)}/80", flush=True)
        del model, processor
        gc.collect()
    frame = pd.DataFrame([{"ID": uid, "raw_hypothesis": done[uid]} for uid in ids])
    if len(frame) != 80 or frame.ID.nunique() != 80 or frame.raw_hypothesis.eq("").any():
        raise RuntimeError(f"{lang}: incomplete raw decode")
    return frame


def text_profile(values: pd.Series) -> dict[str, Any]:
    strings = values.astype(str).tolist()
    punctuation = {}
    for value in strings:
        for char in value:
            if unicodedata.category(char).startswith("P"):
                punctuation[char] = punctuation.get(char, 0) + 1
    return {
        "rows": len(strings),
        "uppercase_rows": sum(any(ch.isupper() for ch in value) for value in strings),
        "punctuation_rows": sum(any(unicodedata.category(ch).startswith("P") for ch in value) for value in strings),
        "punctuation_counts": dict(sorted(punctuation.items())),
        "leading_or_trailing_space_rows": sum(value != value.strip() for value in strings),
        "multiple_whitespace_rows": sum(bool(re.search(r"\s{2,}", value)) for value in strings),
    }


def evaluate_language(lang: str, raw_hyps: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    cached = existing_candidate(lang)
    refs = raw_reference_map(lang)
    detail = cached.merge(raw_hyps, on="ID", validate="one_to_one")
    detail["raw_reference"] = detail.ID.map(refs)
    if detail.raw_reference.isna().any():
        raise RuntimeError(f"{lang}: missing raw references")
    detail["raw_to_cached_lower_strip"] = detail.raw_hypothesis.map(lower_strip)
    detail["cache_exact_after_lower_strip"] = detail.raw_to_cached_lower_strip.eq(detail.cached_lower_strip)
    if not detail.cache_exact_after_lower_strip.all():
        bad = detail.loc[~detail.cache_exact_after_lower_strip, "ID"].tolist()
        raise RuntimeError(f"{lang}: raw decode does not reproduce normalized cache: {bad[:5]}")

    metrics: list[dict[str, Any]] = []
    raw_refs = detail.raw_reference.astype(str).tolist()
    raw_preds = detail.raw_hypothesis.astype(str).tolist()
    cached_preds = detail.cached_lower_strip.astype(str).tolist()
    for regime, transform in REGIMES.items():
        # Symmetric score answers the model-card/normalized-evaluation question.
        sym = corpus_metric([transform(x) for x in raw_refs], [transform(x) for x in raw_preds])
        metrics.append({"language": lang, "comparison": "symmetric", "regime": regime, **sym})
        # Fixed raw reference answers which form is safest if the leaderboard keeps text.
        sub = corpus_metric([keep(x) for x in raw_refs], [transform(x) for x in raw_preds])
        metrics.append({"language": lang, "comparison": "raw_reference_submission_counterfactual", "regime": regime, **sub})
        documented = documented_evidence_metric(raw_refs, [transform(x) for x in raw_preds])
        metrics.append({"language": lang, "comparison": "documented_zindi_emulation", "regime": regime, **documented})
    cached_score = corpus_metric([keep(x) for x in raw_refs], [keep(x) for x in cached_preds])
    raw_score = corpus_metric([keep(x) for x in raw_refs], [keep(x) for x in raw_preds])
    cached_documented = documented_evidence_metric(raw_refs, cached_preds)
    raw_documented = documented_evidence_metric(raw_refs, raw_preds)
    loss = {
        "raw_hypothesis_against_raw_reference": raw_score,
        "existing_lower_strip_cache_against_raw_reference": cached_score,
        "raw_minus_cache_zindi": raw_score["zindi"] - cached_score["zindi"],
        "documented_zindi_emulation_raw": raw_documented,
        "documented_zindi_emulation_existing_cache": cached_documented,
        "documented_zindi_emulation_raw_minus_cache": raw_documented["zindi"] - cached_documented["zindi"],
        "cache_rows_reproduced_from_raw": int(detail.cache_exact_after_lower_strip.sum()),
    }
    detail.to_csv(OUT / f"matched_forensics_{lang}.csv", index=False)
    return metrics, loss, detail


def model_card_metrics() -> dict[str, Any]:
    return {
        "lin": {
            "wer_keep": 0.1038,
            "cer_keep": 0.0301,
            "zindi_keep": 0.9331,
            "wer_strip": 0.0530,
            "zindi_strip": 0.9663,
            "zindi_lower": 0.9746,
            "source": SOURCES["lin_card"],
        },
        "sna": {
            "wer_keep": 0.1163,
            "cer_keep": 0.0183,
            "zindi_keep": 0.9327,
            "wer_strip": 0.0318,
            "zindi_strip": 0.9820,
            "zindi_lower": 0.9977,
            "source": SOURCES["sna_card"],
        },
    }


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# WAXAL/Zindi scoring and normalization forensics",
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "## What is official and what is not",
        "",
        "- The competition specifies a 50/50 mean of WER and CER. The observed leaderboard components reproduce `score = 1 - 0.5*WER - 0.5*CER` to rounding precision.",
        "- Zindi's WER explainer says punctuation is not included in WER, while spelling and diacritics matter. It does not say case is folded.",
        "- Zindi's language-generation metric article documents automatic removal of conversational role tags; it does not document automatic lowercasing or general punctuation removal for CER.",
        "- The official starter notebook is listed on the competition data page, but its download endpoint returned HTTP 401 without a joined/authenticated Zindi session. Therefore no undocumented notebook transform is asserted here.",
        "- `src.text_norm` is not an official artifact. Its lowercase-plus-partial-punctuation-strip behavior must not be presented as leaderboard preprocessing.",
        "",
        "## Exact matched validation",
        "",
        "Scores below use corpus Levenshtein WER/CER and the leaderboard complement. `symmetric` transforms both reference and hypothesis; `raw_reference_submission_counterfactual` keeps the reference and changes only what would be submitted.",
        "",
        "| Language | Comparison | Regime | WER | CER | Zindi |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["metrics"]:
        lines.append(
            f"| {row['language']} | {row['comparison']} | {row['regime']} | "
            f"{row['wer']:.6f} | {row['cer']:.6f} | {row['zindi']:.6f} |"
        )
    lines += ["", "## Existing cache loss", ""]
    for lang, loss in payload["cache_loss"].items():
        lines.append(
            f"- `{lang}`: raw model output vs raw reference `{loss['raw_hypothesis_against_raw_reference']['zindi']:.6f}`; "
            f"existing lower/strip cache vs raw reference `{loss['existing_lower_strip_cache_against_raw_reference']['zindi']:.6f}`; "
            f"raw-minus-cache `{loss['raw_minus_cache_zindi']:+.6f}`."
        )
        lines.append(
            f"  Under documented-Zindi emulation (punctuation stripped for WER, raw CER), "
            f"raw `{loss['documented_zindi_emulation_raw']['zindi']:.6f}` vs cache "
            f"`{loss['documented_zindi_emulation_existing_cache']['zindi']:.6f}`; "
            f"delta `{loss['documented_zindi_emulation_raw_minus_cache']:+.6f}`."
        )
    lines += [
        "",
        "## Raw-reference profile",
        "",
        "The official validation text contains material capitalization and punctuation; these are not cosmetic if CER keeps raw characters.",
        "",
        "```json",
        json.dumps(payload["profiles"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Model-card evidence",
        "",
        "Both specialist cards explicitly report much better `Strip` and `Lower` evaluation variants than `Keep`. Those are diagnostic variants, not evidence that Zindi applies those transforms. The cards do not disclose the evaluation set or transform implementation.",
        "",
        "```json",
        json.dumps(payload["model_cards"], indent=2, sort_keys=True),
        "```",
        "",
        "## Sources",
        "",
    ]
    lines.extend(f"- [{name}]({url})" for name, url in SOURCES.items())
    lines += [
        "",
        "## Reproducibility",
        "",
        f"- Script: `{Path(__file__).relative_to(ROOT)}`",
        "- Validation IDs: the existing immutable 80-row matched manifests for each language.",
        "- No test references read; no prediction/submission artifacts modified or produced.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-decode", action="store_true", help="require completed raw caches")
    args = parser.parse_args()
    if args.device not in {"cpu", "mps", "cuda"}:
        raise ValueError("unsupported device")
    if not 1 <= args.batch_size <= 8:
        raise ValueError("batch size must be 1..8")
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    formula_checks = []
    for row in OFFICIAL_COMPONENT_EXAMPLES:
        reproduced = 1.0 - 0.5 * row["wer"] - 0.5 * row["cer"]
        formula_checks.append({**row, "reproduced": reproduced, "absolute_error": abs(row["score"] - reproduced)})
    if max(row["absolute_error"] for row in formula_checks) > 1e-8:
        raise RuntimeError("leaderboard formula check failed")

    all_metrics: list[dict[str, Any]] = []
    losses: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    artifact_hashes: dict[str, str] = {}
    for lang in ("lin", "sna"):
        raw_path = OUT / f"raw_hypotheses_{lang}.csv"
        if args.skip_decode and not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        raw = decode_raw(lang, args.batch_size, device)
        metrics, loss, detail = evaluate_language(lang, raw)
        all_metrics.extend(metrics)
        losses[lang] = loss
        profiles[lang] = {
            "matched_raw_references": text_profile(detail.raw_reference),
            "raw_model_hypotheses": text_profile(detail.raw_hypothesis),
            "full_validation_raw_references": text_profile(pd.Series(raw_reference_map(lang).values())),
        }
        artifact_hashes[raw_path.name] = sha256(raw_path)
        artifact_hashes[f"matched_forensics_{lang}.csv"] = sha256(OUT / f"matched_forensics_{lang}.csv")

    metric_frame = pd.DataFrame(all_metrics)
    metric_frame.to_csv(OUT / "metrics_by_normalization.csv", index=False)
    payload = {
        "decision": (
            "Submit punctuation with original model capitalization when raw specialist output is available. "
            "Do not lowercase. Zindi explicitly excludes punctuation from WER, but no authoritative source "
            "establishes lowercase folding, and CER may still score punctuation. Existing `src.text_norm` "
            "lower/strip caches are not faithful official-score artifacts; the measured raw-vs-cache deltas "
            "below quantify the loss under a raw CER/case-sensitive scorer."
        ),
        "official_formula_checks": formula_checks,
        "authoritative_preprocessing": {
            "wer_punctuation": "excluded according to Zindi WER explainer",
            "case_folding": "not documented; assume case-sensitive",
            "cer_punctuation": "not documented as stripped; assume retained",
            "automatic_general_cleanup": "only conversational role tags explicitly documented",
            "starter_notebook": "listed but HTTP 401 without authenticated/joined Zindi session",
        },
        "metrics": all_metrics,
        "cache_loss": losses,
        "profiles": profiles,
        "model_cards": model_card_metrics(),
        "sources": SOURCES,
        "artifact_hashes": artifact_hashes,
        "safety": {"test_references_read": False, "submissions_built": False, "uploads": False},
    }
    (OUT / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    artifact_hashes["metrics_by_normalization.csv"] = sha256(OUT / "metrics_by_normalization.csv")
    write_report(payload)
    print(json.dumps({"decision": payload["decision"], "cache_loss": losses}, indent=2))


if __name__ == "__main__":
    main()
