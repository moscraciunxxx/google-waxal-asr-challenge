#!/usr/bin/env python3
"""Phase-3 routing / multi-hyp features on labeled proxy (n=200).

Systematically tests routing features that map to deployable Phase-2 LID rules.

Features (oracle / labeled proxy):
  1. oracle_lang           — true-lang waxal only
  2. multihyp_conf         — true-lang candidate set, max CTC mean logprob
  3. hard_true             — same as oracle_lang
  4. length_guard_multihyp — drop length-outlier hyps, then max conf
  5. margin_multihyp       — require conf margin; oracle or first-cand fallback
  6. ft_lug_when_true_lug  — FT-v2 on true lug, else true-lang waxal

Candidate map (multihyp by true_lang):
  lug: lug|nyn|sog
  ach: ach|lug|sog
  nyn: nyn|lug
  sog: sog|lug
  mas: mas|lug

Deliverables:
  outputs/phase3_routing.json
  outputs/phase3_routing.md
  outputs/phase3_routing_detail.csv  (optional diagnostic)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CHECKPOINT_DIR,
    DATA_DIR,
    FORBIDDEN_TRAIN_SPLITS,
    HF_DATASET,
    OUTPUT_DIR,
    SEED,
    TARGET_SR,
)
from src.metrics import score_by_language, score_pairs  # noqa: E402
from src.mms_infer import pick_device, transcribe_waveform  # noqa: E402
from src.text_norm import normalize_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase3_routing")

PROXY_INDEX = DATA_DIR / "proxy_val_index.csv"
PROXY_LANGS = ("ach", "nyn", "lug", "sog", "mas")

WAXAL300 = {
    lang: f"waxal-benchmarking/mms-300m-waxal-{lang}"
    for lang in ("ach", "lug", "nyn", "sog", "mas", "sna", "lin")
}
FT_LUG = CHECKPOINT_DIR / "mms-lug-ft-v2"

# Phase-3 multihyp candidate map (by true language)
MULTI_CANDS: dict[str, list[str]] = {
    "lug": ["lug", "nyn", "sog"],
    "ach": ["ach", "lug", "sog"],
    "nyn": ["nyn", "lug"],
    "sog": ["sog", "lug"],
    "mas": ["mas", "lug"],
}

MARGIN_THRESH = 0.01
LENGTH_LO = 0.5
LENGTH_HI = 2.0


def set_seed(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parquet_paths_for_lang(lang: str) -> list[str]:
    summary = DATA_DIR / "proxy_val_index.summary.json"
    if summary.exists():
        try:
            s = json.loads(summary.read_text())
            paths = s.get("per_lang", {}).get(lang, {}).get("parquet_files") or []
            paths = [p for p in paths if Path(p).is_file()]
            if paths:
                return paths
        except Exception as e:
            logger.warning("summary parquet resolve failed: %s", e)
    hub_ds = HF_DATASET.replace("/", "--")
    cache_root = Path(
        os.environ.get("HF_HUB_CACHE") or (Path.home() / ".cache/huggingface/hub")
    )
    snap_root = cache_root / f"datasets--{hub_ds}" / "snapshots"
    needle = f"{lang}-validation-"
    if snap_root.is_dir():
        for snap in sorted(snap_root.iterdir(), reverse=True):
            asr_dir = snap / "data" / "ASR" / lang
            if not asr_dir.is_dir():
                continue
            found = sorted(
                str(p.resolve())
                for p in asr_dir.glob("*.parquet")
                if needle in p.name and p.resolve().is_file()
            )
            if found:
                return found
    return []


def load_proxy_clips(index_csv: Path) -> list[dict]:
    if not index_csv.exists():
        raise SystemExit(f"Missing {index_csv}")
    idx = pd.read_csv(index_csv)
    for col in ("id", "language", "split", "transcription"):
        if col not in idx.columns:
            raise SystemExit(f"proxy index missing column {col}")
    if (idx["split"].astype(str) != "validation").any():
        raise SystemExit("proxy index must be validation-only (no test gold)")

    from datasets import Audio
    from src.dataset import _decode_audio_item

    clips: list[dict] = []
    for lang, g in idx.groupby("language"):
        lang = str(lang)
        if lang in FORBIDDEN_TRAIN_SPLITS:
            raise SystemExit(f"forbidden lang/split key {lang}")
        want_ids = set(g["id"].astype(str))
        ref_by_id = {
            str(r.id): normalize_text(r.transcription)
            for r in g.itertuples(index=False)
        }
        logger.info("Loading validation audio for %s (need %d ids)", lang, len(want_ids))
        paths = _parquet_paths_for_lang(lang)
        if paths:
            logger.info("%s: parquet %s", lang, paths)
            ds = load_dataset("parquet", data_files={"validation": paths}, split="validation")
        else:
            config = f"{lang}_asr"
            logger.info("%s: fallback HF config %s", lang, config)
            ds = load_dataset(HF_DATASET, config, split="validation")
        try:
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception:
            pass

        found = 0
        for i in tqdm(range(len(ds)), desc=f"load-{lang}"):
            ex = ds[i]
            uid = str(ex.get("id") or ex.get("ID") or "")
            if uid not in want_ids:
                continue
            audio = _decode_audio_item(ex["audio"], TARGET_SR)
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            ref = ref_by_id.get(uid) or normalize_text(
                ex.get("transcription") or ex.get("text") or ""
            )
            if not ref:
                continue
            clips.append(
                {
                    "id": uid,
                    "language": lang,
                    "array": arr,
                    "sr": sr,
                    "ref": ref,
                }
            )
            found += 1
            if found >= len(want_ids):
                break
        missing = want_ids - {c["id"] for c in clips if c["language"] == lang}
        if missing:
            logger.warning(
                "%s: %d ids missing from HF val: %s",
                lang,
                len(missing),
                list(missing)[:5],
            )
        logger.info("%s: loaded %d/%d", lang, found, len(want_ids))
    clips.sort(key=lambda c: (c["language"], c["id"]))
    return clips


def free_model(model) -> None:
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


def load_waxal(lang: str, device: torch.device):
    mid = WAXAL300[lang]
    logger.info("Loading %s on %s (local_files_only)", mid, device)
    try:
        proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
    except Exception as e:
        logger.warning("local load failed (%s); retry with network", e)
        proc = AutoProcessor.from_pretrained(mid)
        model = Wav2Vec2ForCTC.from_pretrained(mid)
    model.to(device).eval()
    return model, proc


def load_ft_lug(device: torch.device):
    mid = str(FT_LUG)
    logger.info("Loading FT-lug %s on %s (local_files_only)", mid, device)
    try:
        proc = AutoProcessor.from_pretrained(mid, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
    except Exception as e:
        logger.warning("local FT load failed (%s); retry", e)
        proc = AutoProcessor.from_pretrained(mid)
        model = Wav2Vec2ForCTC.from_pretrained(mid)
    model.to(device).eval()
    return model, proc


def clips_needed_for_model(clips: list[dict], mlang: str) -> list[dict]:
    """Only decode clips where mlang is true_lang or in MULTI_CANDS[true_lang]."""
    out = []
    for c in clips:
        true = c["language"]
        if mlang == true or mlang in MULTI_CANDS.get(true, []):
            out.append(c)
    return out


@torch.inference_mode()
def score_models(
    clips: list[dict],
    model_langs: list[str],
    device: torch.device,
    include_ft_lug: bool = True,
) -> dict[str, dict[str, tuple[str, float]]]:
    """results[model_key][clip_id] = (hyp, conf). model_key is lang or 'ft_lug'."""
    results: dict[str, dict[str, tuple[str, float]]] = {}
    for mlang in model_langs:
        if mlang not in WAXAL300:
            logger.warning("skip unknown model lang %s", mlang)
            continue
        subset = clips_needed_for_model(clips, mlang)
        model, proc = load_waxal(mlang, device)
        bucket: dict[str, tuple[str, float]] = {}
        for c in tqdm(subset, desc=f"decode-{mlang}(n={len(subset)})"):
            hyp, conf = transcribe_waveform(
                model,
                proc,
                c["array"],
                c["sr"],
                device=device,
                return_confidence=True,
            )
            bucket[c["id"]] = (hyp, float(conf))
        results[mlang] = bucket
        free_model(model)
        del proc
        gc.collect()
        logger.info("done model %s n=%d", mlang, len(bucket))

    if include_ft_lug and FT_LUG.is_dir():
        model, proc = load_ft_lug(device)
        bucket = {}
        # Only lug clips needed for ft_lug_when_true_lug (+ conf-scale on lug)
        subset = [c for c in clips if c["language"] == "lug"]
        for c in tqdm(subset, desc=f"decode-ft_lug(n={len(subset)})"):
            hyp, conf = transcribe_waveform(
                model,
                proc,
                c["array"],
                c["sr"],
                device=device,
                return_confidence=True,
            )
            bucket[c["id"]] = (hyp, float(conf))
        results["ft_lug"] = bucket
        free_model(model)
        del proc
        gc.collect()
        logger.info("done model ft_lug n=%d", len(bucket))
    return results


def word_count(text: str) -> int:
    t = (text or "").strip()
    if not t or t == ".":
        return 0
    return len(t.split())


def gather_cands(
    clip_id: str,
    candidates: list[str],
    results: dict[str, dict[str, tuple[str, float]]],
) -> list[tuple[str, str, float]]:
    """Return list of (lang, hyp, conf) for available candidates."""
    out = []
    for lang in candidates:
        if lang not in results or clip_id not in results[lang]:
            continue
        hyp, conf = results[lang][clip_id]
        out.append((lang, hyp, float(conf)))
    return out


def pick_max_conf(
    items: list[tuple[str, str, float]],
) -> tuple[str, str, float] | None:
    if not items:
        return None
    return max(items, key=lambda x: x[2])


def apply_length_guard(
    items: list[tuple[str, str, float]],
    lo: float = LENGTH_LO,
    hi: float = LENGTH_HI,
) -> list[tuple[str, str, float]]:
    if len(items) <= 1:
        return items
    counts = [word_count(h) for _, h, _ in items]
    med = median(counts) if counts else 0
    if med <= 0:
        # all empty-ish: keep as-is
        return items
    kept = []
    for item, wc in zip(items, counts):
        if wc < lo * med or wc > hi * med:
            continue
        kept.append(item)
    return kept if kept else items  # never empty-out


def metrics_block(refs, hyps, langs, detail_rows, name: str) -> dict:
    m = score_by_language(refs, hyps, langs)
    overall = m["overall"]
    per_lang = {
        k: {
            "wer": v["wer"],
            "cer": v["cer"],
            "score": v["score"],
            "error": v["score"],
            "zindi_est": 1.0 - v["score"],
            "n": int(v["n"]),
        }
        for k, v in m.items()
        if k != "overall"
    }
    route_acc = (
        float(np.mean([r["route_match"] for r in detail_rows])) if detail_rows else 0.0
    )
    return {
        "method": name,
        "n": len(detail_rows),
        "wer": overall["wer"],
        "cer": overall["cer"],
        "score": overall["score"],
        "error": overall["score"],
        "zindi_est": 1.0 - overall["score"],
        "route_acc": route_acc,
        "decode_mass": dict(Counter(r["decode_lang"] for r in detail_rows)),
        "per_lang": per_lang,
    }


def evaluate_oracle(clips, results) -> tuple[dict, list[dict]]:
    refs, hyps, langs, rows = [], [], [], []
    for c in clips:
        lang = c["language"]
        hyp, conf = results[lang][c["id"]]
        refs.append(c["ref"])
        hyps.append(hyp)
        langs.append(lang)
        rows.append(
            {
                "id": c["id"],
                "true_lang": lang,
                "method": "oracle_lang",
                "candidates": lang,
                "decode_lang": lang,
                "confidence": conf,
                "margin": None,
                "ref": c["ref"],
                "hyp": hyp,
                "route_match": 1,
                "note": "",
            }
        )
    return metrics_block(refs, hyps, langs, rows, "oracle_lang"), rows


def evaluate_multihyp(clips, results, name="multihyp_conf") -> tuple[dict, list[dict]]:
    refs, hyps, langs, rows = [], [], [], []
    for c in clips:
        true = c["language"]
        cands = [x for x in MULTI_CANDS.get(true, [true]) if x in results]
        items = gather_cands(c["id"], cands, results)
        pick = pick_max_conf(items)
        if pick is None:
            hyp, dlang, conf = ".", true, -1e9
        else:
            dlang, hyp, conf = pick
        refs.append(c["ref"])
        hyps.append(hyp)
        langs.append(true)
        rows.append(
            {
                "id": c["id"],
                "true_lang": true,
                "method": name,
                "candidates": "|".join(cands),
                "decode_lang": dlang,
                "confidence": conf,
                "margin": None,
                "ref": c["ref"],
                "hyp": hyp,
                "route_match": int(dlang == true),
                "note": "",
            }
        )
    return metrics_block(refs, hyps, langs, rows, name), rows


def evaluate_length_guard(clips, results) -> tuple[dict, list[dict]]:
    name = "length_guard_multihyp"
    refs, hyps, langs, rows = [], [], [], []
    for c in clips:
        true = c["language"]
        cands = [x for x in MULTI_CANDS.get(true, [true]) if x in results]
        items = gather_cands(c["id"], cands, results)
        kept = apply_length_guard(items)
        pick = pick_max_conf(kept)
        if pick is None:
            hyp, dlang, conf = ".", true, -1e9
            note = "empty"
        else:
            dlang, hyp, conf = pick
            dropped = len(items) - len(kept)
            note = f"kept={len(kept)}/{len(items)} dropped={dropped}"
        refs.append(c["ref"])
        hyps.append(hyp)
        langs.append(true)
        rows.append(
            {
                "id": c["id"],
                "true_lang": true,
                "method": name,
                "candidates": "|".join(cands),
                "decode_lang": dlang,
                "confidence": conf,
                "margin": None,
                "ref": c["ref"],
                "hyp": hyp,
                "route_match": int(dlang == true),
                "note": note,
            }
        )
    return metrics_block(refs, hyps, langs, rows, name), rows


def evaluate_margin(
    clips,
    results,
    *,
    oracle_fallback: bool,
    name: str,
    thresh: float = MARGIN_THRESH,
) -> tuple[dict, list[dict]]:
    refs, hyps, langs, rows = [], [], [], []
    for c in clips:
        true = c["language"]
        cands = [x for x in MULTI_CANDS.get(true, [true]) if x in results]
        items = gather_cands(c["id"], cands, results)
        items_sorted = sorted(items, key=lambda x: x[2], reverse=True)
        if not items_sorted:
            hyp, dlang, conf, margin, note = ".", true, -1e9, None, "empty"
        elif len(items_sorted) == 1:
            dlang, hyp, conf = items_sorted[0]
            margin = None
            note = "single"
        else:
            best = items_sorted[0]
            second = items_sorted[1]
            margin = best[2] - second[2]
            if margin >= thresh:
                dlang, hyp, conf = best
                note = "margin_ok"
            else:
                if oracle_fallback:
                    # use true-lang hyp (ceiling)
                    if true in results and c["id"] in results[true]:
                        hyp, conf = results[true][c["id"]]
                        dlang = true
                        note = "fallback_oracle_true"
                    else:
                        dlang, hyp, conf = best
                        note = "fallback_oracle_missing_true"
                else:
                    # first candidate without oracle (stable deployable: prefer listed order)
                    first_lang = cands[0]
                    found = next((it for it in items if it[0] == first_lang), None)
                    if found is None:
                        dlang, hyp, conf = best
                        note = "fallback_first_missing"
                    else:
                        dlang, hyp, conf = found
                        note = "fallback_first_cand"
        refs.append(c["ref"])
        hyps.append(hyp)
        langs.append(true)
        rows.append(
            {
                "id": c["id"],
                "true_lang": true,
                "method": name,
                "candidates": "|".join(cands),
                "decode_lang": dlang,
                "confidence": conf,
                "margin": margin,
                "ref": c["ref"],
                "hyp": hyp,
                "route_match": int(dlang == true),
                "note": note,
            }
        )
    return metrics_block(refs, hyps, langs, rows, name), rows


def evaluate_ft_lug_when_true(clips, results) -> tuple[dict, list[dict]]:
    name = "ft_lug_when_true_lug"
    refs, hyps, langs, rows = [], [], [], []
    has_ft = "ft_lug" in results
    for c in clips:
        true = c["language"]
        if true == "lug" and has_ft:
            hyp, conf = results["ft_lug"][c["id"]]
            dlang = "ft_lug"
            note = "ft_v2"
        else:
            hyp, conf = results[true][c["id"]]
            dlang = true
            note = "waxal_true"
        refs.append(c["ref"])
        hyps.append(hyp)
        langs.append(true)
        rows.append(
            {
                "id": c["id"],
                "true_lang": true,
                "method": name,
                "candidates": dlang,
                "decode_lang": dlang,
                "confidence": conf,
                "margin": None,
                "ref": c["ref"],
                "hyp": hyp,
                "route_match": int(true == "lug" or dlang == true),
                "note": note,
            }
        )
    return metrics_block(refs, hyps, langs, rows, name), rows


def conf_scale_diagnostic(clips, results) -> dict:
    """Report conf scale stats; do not conf-mix FT with waxal blindly."""
    out = {}
    for key in sorted(results.keys()):
        confs = [results[key][c["id"]][1] for c in clips if c["id"] in results[key]]
        if not confs:
            continue
        arr = np.asarray(confs, dtype=np.float64)
        out[key] = {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    # lug-only compare waxal vs ft
    lug_ids = [c["id"] for c in clips if c["language"] == "lug"]
    if "lug" in results and "ft_lug" in results and lug_ids:
        w = np.asarray([results["lug"][i][1] for i in lug_ids])
        f = np.asarray([results["ft_lug"][i][1] for i in lug_ids])
        out["lug_subset_compare"] = {
            "n": len(lug_ids),
            "waxal_lug_mean": float(w.mean()),
            "ft_lug_mean": float(f.mean()),
            "delta_mean_ft_minus_waxal": float(f.mean() - w.mean()),
            "note": "Do NOT conf-mix FT with waxal without scale alignment",
        }
    return out


# Deployable mapping knowledge from phase2_lid126 (portal LID codes → ASR cand sets)
DEPLOYABLE_LID_RULES = {
    "description": (
        "Map mms-lid-126 top-1 to candidate sets without true labels. "
        "Knowledge from outputs/phase2_lid126_full.csv hist + openset champion."
    ),
    "lid_to_cands": {
        "luo": ["ach", "lug", "sog"],  # portal: luo multi; ach is primary luo-proxy
        "lug": ["lug", "nyn", "sog"],
        "nyn": ["nyn", "lug"],  # if LID ever emits nyn
        "sog": ["sog", "lug"],
        "mas": ["mas", "lug"],
        "sna": ["sna", "lug"],
        "lin": ["lin", "lug"],
        "nso": ["sna", "lug"],
        "umb": ["sog", "lug"],
        "nya": ["sna", "lug"],
        "swh": ["lug", "lin"],
        "kin": ["nyn", "lug"],
        "default": ["lug"],  # conf sink but safest mass
    },
    "feature_to_deployable": {
        "oracle_lang": {
            "deployable": False,
            "reason": "Requires true language label",
            "proxy": "LID top-1 single-model if lid in WAXAL300 else FALLBACK single",
        },
        "hard_true": {
            "deployable": False,
            "reason": "Same as oracle_lang",
        },
        "multihyp_conf": {
            "deployable": True,
            "rule": (
                "LID→cands (luo→ach|lug|sog, lug→lug|nyn|sog, nyn→nyn|lug, "
                "sog→sog|lug, mas→mas|lug, else resolve+optional lug); pick max CTC conf"
            ),
            "note": "Closest to current openset champion for luo/lug; tighter than expand_5way",
        },
        "length_guard_multihyp": {
            "deployable": True,
            "rule": (
                "Same LID→cands as multihyp_conf; discard hyps with word_count "
                "<0.5*median or >2*median among candidates; then max conf"
            ),
        },
        "margin_multihyp_oracle_fb": {
            "deployable": False,
            "reason": "Oracle fallback uses true lang (ceiling only)",
        },
        "margin_multihyp_first_fb": {
            "deployable": True,
            "rule": (
                "LID→cands; if conf_best - conf_second >= 0.01 pick best; "
                "else decode with first candidate (primary LID-mapped model)"
            ),
        },
        "ft_lug_when_true_lug": {
            "deployable": "partial",
            "rule": (
                "When LID top-1 == lug (high p1), decode with checkpoints/mms-lug-ft-v2; "
                "else true-lang/LID waxal. Do NOT conf-mix FT conf with waxal conf."
            ),
            "note": "Already tested as ft_lug_safe/hard on portal; small proxy gain",
        },
    },
}


def write_md(
    methods: list[dict],
    out_md: Path,
    *,
    multihyp_base: dict | None,
    conf_diag: dict,
    notes: list[str],
    wall_s: float,
) -> None:
    ranked = sorted(methods, key=lambda m: -m["zindi_est"])
    lines = [
        "# Phase-3 routing / multi-hyp features (proxy n=200)",
        "",
        "Proxy: `data/proxy_val_index.csv` (validation only, seed 42, ach/nyn/lug/sog/mas).",
        "Family: **waxal-300m** same-family only (+ optional `checkpoints/mms-lug-ft-v2` for lug).",
        "CTC conf = mean frame logprob of argmax path (higher better).",
        "`zindi_est = 1 − 0.5·WER − 0.5·CER` (higher better).",
        "",
        "## Candidate map (by true_lang / oracle-LID)",
        "",
        "```",
        "lug → lug|nyn|sog",
        "ach → ach|lug|sog",
        "nyn → nyn|lug",
        "sog → sog|lug",
        "mas → mas|lug",
        "```",
        "",
        "## Methods",
        "",
        "| method | description |",
        "|--------|-------------|",
        "| oracle_lang | true-lang waxal only (ceiling) |",
        "| hard_true | alias of oracle_lang |",
        "| multihyp_conf | cand set by true_lang, max CTC conf |",
        "| length_guard_multihyp | drop word-count outliers vs median, then max conf |",
        "| margin_multihyp_oracle_fb | pick best if margin≥0.01 else true-lang hyp |",
        "| margin_multihyp_first_fb | pick best if margin≥0.01 else first candidate |",
        "| ft_lug_when_true_lug | true==lug → FT-v2 else true-lang waxal (no conf mix) |",
        "",
        "## Ranking by zindi_est (higher better)",
        "",
        "| rank | method | n | zindi_est↑ | WER | CER | error↓ | route_acc |",
        "|-----:|--------|--:|----------:|----:|----:|-------:|----------:|",
    ]
    for i, m in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {m['method']} | {m['n']} | {m['zindi_est']:.4f} | "
            f"{m['wer']:.4f} | {m['cer']:.4f} | {m['error']:.4f} | {m['route_acc']:.3f} |"
        )

    if multihyp_base is not None:
        lines += [
            "",
            "## Deltas vs multihyp_conf",
            "",
            f"Baseline multihyp_conf zindi_est = **{multihyp_base['zindi_est']:.4f}**",
            "",
            "| method | Δzindi_est | ≥ +0.005? |",
            "|--------|----------:|:---------:|",
        ]
        for m in ranked:
            if m["method"] == "multihyp_conf":
                continue
            d = m["zindi_est"] - multihyp_base["zindi_est"]
            flag = "YES" if d >= 0.005 else "no"
            lines.append(f"| {m['method']} | {d:+.4f} | {flag} |")

    lines += ["", "## Per-language zindi_est", ""]
    all_langs = sorted({lang for m in methods for lang in m["per_lang"]})
    header = "| lang | " + " | ".join(m["method"] for m in ranked) + " |"
    sep = "|------|" + "|".join(["------:"] * len(ranked)) + "|"
    lines += [header, sep]
    for lang in all_langs:
        cells = []
        for m in ranked:
            pl = m["per_lang"].get(lang)
            cells.append(f"{pl['zindi_est']:.3f}" if pl else "—")
        lines.append(f"| {lang} | " + " | ".join(cells) + " |")

    lines += ["", "## Decode mass", ""]
    for m in ranked:
        lines.append(f"- **{m['method']}**: `{m['decode_mass']}`")

    lines += [
        "",
        "## Confidence scale diagnostic (do not conf-mix FT↔waxal blindly)",
        "",
        "```json",
        json.dumps(conf_diag, indent=2),
        "```",
        "",
        "## Deployable Phase-2 rules (no true labels)",
        "",
        "Portal LID hist is dominated by **luo** (~785) and **lug** (~665) of 1500.",
        "Map LID top-1 → candidate set, then apply the feature:",
        "",
        "| LID top-1 | candidate set (deployable) |",
        "|-----------|----------------------------|",
        "| luo | ach\\|lug\\|sog |",
        "| lug | lug\\|nyn\\|sog |",
        "| nyn | nyn\\|lug |",
        "| sog | sog\\|lug |",
        "| mas | mas\\|lug |",
        "| sna / nso / nya | sna\\|lug (or sna only) |",
        "| other | resolve via FALLBACK or single lug |",
        "",
        "### Feature → deployable rule",
        "",
    ]
    for feat, info in DEPLOYABLE_LID_RULES["feature_to_deployable"].items():
        dep = info.get("deployable")
        lines.append(f"- **{feat}** (deployable={dep}): {info.get('rule') or info.get('reason')}")
        if info.get("note"):
            lines.append(f"  - note: {info['note']}")

    lines += ["", "## Notes", ""]
    for n in notes:
        lines.append(f"- {n}")
    lines += [
        f"- wall_s={wall_s:.1f}",
        "",
        "## Recommendation",
        "",
    ]
    # Build recommendation from ranking
    if multihyp_base is not None:
        beaters = [
            m
            for m in ranked
            if m["method"] != "multihyp_conf"
            and m["zindi_est"] - multihyp_base["zindi_est"] >= 0.005
            and m["method"]
            not in ("oracle_lang", "hard_true", "margin_multihyp_oracle_fb")
        ]
        ceil = [
            m
            for m in ranked
            if m["method"] in ("oracle_lang", "hard_true", "margin_multihyp_oracle_fb")
            and m["zindi_est"] - multihyp_base["zindi_est"] >= 0.005
        ]
        if beaters:
            best = beaters[0]
            lines.append(
                f"**Deploy candidate:** `{best['method']}` "
                f"(zindi_est={best['zindi_est']:.4f}, "
                f"Δ vs multihyp={best['zindi_est'] - multihyp_base['zindi_est']:+.4f})."
            )
        else:
            lines.append(
                "**No deployable feature ≥ +0.005 over multihyp_conf on full proxy.**"
            )
            if ceil:
                lines.append(
                    "Ceiling features that beat multihyp (need better LID / margin): "
                    + ", ".join(
                        f"{m['method']} (Δ{m['zindi_est'] - multihyp_base['zindi_est']:+.4f})"
                        for m in ceil
                    )
                    + "."
                )
            lines.append(
                "Default Phase-2 remains openset champion (LID→cands max conf) unless "
                "margin_first or length_guard show non-negative + intentional hyp-diff."
            )
        lines.append("")
        lines.append(
            "Upload rule: only if deployable method strictly beats champion proxy "
            "and public risk is accepted; never overwrite openset champion without that."
        )
    out_md.write_text("\n".join(lines))
    logger.info("wrote %s", out_md)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=PROXY_INDEX)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--skip-ft", action="store_true", help="skip FT-lug decode")
    p.add_argument("--margin", type=float, default=MARGIN_THRESH)
    p.add_argument("--out-json", type=Path, default=OUTPUT_DIR / "phase3_routing.json")
    p.add_argument("--out-md", type=Path, default=OUTPUT_DIR / "phase3_routing.md")
    p.add_argument(
        "--out-csv", type=Path, default=OUTPUT_DIR / "phase3_routing_detail.csv"
    )
    p.add_argument(
        "--cache-hyps",
        type=Path,
        default=OUTPUT_DIR / "phase3_routing_hyps_cache.json",
        help="cache all model×clip hyps/conf for reuse",
    )
    p.add_argument(
        "--reuse-cache",
        action="store_true",
        help="reuse --cache-hyps if present instead of decoding",
    )
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    logger.info("device=%s seed=%d", device, args.seed)
    t0 = time.time()

    clips = load_proxy_clips(args.index)
    if args.max_clips:
        clips = clips[: args.max_clips]
    logger.info(
        "clips loaded: %d mass=%s",
        len(clips),
        dict(Counter(c["language"] for c in clips)),
    )

    # models: all langs appearing as true or as multihyp cands
    need: set[str] = set()
    for c in clips:
        need.add(c["language"])
        need.update(MULTI_CANDS.get(c["language"], []))
    model_langs = [m for m in sorted(need) if m in WAXAL300]
    logger.info("waxal models: %s", model_langs)

    results: dict[str, dict[str, tuple[str, float]]]
    if args.reuse_cache and args.cache_hyps.exists():
        logger.info("reusing cache %s", args.cache_hyps)
        raw = json.loads(args.cache_hyps.read_text())
        results = {
            k: {cid: (v[0], float(v[1])) for cid, v in bucket.items()}
            for k, bucket in raw.items()
        }
    else:
        results = score_models(
            clips,
            model_langs,
            device,
            include_ft_lug=not args.skip_ft,
        )
        # persist cache (hyps only, no audio)
        serial = {
            k: {cid: [hyp, conf] for cid, (hyp, conf) in bucket.items()}
            for k, bucket in results.items()
        }
        args.cache_hyps.parent.mkdir(parents=True, exist_ok=True)
        args.cache_hyps.write_text(json.dumps(serial))
        logger.info("wrote hyp cache %s", args.cache_hyps)

    conf_diag = conf_scale_diagnostic(clips, results)

    method_metrics: list[dict] = []
    all_detail: list[dict] = []

    # 1 + 3 oracle / hard_true
    met, det = evaluate_oracle(clips, results)
    method_metrics.append(met)
    all_detail.extend(det)
    # hard_true = same metrics, duplicate rows with renamed method
    hard = dict(met)
    hard["method"] = "hard_true"
    method_metrics.append(hard)
    for r in det:
        rr = dict(r)
        rr["method"] = "hard_true"
        all_detail.append(rr)

    # 2 multihyp
    met, det = evaluate_multihyp(clips, results, "multihyp_conf")
    method_metrics.append(met)
    all_detail.extend(det)
    multihyp_base = met

    # 4 length guard
    met, det = evaluate_length_guard(clips, results)
    method_metrics.append(met)
    all_detail.extend(det)

    # 5 margin (oracle + first fallback)
    met, det = evaluate_margin(
        clips,
        results,
        oracle_fallback=True,
        name="margin_multihyp_oracle_fb",
        thresh=args.margin,
    )
    method_metrics.append(met)
    all_detail.extend(det)
    met, det = evaluate_margin(
        clips,
        results,
        oracle_fallback=False,
        name="margin_multihyp_first_fb",
        thresh=args.margin,
    )
    method_metrics.append(met)
    all_detail.extend(det)

    # 6 ft lug
    if "ft_lug" in results:
        met, det = evaluate_ft_lug_when_true(clips, results)
        method_metrics.append(met)
        all_detail.extend(det)
    else:
        logger.warning("ft_lug missing; skip ft_lug_when_true_lug")

    # ranking + deltas
    ranked = sorted(method_metrics, key=lambda m: -m["zindi_est"])
    deltas = {}
    for m in method_metrics:
        if m["method"] == "multihyp_conf":
            continue
        deltas[m["method"]] = {
            "delta_zindi_est": m["zindi_est"] - multihyp_base["zindi_est"],
            "delta_error": multihyp_base["error"] - m["error"],
            "beats_multihyp_by_0p005": (m["zindi_est"] - multihyp_base["zindi_est"])
            >= 0.005,
        }

    notes = [
        "Waxal-300m same-family multi-hyp; FT-lug is separate checkpoint (no conf-mix in pickers).",
        f"margin threshold={args.margin}",
        f"length guard: word_count in [{LENGTH_LO}, {LENGTH_HI}] × median among cands",
        f"models: {list(results.keys())}",
        f"clip mass: {dict(Counter(c['language'] for c in clips))}",
        "No test gold used; proxy is validation-only.",
        "Does not overwrite submission_phase2_openset / champion.",
    ]
    # summary of margin fallback rates
    for mname in ("margin_multihyp_oracle_fb", "margin_multihyp_first_fb"):
        rows = [r for r in all_detail if r["method"] == mname]
        if not rows:
            continue
        notes_c = Counter(r["note"] for r in rows)
        notes.append(f"{mname} note counts: {dict(notes_c)}")

    wall_s = time.time() - t0
    payload = {
        "seed": args.seed,
        "n_clips": len(clips),
        "langs": list(PROXY_LANGS),
        "index": str(args.index),
        "device": str(device),
        "models": list(results.keys()),
        "multi_cands": MULTI_CANDS,
        "margin_thresh": args.margin,
        "length_guard": {"lo": LENGTH_LO, "hi": LENGTH_HI},
        "methods": method_metrics,
        "ranking": [
            {"rank": i + 1, "method": m["method"], "zindi_est": m["zindi_est"]}
            for i, m in enumerate(ranked)
        ],
        "deltas_vs_multihyp_conf": deltas,
        "conf_scale_diagnostic": conf_diag,
        "deployable_rules": DEPLOYABLE_LID_RULES,
        "notes": notes,
        "wall_s": wall_s,
        "target": "find method >= +0.005 zindi_est vs multihyp_conf on n=200",
        "target_met_deployable": any(
            deltas[k]["beats_multihyp_by_0p005"]
            and k
            not in (
                "oracle_lang",
                "hard_true",
                "margin_multihyp_oracle_fb",
            )
            for k in deltas
        ),
        "target_met_any_including_oracle": any(
            deltas[k]["beats_multihyp_by_0p005"] for k in deltas
        ),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", args.out_json)

    pd.DataFrame(all_detail).to_csv(args.out_csv, index=False)
    logger.info("wrote %s", args.out_csv)

    write_md(
        method_metrics,
        args.out_md,
        multihyp_base=multihyp_base,
        conf_diag=conf_diag,
        notes=notes,
        wall_s=wall_s,
    )

    print(
        json.dumps(
            {
                "ranking": payload["ranking"],
                "deltas_vs_multihyp_conf": {
                    k: {
                        "delta_zindi_est": round(v["delta_zindi_est"], 4),
                        "beats": v["beats_multihyp_by_0p005"],
                    }
                    for k, v in deltas.items()
                },
                "target_met_deployable": payload["target_met_deployable"],
                "wall_s": round(wall_s, 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
