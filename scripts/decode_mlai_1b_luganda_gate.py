#!/usr/bin/env python3
"""Decode and score the prepared locked Luganda gate with OmniASR 1B."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from jiwer import cer, wer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/lug_gate"
TABLE = ROOT / "outputs/goal_2026_08_08/luganda_fusion/matched_hypotheses.csv"


def normalize(text: object) -> str:
    import re
    import unicodedata

    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"[^\w\s']+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def metric(refs: list[str], hyps: list[str]) -> dict[str, float | int]:
    refs = [normalize(x) for x in refs]
    hyps = [normalize(x) or "." for x in hyps]
    w, c = float(wer(refs, hyps)), float(cer(refs, hyps))
    return {"n": len(refs), "wer": w, "cer": c, "zindi": 1.0 - 0.5 * (w + c)}


def load_pipeline():
    loader_path = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/omni_cpu_screen.py"
    spec = importlib.util.spec_from_file_location("mlai_omni_cpu_screen", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_pipeline()


def bootstrap(frame: pd.DataFrame, *, draws: int = 2000) -> dict[str, float | int]:
    rng = np.random.default_rng(20260810)
    speakers = np.array(sorted(frame["speaker_id"].astype(str).unique()))
    groups = {
        speaker: np.flatnonzero(frame["speaker_id"].astype(str).to_numpy() == speaker)
        for speaker in speakers
    }
    deltas = np.empty(draws, dtype=np.float64)
    refs = frame["reference"].tolist()
    candidate = frame["candidate"].tolist()
    baseline = frame["baseline"].tolist()
    for i in range(draws):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        idx = np.concatenate([groups[speaker] for speaker in sampled])
        r = [refs[j] for j in idx]
        deltas[i] = metric(r, [candidate[j] for j in idx])["zindi"] - metric(
            r, [baseline[j] for j in idx]
        )["zindi"]
    return {
        "draws": draws,
        "seed": 20260810,
        "delta_mean": float(deltas.mean()),
        "delta_p05": float(np.quantile(deltas, 0.05)),
        "delta_p50": float(np.quantile(deltas, 0.50)),
        "delta_p95": float(np.quantile(deltas, 0.95)),
        "probability_delta_positive": float(np.mean(deltas > 0)),
    }


def main() -> None:
    manifest = pd.read_csv(OUT / "manifest.csv", dtype=str).fillna("")
    table = pd.read_csv(TABLE, dtype=str).fillna("")
    baseline_by_id = table.set_index("ID")["mms_ft_v3_domain_beam_splitjoin"].to_dict()
    manifest["baseline"] = [baseline_by_id[uid] for uid in manifest["ID"]]
    pipe = load_pipeline()
    durations = []
    for path in manifest["audio"]:
        with wave.open(path) as handle:
            durations.append(handle.getnframes() / handle.getframerate())
    # OmniASR's public pipeline hard-caps single examples at 40 s. Keep the
    # incumbent on longer validation rows as an explicit duration guard; this
    # is the same failure-safe policy used by the existing specialist routes.
    short_idx = [i for i, seconds in enumerate(durations) if seconds <= 40.0]
    short_paths = [manifest.iloc[i]["audio"] for i in short_idx]
    short_raw = pipe.transcribe(
        short_paths,
        lang=["lug_Latn"] * len(short_paths),
        batch_size=1,
    )
    raw = list(manifest["baseline"])
    for i, value in zip(short_idx, short_raw):
        raw[i] = str(value)
    manifest["duration_sec"] = durations
    manifest["candidate_source"] = ["mlai_1b" if i in short_idx else "incumbent_duration_guard" for i in range(len(manifest))]
    manifest["raw_candidate"] = [str(x) for x in raw]
    manifest["candidate"] = [normalize(x) or "." for x in raw]
    manifest.to_csv(OUT / "hypotheses.csv", index=False)
    refs = manifest["reference"].tolist()
    candidate = manifest["candidate"].tolist()
    baseline = [normalize(x) for x in manifest["baseline"]]
    report = {
        "protocol": {
            "dataset": "google/WaxalNLP Luganda validation",
            "rows": len(manifest),
            "locked_source": str(TABLE),
            "speaker_disjoint_split": "locked order first 20 tune, last 20 holdout",
            "test_labels_used": False,
            "phase2_labels_or_audio_used": False,
            "max_single_audio_seconds": 40.0,
            "candidate_short_rows": len(short_idx),
            "candidate_duration_guard_rows": len(manifest) - len(short_idx),
        },
        "metrics": {
            "baseline": metric(refs, baseline),
            "candidate": metric(refs, candidate),
            "baseline_tune": metric(refs[:20], baseline[:20]),
            "candidate_tune": metric(refs[:20], candidate[:20]),
            "baseline_holdout": metric(refs[20:], baseline[20:]),
            "candidate_holdout": metric(refs[20:], candidate[20:]),
        },
        "bootstrap_speakers": bootstrap(manifest),
        "model": {
            "repo": "mlai-dante/waxal-omniASR-LLM-1B-v2",
            "checkpoint": "step_1000/model",
            "device": "cpu",
            "torch_device": "torch.device('cpu')",
        },
        "cache": str(OUT / "route_cache_mlai_1b_lug_duration_guard_raw.csv"),
        "id_sha256": hashlib.sha256("\n".join(manifest["ID"]).encode()).hexdigest(),
    }
    m = report["metrics"]
    report["delta"] = {
        "all": m["candidate"]["zindi"] - m["baseline"]["zindi"],
        "tune": m["candidate_tune"]["zindi"] - m["baseline_tune"]["zindi"],
        "holdout": m["candidate_holdout"]["zindi"] - m["baseline_holdout"]["zindi"],
    }
    report["strong_pass"] = bool(
        report["delta"]["all"] >= 0.01
        and report["delta"]["tune"] > 0
        and report["delta"]["holdout"] > 0
        and report["bootstrap_speakers"]["delta_p05"] > 0
        and report["bootstrap_speakers"]["probability_delta_positive"] >= 0.95
        and m["candidate"]["wer"] <= m["baseline"]["wer"]
        and m["candidate"]["cer"] <= m["baseline"]["cer"]
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
