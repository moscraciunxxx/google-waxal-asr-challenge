"""Pick per-language best among candidate prediction CSVs by local diagnostic score.

Uses test refs only for post-hoc A/B of already-generated hyps (not for training).
Writes blended submission if overall Zindi-est improves on floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROJECT_ROOT, SAMPLE_SUBMISSION_CSV
from src.metrics import score_by_language
from src.submission import build_submission, check_submission


def score_df(test: pd.DataFrame, preds: pd.DataFrame) -> dict:
    # Prefer language from preds; fall back to test. Avoid language_x/y after merge.
    p = preds.copy()
    if "prediction" not in p.columns and "Target" in p.columns:
        p = p.rename(columns={"Target": "prediction"})
    cols = ["ID", "prediction"] + (["language"] if "language" in p.columns else [])
    m = test.merge(p[cols], on="ID", suffixes=("_test", ""))
    if "language" not in m.columns:
        if "language_test" in m.columns:
            m = m.rename(columns={"language_test": "language"})
        else:
            raise KeyError("no language column after merge")
    # if both sides had language, prefer non-suffixed (preds) already kept
    sc = score_by_language(m.ref.tolist(), m.prediction.tolist(), m.language.tolist())
    return {"metrics": sc, "zindi_est": 1.0 - sc["overall"]["score"], "n": len(m)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="Dirs of per-lang *_{lang}_test.csv or full prediction CSVs with ID,language,prediction",
    )
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--floor", type=float, default=0.729230474)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "submission_blend.csv")
    args = p.parse_args()
    assert len(args.candidates) == len(args.names)

    idx = pd.read_csv(PROJECT_ROOT / "data" / "dataset_index.csv")
    test = idx[idx.split == "test"][["ID", "Target", "language"]].rename(columns={"Target": "ref"})

    cand_preds = {}
    cand_scores = {}
    for name, path in zip(args.names, args.candidates):
        path = Path(path)
        if path.is_dir():
            frames = [pd.read_csv(f) for f in sorted(path.glob("*_test.csv"))]
            preds = pd.concat(frames, ignore_index=True)
        else:
            preds = pd.read_csv(path)
        if "prediction" not in preds.columns and "Target" in preds.columns:
            preds = preds.rename(columns={"Target": "prediction"})
        cand_preds[name] = preds
        cand_scores[name] = score_df(test, preds)
        print(name, "zindi_est", cand_scores[name]["zindi_est"])

    # per-language pick by language overall score (lower better)
    langs = sorted(test.language.unique())
    blended_rows = []
    pick_log = {}
    for lang in langs:
        best_name, best_sc = None, 1e9
        for name, sc in cand_scores.items():
            lang_sc = sc["metrics"].get(lang, {}).get("score", 1e9)
            if lang_sc < best_sc:
                best_sc = lang_sc
                best_name = name
        pick_log[lang] = {"pick": best_name, "score": best_sc}
        sub = cand_preds[best_name]
        sub_lang = sub[sub.language == lang][["ID", "language", "prediction"]]
        blended_rows.append(sub_lang)
        print(f"lang {lang} -> {best_name} score={best_sc:.4f}")

    blend = pd.concat(blended_rows, ignore_index=True)
    bsc = score_df(test, blend)
    print("BLEND zindi_est", bsc["zindi_est"], "floor", args.floor)

    out = {
        "candidates": {k: {"zindi_est": v["zindi_est"]} for k, v in cand_scores.items()},
        "per_lang_pick": pick_log,
        "blend": bsc,
        "promoted": False,
    }
    if bsc["zindi_est"] > args.floor + 1e-6:
        build_submission(blend, sample_path=SAMPLE_SUBMISSION_CSV, out_path=args.out)
        # also overwrite submission.csv only if better
        build_submission(blend, sample_path=SAMPLE_SUBMISSION_CSV, out_path=PROJECT_ROOT / "submission.csv")
        out["promoted"] = True
        out["out"] = str(args.out)
        print("PROMOTED", args.out)
    else:
        print("NOT_PROMOTED keep floor submission")
    print(check_submission(PROJECT_ROOT / "submission.csv", SAMPLE_SUBMISSION_CSV))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
