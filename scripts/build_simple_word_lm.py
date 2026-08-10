"""Export train texts already in data/lms; optionally build ARPA with lmplz if present.

Also writes a pure-Python unigram+bigram JSON usable without KenLM.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text_norm import normalize_text


def build_counts(txt_path: Path) -> dict:
    uni: Counter[str] = Counter()
    bi: Counter[tuple[str, str]] = Counter()
    n_sents = 0
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        words = normalize_text(line).split()
        if not words:
            continue
        n_sents += 1
        uni["<s>"] += 1
        uni["</s>"] += 1
        prev = "<s>"
        for w in words:
            uni[w] += 1
            bi[(prev, w)] += 1
            prev = w
        bi[(prev, "</s>")] += 1
    return {"uni": dict(uni), "bi": {f"{a}\t{b}": c for (a, b), c in bi.items()}, "n_sents": n_sents}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=["lin", "lug", "sna"])
    p.add_argument("--lm-dir", type=Path, default=ROOT / "data" / "lms")
    p.add_argument("--order", type=int, default=3)
    args = p.parse_args()
    args.lm_dir.mkdir(parents=True, exist_ok=True)
    lmplz = None
    for cand in ("lmplz", "/opt/homebrew/bin/lmplz", "/usr/local/bin/lmplz"):
        try:
            subprocess.check_call([cand, "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            lmplz = cand
            break
        except Exception:
            continue
    for lang in args.langs:
        txt = args.lm_dir / f"{lang}_train.txt"
        if not txt.exists():
            print("missing", txt)
            continue
        counts = build_counts(txt)
        out_json = args.lm_dir / f"{lang}_counts.json"
        out_json.write_text(json.dumps(counts))
        print(lang, "types", len(counts["uni"]), "bigrams", len(counts["bi"]), "->", out_json)
        if lmplz:
            arpa = args.lm_dir / f"{lang}_{args.order}gram.arpa"
            cmd = [
                lmplz,
                "-o",
                str(args.order),
                "--discount_fallback",
                "-S",
                "30%",
                "--text",
                str(txt),
                "--arpa",
                str(arpa),
            ]
            print("running", " ".join(cmd))
            subprocess.check_call(cmd)
            print("wrote", arpa)
        else:
            print("lmplz not found; pure-python counts only")


if __name__ == "__main__":
    main()
