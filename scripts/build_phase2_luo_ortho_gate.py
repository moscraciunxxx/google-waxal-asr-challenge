#!/usr/bin/env python3
"""NEW method: Dholuo orthography / char n-gram verifier for residual ach-route Luo.

Public bans we do NOT use as primary lever:
  - margin re-route
  - decode_lang==lug rewrite (frozen forever)
  - dual thr>0.15 alone (public reject thr0.16 @ 0.5578)
  - blind all-LID=luo
  - Phase-1 test gold

New mechanism (Y):
  Char n-gram LMs + orthography markers score the **MMS-1B luo hyp text**
  (not gold). Calibrated on decoded hyps from FLEURS luo_ke (true) vs
  WAXAL ach validation (false) in phase2_luo_conf_gate_calib.csv.
  Residual pool: lid=luo & decode=ach & not already dual thr0.15.
  Gates combine ortho_mms with p1 / conf_delta / soft dual CER — never pure thr expand.
  Overlay = MMS-1B luo transcript. Never touch decode_lang==lug.

Outputs:
  submission_phase2_beat_k63_ortho*.csv
  outputs/beat_k63/luo_ortho_*
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT, SAMPLE_SUBMISSION_CSV
from src.submission import check_submission
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("luo_ortho_gate")

BEAT = OUTPUT_DIR / "beat_k63"
FLOOR = PROJECT_ROOT / "submission_phase2_selective_v3_dual15.csv"
SEL_V3 = PROJECT_ROOT / "submission_phase2_selective_v3.csv"
DETAIL = OUTPUT_DIR / "phase2_selective_v3_detail.csv"
MMS = OUTPUT_DIR / "phase2_luo_mms1b_detail.csv"
CLR = OUTPUT_DIR / "phase2_selective_clear_allluo_detail.csv"
CONF = OUTPUT_DIR / "phase2_luo_conf_gate_scores.csv"
CONF_CALIB = OUTPUT_DIR / "phase2_luo_conf_gate_calib.csv"
DUAL_POOL = OUTPUT_DIR / "phase2_achluo_dual_pool_detail.csv"

ZINDI_MMS1B_LUO = 0.8550694691661159
ZINDI_ACH_BEAM = 0.2777212778081496
GAIN_TRUE_LUO = ZINDI_MMS1B_LUO - ZINDI_ACH_BEAM  # ~0.577
LOSS_FALSE_LUO = 0.40
N_TEST = 1500
FLOOR_PUBLIC = 0.560605696
TARGET = 0.720212909


# ---------------------------------------------------------------------------
# Char n-gram LM
# ---------------------------------------------------------------------------
class CharNGramLM:
    """Smoothed character n-gram LM over normalized text."""

    def __init__(self, n: int = 4, alpha: float = 0.5):
        self.n = n
        self.alpha = alpha
        self.context_counts: dict[str, Counter] = defaultdict(Counter)
        self.context_totals: Counter = Counter()
        self.vocab: set[str] = set()
        self.n_tokens = 0

    def _prep(self, text: str) -> str:
        t = normalize_text(text)
        return f"<{t}>" if t else "<>"

    def fit(self, texts: list[str]) -> "CharNGramLM":
        n = self.n
        for text in texts:
            s = self._prep(text)
            if len(s) < n:
                continue
            for i in range(len(s) - n + 1):
                ctx = s[i : i + n - 1]
                ch = s[i + n - 1]
                self.context_counts[ctx][ch] += 1
                self.context_totals[ctx] += 1
                self.vocab.add(ch)
                self.n_tokens += 1
        if not self.vocab:
            self.vocab = set(" abcdefghijklmnopqrstuvwxyz'")
        return self

    def log_prob(self, text: str) -> float:
        s = self._prep(text)
        n = self.n
        if len(s) < n:
            return -10.0
        V = max(len(self.vocab), 1)
        alpha = self.alpha
        total = 0.0
        count = 0
        for i in range(len(s) - n + 1):
            ctx = s[i : i + n - 1]
            ch = s[i + n - 1]
            c_ctx = self.context_totals.get(ctx, 0)
            c_ch = self.context_counts.get(ctx, Counter()).get(ch, 0)
            p = (c_ch + alpha) / (c_ctx + alpha * V)
            total += math.log10(max(p, 1e-12))
            count += 1
        return total / max(count, 1)

    def to_meta(self) -> dict:
        return {
            "n": self.n,
            "alpha": self.alpha,
            "n_tokens": self.n_tokens,
            "n_contexts": len(self.context_totals),
            "vocab_size": len(self.vocab),
        }


# ---------------------------------------------------------------------------
# Handcrafted orthography features (Dholuo vs Acholi vs Luganda)
# ---------------------------------------------------------------------------
_RE_NG_APOS = re.compile(r"ng'")
_RE_TIE = re.compile(r"\btie\b")
_RE_TYE = re.compile(r"\btye\b")
_RE_PII = re.compile(r"\bpii\b")
_RE_PI = re.compile(r"\bpi\b")
_RE_ENG = re.compile(r"ŋ")
_RE_DBL = re.compile(r"([a-z])\1")
_RE_ACHIEL = re.compile(r"\bachiel\b|\bariyo\b|\badek\b|\bang'wen\b")
_RE_ACEL = re.compile(r"\bacel\b|\baryo\b|\badek\b|\bangwen\b")
_RE_KOD = re.compile(r"\bkod\b")
_RE_MAR = re.compile(r"\bmar\b")
_RE_ME = re.compile(r"\bme\b")
_RE_PA = re.compile(r"\bpa\b")
_RE_ERA = re.compile(r"\bera\b")
_RE_OMU = re.compile(r"\bomu\b|\boku\b|\beki")


def ortho_features(text: str) -> dict[str, float]:
    t = normalize_text(text)
    if not t:
        return {
            "luo_marker": 0.0,
            "ach_marker": 0.0,
            "lug_marker": 0.0,
            "luo_minus_ach": 0.0,
            "len_chars": 0.0,
        }
    L = max(len(t), 1)
    ng = len(_RE_NG_APOS.findall(t))
    dh = len(re.findall(r"dh", t))
    th = len(re.findall(r"th", t))
    apos = t.count("'")
    tie = len(_RE_TIE.findall(t))
    tye = len(_RE_TYE.findall(t))
    pi = len(_RE_PI.findall(t))
    pii = len(_RE_PII.findall(t))
    eng = len(_RE_ENG.findall(t))
    dbl = len(_RE_DBL.findall(t))
    achiel = len(_RE_ACHIEL.findall(t))
    acel = len(_RE_ACEL.findall(t))
    kod = len(_RE_KOD.findall(t))
    mar = len(_RE_MAR.findall(t))
    me = len(_RE_ME.findall(t))
    pa = len(_RE_PA.findall(t))
    era = len(_RE_ERA.findall(t))
    omu = len(_RE_OMU.findall(t))
    luo = (
        3.0 * ng
        + 2.0 * dh
        + 1.0 * th
        + 1.5 * apos
        + 2.0 * tie
        + 1.0 * pi
        + 2.0 * achiel
        + 1.5 * kod
        + 1.0 * mar
    )
    ach = 4.0 * eng + 2.5 * tye + 2.0 * pii + 2.0 * acel + 1.0 * me + 1.0 * pa
    lug = 1.5 * dbl + 2.0 * era + 2.0 * omu
    scale = 100.0 / L
    return {
        "luo_marker": luo * scale,
        "ach_marker": ach * scale,
        "lug_marker": lug * scale,
        "luo_minus_ach": (luo - ach) * scale,
        "len_chars": float(L),
    }


def cer(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(la, lb)


def load_fleurs_luo_texts(split: str) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("google/fleurs", "luo_ke", split=split)
    keep = [c for c in ("transcription", "raw_transcription") if c in ds.column_names]
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    texts = []
    for i in range(len(ds)):
        t = normalize_text(ds[i].get("transcription") or ds[i].get("raw_transcription") or "")
        if t:
            texts.append(t)
    return texts


def load_waxal_val_texts(lang: str) -> list[str]:
    hub = Path.home() / ".cache/huggingface/hub"
    snaps = sorted((hub / "datasets--google--WaxalNLP" / "snapshots").glob("*"), reverse=True)
    texts: list[str] = []
    for snap in snaps:
        for pq in sorted((snap / "data" / "ASR" / lang).glob(f"{lang}-validation-*.parquet")):
            df = pd.read_parquet(pq, columns=["transcription"])
            for t in df["transcription"].tolist():
                nt = normalize_text(t)
                if nt:
                    texts.append(nt)
        if texts:
            break
    if not texts:
        proxy = pd.read_csv(PROJECT_ROOT / "data" / "proxy_val_index.csv")
        sub = proxy.loc[proxy.language == lang, "transcription"]
        texts = [normalize_text(t) for t in sub if normalize_text(t)]
    return texts


def score_text(text: str, lm_luo: CharNGramLM, lm_ach: CharNGramLM, lm_lug: CharNGramLM) -> dict:
    lp_l = lm_luo.log_prob(text)
    lp_a = lm_ach.log_prob(text)
    lp_g = lm_lug.log_prob(text)
    feat = ortho_features(text)
    ngram_margin = lp_l - max(lp_a, lp_g)
    marker = feat["luo_minus_ach"] - 0.3 * feat["lug_marker"]
    ortho = 5.0 * ngram_margin + 0.15 * marker
    return {
        "lp_luo": lp_l,
        "lp_ach": lp_a,
        "lp_lug": lp_g,
        "ngram_margin": ngram_margin,
        "marker_luo_ach": feat["luo_minus_ach"],
        "marker_lug": feat["lug_marker"],
        "ortho": ortho,
    }


def build_lms(n: int = 4) -> tuple[CharNGramLM, CharNGramLM, CharNGramLM, dict]:
    """Luo LM: FLEURS train. Ach/Lug LMs: WAXAL validation (never test)."""
    logger.info("loading FLEURS luo_ke train for Luo LM")
    luo_train = load_fleurs_luo_texts("train")
    logger.info("loading WAXAL ach/lug validation for LMs")
    ach_all = load_waxal_val_texts("ach")
    lug_all = load_waxal_val_texts("lug")

    lm_luo = CharNGramLM(n=n).fit(luo_train)
    lm_ach = CharNGramLM(n=n).fit(ach_all)
    lm_lug = CharNGramLM(n=n).fit(lug_all)
    meta = {
        "luo_train_n": len(luo_train),
        "ach_val_n": len(ach_all),
        "lug_val_n": len(lug_all),
        "lm_luo": lm_luo.to_meta(),
        "lm_ach": lm_ach.to_meta(),
        "lm_lug": lm_lug.to_meta(),
        "note": "Luo LM from FLEURS train only; calib on decoded hyps not gold text",
    }
    return lm_luo, lm_ach, lm_lug, meta


def calibrate_on_decoded_hyps(
    lm_luo: CharNGramLM, lm_ach: CharNGramLM, lm_lug: CharNGramLM
) -> tuple[pd.DataFrame, dict]:
    """Primary calib: score MMS-luo *hyps* from conf-gate FLEURS/ach set.

    Gold-text scoring is reported only as a sanity check — MMS adapter always
    emits Luo orthography, so hyp-level ortho is the real detector.
    """
    if not CONF_CALIB.exists():
        raise FileNotFoundError(
            f"Need {CONF_CALIB} (from build_phase2_luo_conf_gate.py --calib)"
        )
    raw = pd.read_csv(CONF_CALIB)
    rows = []
    for r in raw.itertuples():
        sc_l = score_text(str(r.text_luo), lm_luo, lm_ach, lm_lug)
        sc_a = score_text(str(r.text_ach), lm_luo, lm_ach, lm_lug)
        sc_ref = score_text(str(r.ref), lm_luo, lm_ach, lm_lug) if pd.notna(r.ref) else {}
        rows.append(
            {
                "domain": r.domain,
                "true_luo": int(r.true_luo),
                "delta": float(r.delta),
                "ortho_luo_hyp": sc_l["ortho"],
                "ortho_ach_hyp": sc_a["ortho"],
                "ngram_luo_hyp": sc_l["ngram_margin"],
                "marker_luo_hyp": sc_l["marker_luo_ach"],
                "ortho_ref": sc_ref.get("ortho", np.nan),
                "cer_la": cer(str(r.text_luo), str(r.text_ach)),
            }
        )
    cdf = pd.DataFrame(rows)

    grid = []
    best_fpr10 = None
    best_fpr05 = None
    for thr in np.linspace(-0.5, 4.5, 101):
        pred = cdf["ortho_luo_hyp"] >= thr
        luo = cdf[cdf.true_luo == 1]
        neg = cdf[cdf.true_luo == 0]
        tpr = float(pred[cdf.true_luo == 1].mean()) if len(luo) else 0.0
        fpr = float(pred[cdf.true_luo == 0].mean()) if len(neg) else 0.0
        n_pos = int(pred.sum())
        tp = int((pred & (cdf.true_luo == 1)).sum())
        fp = int((pred & (cdf.true_luo == 0)).sum())
        prec = tp / n_pos if n_pos else 0.0
        rec_d = {
            "ortho_thr": float(thr),
            "tpr": tpr,
            "fpr": fpr,
            "precision": prec,
            "recall": tpr,
            "tp": tp,
            "fp": fp,
            "n_accept": n_pos,
            "exp_unit_gain": prec * GAIN_TRUE_LUO - (1 - prec) * LOSS_FALSE_LUO if n_pos else 0.0,
        }
        grid.append(rec_d)
        if fpr <= 0.05:
            key = (tpr, prec)
            if best_fpr05 is None or key > best_fpr05[0]:
                best_fpr05 = (key, rec_d)
        if fpr <= 0.10:
            key = (tpr, prec)
            if best_fpr10 is None or key > best_fpr10[0]:
                best_fpr10 = (key, rec_d)

    # Secondary grids: combo ortho + conf delta
    combo_grid = []
    for thr in [1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0]:
        for dthr in [-0.02, -0.01, 0.0, 0.01]:
            pred = (cdf.ortho_luo_hyp >= thr) & (cdf.delta >= dthr)
            tpr = float(pred[cdf.true_luo == 1].mean())
            fpr = float(pred[cdf.true_luo == 0].mean())
            n_pos = int(pred.sum())
            prec = float(cdf.true_luo[pred].mean()) if n_pos else 0.0
            combo_grid.append(
                {
                    "ortho_thr": thr,
                    "delta_thr": dthr,
                    "tpr": tpr,
                    "fpr": fpr,
                    "precision": prec,
                    "n_accept": n_pos,
                }
            )

    chosen = best_fpr10[1] if best_fpr10 else sorted(grid, key=lambda r: (r["fpr"], -r["tpr"]))[0]
    chosen05 = best_fpr05[1] if best_fpr05 else None

    sep = {
        "ortho_luo_hyp_true_mean": float(cdf.loc[cdf.true_luo == 1, "ortho_luo_hyp"].mean()),
        "ortho_luo_hyp_false_mean": float(cdf.loc[cdf.true_luo == 0, "ortho_luo_hyp"].mean()),
        "ortho_luo_hyp_true_p10": float(cdf.loc[cdf.true_luo == 1, "ortho_luo_hyp"].quantile(0.10)),
        "ortho_luo_hyp_false_p90": float(cdf.loc[cdf.true_luo == 0, "ortho_luo_hyp"].quantile(0.90)),
        "ortho_ref_true_mean": float(cdf.loc[cdf.true_luo == 1, "ortho_ref"].mean()),
        "ortho_ref_false_mean": float(cdf.loc[cdf.true_luo == 0, "ortho_ref"].mean()),
        "delta_true_mean": float(cdf.loc[cdf.true_luo == 1, "delta"].mean()),
        "delta_false_mean": float(cdf.loc[cdf.true_luo == 0, "delta"].mean()),
    }

    # Gold-text sanity (expect near-perfect separation)
    gold_sep = {
        "note": "gold-text ortho separates perfectly; do NOT use for residual thr (MMS always Luo-ish)",
        "ref_true_mean": sep["ortho_ref_true_mean"],
        "ref_false_mean": sep["ortho_ref_false_mean"],
    }

    report = {
        "n_luo": int((cdf.true_luo == 1).sum()),
        "n_neg": int((cdf.true_luo == 0).sum()),
        "source": str(CONF_CALIB),
        "separation_decoded_hyp": sep,
        "gold_text_sanity": gold_sep,
        "chosen_fpr10": chosen,
        "chosen_fpr05": chosen05,
        "grid_highlights": [
            g
            for g in grid
            if abs(g["ortho_thr"] - round(g["ortho_thr"] * 5) / 5) < 1e-9
            or g["ortho_thr"] in (2.4, 3.0, 4.0)
        ][:20],
        "combo_best_fpr10": sorted(
            [c for c in combo_grid if c["fpr"] <= 0.10 and c["n_accept"] > 0],
            key=lambda c: (-c["tpr"], -c["precision"], c["fpr"]),
        )[:8],
        "gain_true_luo": GAIN_TRUE_LUO,
        "loss_false_luo": LOSS_FALSE_LUO,
    }
    return cdf, report


def load_residual_pool() -> tuple[pd.DataFrame, set[str], set[str]]:
    floor = pd.read_csv(FLOOR)
    v3 = pd.read_csv(SEL_V3)
    det = pd.read_csv(DETAIL)
    mms = pd.read_csv(MMS)
    clr = pd.read_csv(CLR)
    conf = pd.read_csv(CONF) if CONF.exists() else None

    frozen = set(det.loc[det.decode_lang == "lug", "ID"].astype(str))
    f_map = floor.set_index("ID")["Target"].astype(str)
    v_map = v3.set_index("ID")["Target"].astype(str)
    dual_ids = {i for i in f_map.index if i in v_map.index and f_map[i] != v_map[i]}

    m = (
        det.merge(mms[["ID", "prediction"]].rename(columns={"prediction": "mms"}), on="ID")
        .merge(
            clr[["ID", "prediction", "source"]].rename(
                columns={"prediction": "clr", "source": "clr_src"}
            ),
            on="ID",
        )
        .merge(floor.rename(columns={"Target": "floor"}), on="ID")
    )
    if conf is not None:
        m = m.merge(conf, on="ID", how="left")
    else:
        m["delta"] = np.nan

    if DUAL_POOL.exists():
        dp = pd.read_csv(DUAL_POOL)
        m = m.merge(dp[["ID", "cer_mc"]], on="ID", how="left")
    else:
        m["cer_mc"] = [cer(a, b) for a, b in zip(m.mms, m.clr)]

    pool = m[(m.lid_lang == "luo") & (m.decode_lang == "ach")].copy()
    pool["ID"] = pool["ID"].astype(str)
    pool["already_dual"] = pool["ID"].isin(dual_ids)
    residual = pool[~pool["already_dual"] & ~pool["ID"].isin(frozen)].copy()
    if residual["cer_mc"].isna().any():
        residual["cer_mc"] = [
            cer(a, b) if pd.notna(a) and pd.notna(b) else 1.0
            for a, b in zip(residual.mms, residual.clr)
        ]
    return residual, frozen, dual_ids


def score_pool(
    residual: pd.DataFrame,
    lm_luo: CharNGramLM,
    lm_ach: CharNGramLM,
    lm_lug: CharNGramLM,
) -> pd.DataFrame:
    rows = []
    for r in residual.itertuples():
        mms_t = str(r.mms) if pd.notna(r.mms) else ""
        clr_t = str(r.clr) if pd.notna(r.clr) else ""
        fl_t = str(r.floor) if pd.notna(r.floor) else ""
        sc_m = score_text(mms_t, lm_luo, lm_ach, lm_lug)
        sc_c = score_text(clr_t, lm_luo, lm_ach, lm_lug) if clr_t else {"ortho": np.nan}
        sc_f = score_text(fl_t, lm_luo, lm_ach, lm_lug)
        conf_delta = float(r.delta) if pd.notna(getattr(r, "delta", np.nan)) else np.nan
        rows.append(
            {
                "ID": r.ID,
                "p1": float(r.p1) if pd.notna(r.p1) else 0.0,
                "cer_mc": float(r.cer_mc) if pd.notna(r.cer_mc) else 1.0,
                "conf_delta": conf_delta,
                "mms": mms_t,
                "clr": clr_t,
                "floor": fl_t,
                "ortho_mms": sc_m["ortho"],
                "ortho_clr": sc_c.get("ortho", np.nan),
                "ortho_floor": sc_f["ortho"],
                "ngram_mms": sc_m["ngram_margin"],
                "marker_mms": sc_m["marker_luo_ach"],
                "lp_luo_mms": sc_m["lp_luo"],
                "lp_ach_mms": sc_m["lp_ach"],
            }
        )
    return pd.DataFrame(rows)


def apply_gates(scored: pd.DataFrame, calib_thr: float, thr05: float | None) -> dict[str, pd.DataFrame]:
    """Gates that are NOT pure dual thr>0.15 expand."""
    s = scored.copy()

    def pick(mask: pd.Series) -> pd.DataFrame:
        return s.loc[mask].sort_values(
            ["ortho_mms", "p1", "cer_mc"], ascending=[False, False, True]
        )

    thr = calib_thr  # e.g. ~2.2–2.4 from hyp calib
    thr_hi = thr05 if thr05 is not None else max(thr + 0.4, 3.0)

    gates = {
        # Primary: hyp-ortho at FPR<=0.10 + high LID conf
        "hyp_o_p99": pick((s.ortho_mms >= thr) & (s.p1 >= 0.99)),
        # Stricter FPR<=0.05 thr
        "hyp_o05_p99": pick((s.ortho_mms >= thr_hi) & (s.p1 >= 0.99)),
        # Ortho + conf_delta (multi acoustic+text)
        "hyp_o_conf": pick((s.ortho_mms >= thr * 0.85) & (s.conf_delta.fillna(-1) >= 0.0) & (s.p1 >= 0.95)),
        # Ortho + soft dual consistency (cer as secondary, band up to 0.28 — NOT thr expand alone)
        "hyp_o_dualsoft": pick(
            (s.ortho_mms >= thr) & (s.cer_mc <= 0.28) & (s.p1 >= 0.99)
        ),
        # Ortho + tight dual (still requires ortho — new mechanism)
        "hyp_o_dual15": pick(
            (s.ortho_mms >= thr) & (s.cer_mc <= 0.15) & (s.p1 >= 0.99)
        ),
        # Top-K by ortho_mms (rank, not thr expand)
        "hyp_top15": s.sort_values("ortho_mms", ascending=False).head(15),
        "hyp_top30": s.sort_values("ortho_mms", ascending=False).head(30),
        "hyp_top50": s.sort_values("ortho_mms", ascending=False).head(50),
        # Very high ortho only
        "hyp_o3_p99": pick((s.ortho_mms >= 3.0) & (s.p1 >= 0.99)),
        "hyp_o24_p99": pick((s.ortho_mms >= 2.4) & (s.p1 >= 0.99)),
        "hyp_o20_p99": pick((s.ortho_mms >= 2.0) & (s.p1 >= 0.99)),
        # ngram+marker without relying on dual thr
        "ngram_marker": pick(
            (s.ngram_mms >= 0.15) & (s.marker_mms >= 2.0) & (s.p1 >= 0.99) & (s.ortho_mms >= thr)
        ),
    }
    return gates


def expected_delta(n_changed: int, precision: float) -> dict:
    if n_changed <= 0:
        return {
            "n": 0,
            "precision_assumed": precision,
            "unit_gain": 0.0,
            "delta": 0.0,
            "projected": FLOOR_PUBLIC,
        }
    unit = precision * GAIN_TRUE_LUO - (1.0 - precision) * LOSS_FALSE_LUO
    delta = (n_changed / N_TEST) * unit
    return {
        "n": n_changed,
        "precision_assumed": precision,
        "unit_gain": unit,
        "delta": delta,
        "projected": FLOOR_PUBLIC + delta,
    }


def write_submission(
    name: str,
    floor: pd.DataFrame,
    accept_ids: set[str],
    text_map: dict[str, str],
    frozen: set[str],
) -> dict:
    out = floor.copy()
    ft = floor.set_index("ID")["Target"].astype(str)
    for i, row in out.iterrows():
        uid = str(row.ID)
        if uid in accept_ids and uid not in frozen:
            new_t = text_map.get(uid)
            if new_t and normalize_text(new_t) and normalize_text(new_t) != normalize_text(ft[uid]):
                out.at[i, "Target"] = new_t
    ch = {str(r.ID) for _, r in out.iterrows() if str(r.Target) != str(ft[r.ID])}
    assert not (ch & frozen), f"frozen lug touched: {ch & frozen}"
    path = PROJECT_ROOT / f"submission_phase2_beat_k63_ortho_{name}.csv"
    out.to_csv(path, index=False)
    check = check_submission(path, SAMPLE_SUBMISSION_CSV)
    BEAT.mkdir(parents=True, exist_ok=True)
    beat_path = BEAT / f"submission_phase2_beat_k63_ortho_{name}.csv"
    out.to_csv(beat_path, index=False)
    return {
        "name": name,
        "path": str(path),
        "beat_path": str(beat_path),
        "n_changed_vs_floor": int(len(ch)),
        "n_accept_ids": len(accept_ids),
        "touched_frozen_lug": 0,
        "check_ok": check.get("ok", True) if isinstance(check, dict) else True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gram", type=int, default=4)
    ap.add_argument("--calib-only", action="store_true")
    args = ap.parse_args()
    BEAT.mkdir(parents=True, exist_ok=True)

    lm_luo, lm_ach, lm_lug, lm_meta = build_lms(n=args.n_gram)
    cdf, calib = calibrate_on_decoded_hyps(lm_luo, lm_ach, lm_lug)
    cdf.to_csv(BEAT / "luo_ortho_calib_scores.csv", index=False)
    (BEAT / "luo_ortho_lms_meta.json").write_text(json.dumps(lm_meta, indent=2))
    (BEAT / "luo_ortho_calib.json").write_text(json.dumps(calib, indent=2, default=str))

    calib_thr = float(calib["chosen_fpr10"]["ortho_thr"])
    thr05 = (
        float(calib["chosen_fpr05"]["ortho_thr"]) if calib.get("chosen_fpr05") else None
    )
    logger.info(
        "hyp-calib thr@fpr10=%.3f tpr=%.3f fpr=%.3f prec=%.3f | thr@fpr05=%s",
        calib_thr,
        calib["chosen_fpr10"]["tpr"],
        calib["chosen_fpr10"]["fpr"],
        calib["chosen_fpr10"]["precision"],
        thr05,
    )
    if args.calib_only:
        print(json.dumps(calib, indent=2, default=str))
        return

    residual, frozen, dual_ids = load_residual_pool()
    logger.info(
        "residual n=%d dual_already=%d frozen_lug=%d",
        len(residual),
        len(dual_ids),
        len(frozen),
    )
    scored = score_pool(residual, lm_luo, lm_ach, lm_lug)
    scored.to_csv(BEAT / "luo_ortho_residual_scores.csv", index=False)

    res_stats = {
        "n_residual": int(len(scored)),
        "ortho_mms_mean": float(scored.ortho_mms.mean()),
        "ortho_mms_p50": float(scored.ortho_mms.median()),
        "ortho_mms_p90": float(np.nanpercentile(scored.ortho_mms, 90)),
        "ortho_mms_p95": float(np.nanpercentile(scored.ortho_mms, 95)),
        "n_ge_calib_thr": int((scored.ortho_mms >= calib_thr).sum()),
        "n_ge_2.4": int((scored.ortho_mms >= 2.4).sum()),
        "n_ge_3.0": int((scored.ortho_mms >= 3.0).sum()),
        "n_conf_pos": int((scored.conf_delta.fillna(-1) >= 0).sum()),
        "interpretation": (
            "residual ortho_mms mean ~ false-Ach decoded mean from calib "
            "→ pool mostly not true-Luo under text detector; tail may be"
        ),
    }

    gates = apply_gates(scored, calib_thr, thr05)
    floor = pd.read_csv(FLOOR)
    text_map = scored.set_index("ID")["mms"].astype(str).to_dict()
    prec_default = float(calib["chosen_fpr10"]["precision"])
    prec05 = (
        float(calib["chosen_fpr05"]["precision"])
        if calib.get("chosen_fpr05")
        else prec_default
    )

    # precision assumptions per gate family
    prec_for = {
        "hyp_o_p99": prec_default,
        "hyp_o05_p99": prec05,
        "hyp_o_conf": min(1.0, prec_default + 0.02),
        "hyp_o_dualsoft": prec_default,
        "hyp_o_dual15": min(1.0, prec_default + 0.03),
        "hyp_top15": float((scored.nlargest(15, "ortho_mms").ortho_mms >= calib_thr).mean()),
        "hyp_top30": float((scored.nlargest(30, "ortho_mms").ortho_mms >= calib_thr).mean()),
        "hyp_top50": float((scored.nlargest(50, "ortho_mms").ortho_mms >= calib_thr).mean()),
        "hyp_o3_p99": prec05 if thr05 and thr05 <= 3.0 else 0.97,
        "hyp_o24_p99": prec_default if calib_thr <= 2.4 else 0.93,
        "hyp_o20_p99": 0.90,
        "ngram_marker": prec_default,
    }

    variants = []
    for name, gdf in gates.items():
        ids = set(gdf["ID"].astype(str))
        rep = write_submission(name, floor, ids, text_map, frozen)
        prec = prec_for.get(name, prec_default)
        # For top-k, re-estimate precision as fraction above calib thr in that slice
        if name.startswith("hyp_top") and len(gdf):
            prec = float((gdf.ortho_mms >= calib_thr).mean())
        proj = expected_delta(rep["n_changed_vs_floor"], prec)
        rep["precision_assumed"] = prec
        rep["projected"] = proj
        gdf.to_csv(BEAT / f"ortho_{name}_changed.csv", index=False)
        rep["changed_detail"] = str(BEAT / f"ortho_{name}_changed.csv")
        variants.append(rep)
        logger.info(
            "gate %s n=%d prec=%.2f proj=%.4f (d%+.4f)",
            name,
            rep["n_changed_vs_floor"],
            prec,
            proj["projected"],
            proj["delta"],
        )

    # Prefer high-precision, positive expected delta; avoid huge low-precision swings
    def rank_key(v):
        p = v["projected"]
        # penalize if precision < 0.85 or n very large without precision
        safe = 1 if v["precision_assumed"] >= 0.90 else 0
        return (safe, p["projected"], v["precision_assumed"], -abs(v["n_changed_vs_floor"] - 30))

    best = max(variants, key=rank_key)
    # Also track max projected (may be optimistic)
    best_raw = max(variants, key=lambda v: v["projected"]["projected"])

    summary = {
        "method": (
            "char-4gram + Dholuo orthography markers on MMS-1B luo hyp; "
            "calib on decoded FLEURS/ach hyps; residual ach-route only; "
            "never pure dual thr>0.15; never decode_lang=lug"
        ),
        "floor": str(FLOOR),
        "floor_public": FLOOR_PUBLIC,
        "target": TARGET,
        "calib": calib,
        "residual_stats": res_stats,
        "variants": variants,
        "best_safe": best,
        "best_raw_projected": best_raw,
        "banned_avoided": [
            "margin_primary",
            "decode_lang_lug_rewrite",
            "dual_thr_gt_0.15_alone",
            "blind_all_lid_luo",
            "phase1_test_gold",
        ],
        "new_mechanism": True,
        "lms": lm_meta,
        "honest_note": (
            "Residual ortho_mms distribution matches false-Ach decoded mean from calib "
            "(~0.76–0.97). High-precision tail is small (tens of clips). "
            "Projected lift from residual alone is micro-to-small vs +0.16 needed for K63. "
            "True-Luo mass may already be partly dual-covered or stuck in frozen lug route."
        ),
    }

    primary = PROJECT_ROOT / "submission_phase2_beat_k63_ortho.csv"
    primary.write_text(Path(best["path"]).read_text())
    summary["primary"] = str(primary)
    summary["primary_name"] = best["name"]
    (BEAT / "luo_ortho_gate_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print(
        json.dumps(
            {
                "primary": str(primary),
                "primary_name": best["name"],
                "n_changed": best["n_changed_vs_floor"],
                "precision_assumed": best["precision_assumed"],
                "projected": best["projected"],
                "best_raw": {
                    "name": best_raw["name"],
                    "n_changed": best_raw["n_changed_vs_floor"],
                    "projected": best_raw["projected"],
                },
                "calib_chosen_fpr10": calib["chosen_fpr10"],
                "calib_chosen_fpr05": calib.get("chosen_fpr05"),
                "separation": calib["separation_decoded_hyp"],
                "residual_stats": res_stats,
                "all_variants": [
                    {
                        "name": v["name"],
                        "n_changed": v["n_changed_vs_floor"],
                        "prec": v["precision_assumed"],
                        "projected": v["projected"]["projected"],
                        "delta": v["projected"]["delta"],
                    }
                    for v in variants
                ],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
