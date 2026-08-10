"""Pure helpers for the legitimate multi-family decode pipeline.

No model I/O — safe for unit tests and structural packing of hyp rows.
"""

from __future__ import annotations

from typing import Any

from src.text_norm import normalize_text

# ISO codes with open WAXALNet MMS-300m specialists (decode spine)
WAXAL300_LANGS = frozenset(
    {
        "lug",
        "lin",
        "sna",
        "ach",
        "sog",
        "nyn",
        "mas",
        "aka",
        "ewe",
        "ful",
        "orm",
        "amh",
        "tir",
        "wal",
        "dag",
        "dga",
        "kpo",
        "sid",
        "mlg",
    }
)

# LID → specialist fallback chain (same spirit as phase2 openset)
LID_FALLBACK: dict[str, list[str]] = {
    "luo": ["ach", "lug"],
    "swh": ["lug", "lin"],
    "swa": ["lug", "lin"],
    "kin": ["nyn", "lug"],
    "nya": ["sna", "lug"],
    "umb": ["sog", "lug"],
    "nso": ["sna", "lug"],
    "wol": ["ful", "lug"],
    "eng": ["lug", "lin", "sna"],
}

# Whisper forced-language token names (OpenAI Whisper language codes)
WHISPER_LANG_NAME: dict[str, str] = {
    "lin": "lingala",
    "sna": "shona",
    "swa": "swahili",
    "swh": "swahili",
    "eng": "english",
    "fra": "french",
    "amh": "amharic",
    "yor": "yoruba",
    "hau": "hausa",
}


def resolve_decode_lang(lid_lang: str, *, default: str = "lug") -> str:
    """Map LID code to a WAXALNet specialist code."""
    lid = (lid_lang or "").strip().lower()
    if lid in WAXAL300_LANGS:
        return lid
    for cand in LID_FALLBACK.get(lid, []):
        if cand in WAXAL300_LANGS:
            return cand
    if lid in ("lin", "sna", "lug"):
        return lid
    return default


def whisper_language_name(decode_lang: str) -> str | None:
    """Return Whisper language name for forced_decoder_ids, or None if free-decode."""
    code = (decode_lang or "").strip().lower()
    return WHISPER_LANG_NAME.get(code)


def pack_hyp_row(
    sample_id: str,
    *,
    lid_lang: str,
    lid_p1: float | None,
    decode_lang: str,
    mms_hyp: str,
    mms_score: float | None,
    whisper_hyp: str,
    whisper_score: float | None,
    mms_model_id: str = "",
    whisper_model_id: str = "",
    seed: int = 42,
) -> dict[str, Any]:
    """Pack one multi-family decode row (CSV/JSON schema)."""
    mms_n = normalize_text(mms_hyp)
    wh_n = normalize_text(whisper_hyp)
    return {
        "ID": sample_id,
        "lid_lang": (lid_lang or "").strip().lower(),
        "lid_p1": float(lid_p1) if lid_p1 is not None else None,
        "decode_lang": resolve_decode_lang(decode_lang if decode_lang else lid_lang),
        "mms_model_id": mms_model_id,
        "mms_hyp": mms_n,
        "mms_score": float(mms_score) if mms_score is not None else None,
        "whisper_model_id": whisper_model_id,
        "whisper_hyp": wh_n,
        "whisper_score": float(whisper_score) if whisper_score is not None else None,
        "seed": int(seed),
        "omnilingual": "deferred",
    }


def rows_to_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate required keys and ensure hyp fields are strings."""
    required = ("ID", "mms_hyp", "whisper_hyp", "decode_lang", "lid_lang")
    out: list[dict[str, Any]] = []
    for r in rows:
        for k in required:
            if k not in r:
                raise KeyError(f"missing key {k} in row {r.get('ID')}")
        rec = dict(r)
        rec["mms_hyp"] = normalize_text(rec.get("mms_hyp"))
        rec["whisper_hyp"] = normalize_text(rec.get("whisper_hyp"))
        out.append(rec)
    return out
