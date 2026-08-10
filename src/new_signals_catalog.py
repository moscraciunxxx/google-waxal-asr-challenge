"""Pure helpers for new-signal investigation catalog (no model I/O).

Loads the durable JSON catalog under outputs/new_signals/ and validates
structure required by the multi-agent new-signals goal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

REQUIRED_SIGNAL_FIELDS = (
    "id",
    "name",
    "inputs",
    "gate",
    "why_new",
    "ban_risk",
    "n_replace_band",
)

# Map JSON keys -> plan acceptance (a)-(f)
FIELD_ALIASES = {
    "name": "a_name",
    "inputs": "b_inputs",
    "gate": "c_gate",
    "why_new": "d_why_new",
    "ban_risk": "e_ban_risk",
    "n_replace_band": "f_n_band",
}


def load_catalog(path: Path | str) -> dict[str, Any]:
    """Load signals_catalog.json from disk."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("catalog root must be object")
    return data


def list_signals(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the signals list (empty if missing)."""
    sigs = catalog.get("signals") or []
    if not isinstance(sigs, list):
        raise TypeError("signals must be a list")
    return [s for s in sigs if isinstance(s, dict)]


def signal_has_required_fields(sig: Mapping[str, Any]) -> bool:
    """True iff signal has non-empty string values for plan fields (a)-(f) inputs."""
    for key in REQUIRED_SIGNAL_FIELDS:
        val = sig.get(key)
        if not isinstance(val, str) or not val.strip():
            return False
    return True


def count_valid_signals(catalog: Mapping[str, Any]) -> int:
    """Count signals that pass required-field validation."""
    return sum(1 for s in list_signals(catalog) if signal_has_required_fields(s))


def shortlist_ids(catalog: Mapping[str, Any]) -> list[str]:
    """Return shortlist_top3 ids if present."""
    sl = catalog.get("shortlist_top3") or []
    return [str(x) for x in sl]


def catalog_is_grounded(catalog: Mapping[str, Any], *, floor: float = 0.560605696) -> bool:
    """Check catalog anchors floor score and dual15 saturation notes."""
    try:
        fp = float(catalog.get("floor_public"))
    except (TypeError, ValueError):
        return False
    if abs(fp - floor) > 1e-9:
        return False
    if int(catalog.get("dual15_new_under_thr_p99", -1)) != 0:
        return False
    exhausted = catalog.get("exhausted_micro_levers") or []
    if len(exhausted) < 3:
        return False
    bans = catalog.get("bans") or []
    if len(bans) < 3:
        return False
    return True


def validate_catalog(
    catalog: Mapping[str, Any],
    *,
    min_signals: int = 10,
    min_shortlist: int = 3,
) -> dict[str, Any]:
    """Return validation report; does not raise on soft failures."""
    n_valid = count_valid_signals(catalog)
    sl = shortlist_ids(catalog)
    return {
        "n_signals_valid": n_valid,
        "min_signals_ok": n_valid >= min_signals,
        "shortlist": sl,
        "shortlist_ok": len(sl) >= min_shortlist,
        "grounded": catalog_is_grounded(catalog),
        "public_win_claimed": bool(catalog.get("public_win_claimed", False)),
        "ok": (
            n_valid >= min_signals
            and len(sl) >= min_shortlist
            and catalog_is_grounded(catalog)
            and not catalog.get("public_win_claimed", False)
        ),
    }
