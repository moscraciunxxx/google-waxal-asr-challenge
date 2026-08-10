#!/usr/bin/env python3
"""Strictly compose validated Phase-2 route caches into submission ablations.

This builder is deliberately prediction-only.  It reads:

* a complete Phase-2 submission used as the immutable base;
* the public-visible routing index, projected to ID/route/split only;
* the public-descendant validation report; and
* model prediction caches containing only ID and Target.

It never reads a Phase-2 reference, transcript, ``floor`` value, or audio file,
and it never uploads a submission.  A model is eligible only when the report
sets ``metrics.strong_pass`` to true and its cache is an exact, non-empty ID-set
match for a route scope represented in ``public_visible_index.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submission import check_phase2_submission


DEFAULT_BASE = (
    ROOT
    / "outputs"
    / "goal_2026_08_08"
    / "nyn_ensemble"
    / "submission_phase2_nyn_cv_ensemble_guarded.csv"
)
DEFAULT_INDEX = ROOT / "outputs" / "beat075" / "public_visible_index.csv"
DEFAULT_REPORT = (
    ROOT
    / "outputs"
    / "goal_2026_08_08"
    / "sulaiman_public_descendants"
    / "report.json"
)
DEFAULT_OUT_DIR = ROOT / "outputs" / "goal_2026_08_08" / "combined"

# The producer's Phase-2 descendant caches are decoded over the expanded
# (split=new) LIN/SNA block.  A future cache covering the complete route is also
# accepted, but arbitrary subsets and supersets are rejected.
SUPPORTED_ROUTES = {"lin", "sna"}
LANGUAGE_MARKERS = {
    "lin": ("lingala",),
    "sna": ("shona",),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_report_path(value: Any, report_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (report_path.parent / path).resolve()
    return path


def infer_route(tag: str, model: dict[str, Any]) -> str | None:
    audit = model.get("audit") if isinstance(model.get("audit"), dict) else {}
    haystack = " ".join(
        str(value).lower()
        for value in (tag, audit.get("model_id"), audit.get("checkpoint"))
        if value
    )
    matches = [
        route
        for route, markers in LANGUAGE_MARKERS.items()
        if any(marker in haystack for marker in markers)
    ]
    return matches[0] if len(matches) == 1 else None


def metric_number(metrics: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_submission(path: Path, label: str) -> pd.DataFrame:
    # Inspect only the header before parsing rows.  This both enforces the
    # narrow prediction-only contract and prevents accidental ingestion of an
    # unexpected transcript/reference column.
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), None)
    if header != ["ID", "Target"]:
        raise RuntimeError(f"{label}: strict schema must be ID,Target; got {header}")
    frame = pd.read_csv(
        path,
        usecols=["ID", "Target"],
        dtype={"ID": str, "Target": str},
        keep_default_na=False,
    )
    if frame.ID.duplicated().any():
        raise RuntimeError(f"{label}: duplicate IDs")
    invalid = frame.Target.map(lambda value: not str(value).strip())
    if invalid.any():
        raise RuntimeError(f"{label}: {int(invalid.sum())} empty targets")
    return frame


def cache_path_for(tag: str, model: dict[str, Any], report_path: Path) -> Path:
    explicit = resolve_report_path(model.get("phase2_cache"), report_path)
    if explicit is not None:
        return explicit
    return report_path.parent / f"phase2_cache_{tag}.csv"


def inspect_model_cache(
    *,
    tag: str,
    model: dict[str, Any],
    report_path: Path,
    route_index: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    metrics = model.get("metrics") if isinstance(model.get("metrics"), dict) else {}
    strong_pass = metrics.get("strong_pass") is True
    route = infer_route(tag, model)
    path = cache_path_for(tag, model, report_path)
    audit: dict[str, Any] = {
        "tag": tag,
        "route": route,
        "strong_pass": strong_pass,
        "cache": str(path),
        "cache_exists": path.is_file(),
        "eligible": False,
        "reason": None,
        "metrics": {
            "candidate_zindi": metric_number(metrics, ("candidate", "zindi")),
            "incumbent_zindi": metric_number(metrics, ("incumbent", "zindi")),
            "delta_zindi": metric_number(metrics, ("delta_zindi",)),
            "candidate_wer": metric_number(metrics, ("candidate", "wer")),
            "candidate_cer": metric_number(metrics, ("candidate", "cer")),
            "n": metric_number(metrics, ("candidate", "n")),
        },
    }
    if not strong_pass:
        audit["reason"] = "report metrics.strong_pass is not exactly true"
        return audit, None
    if route not in SUPPORTED_ROUTES:
        audit["reason"] = "model route is missing, ambiguous, or unsupported"
        return audit, None
    if not path.is_file():
        audit["reason"] = "Phase2 prediction cache does not exist"
        return audit, None

    try:
        cache = read_submission(path, f"cache {tag}")
    except Exception as exc:  # A producer may be replacing a partial CSV now.
        audit["reason"] = f"cache unreadable or structurally invalid: {exc}"
        return audit, None

    cache_ids = set(cache.ID)
    all_route_ids = set(route_index.loc[route_index.decode_lang == route, "ID"])
    new_route_ids = set(
        route_index.loc[
            (route_index.decode_lang == route) & (route_index.split == "new"), "ID"
        ]
    )
    accepted_scopes = {
        "public_visible_route_all_splits": all_route_ids,
        "public_visible_route_split_new": new_route_ids,
    }
    # When all route rows are new (currently LIN), avoid two names for one set.
    if all_route_ids == new_route_ids:
        accepted_scopes.pop("public_visible_route_split_new")
    matching = [name for name, expected in accepted_scopes.items() if cache_ids == expected]
    audit.update(
        {
            "cache_rows": len(cache),
            "cache_unique_ids": len(cache_ids),
            "cache_id_sha256": sha256_lines(sorted(cache_ids)),
            "expected_all_route_rows": len(all_route_ids),
            "expected_new_route_rows": len(new_route_ids),
            "cache_sha256": sha256_file(path),
        }
    )
    if len(matching) != 1:
        best_expected = new_route_ids if len(cache_ids) <= len(new_route_ids) else all_route_ids
        audit["missing_expected_ids"] = len(best_expected - cache_ids)
        audit["extra_ids"] = len(cache_ids - all_route_ids)
        audit["reason"] = "cache ID set is not an exact accepted public-visible route scope"
        return audit, None

    reported_rows = model.get("phase2_cache_rows")
    if reported_rows is not None and int(reported_rows) != len(cache):
        audit["reason"] = (
            f"report phase2_cache_rows={reported_rows} disagrees with cache rows={len(cache)}"
        )
        return audit, None

    delta = audit["metrics"]["delta_zindi"]
    candidate_zindi = audit["metrics"]["candidate_zindi"]
    if not isinstance(delta, (int, float)) or not math.isfinite(delta):
        audit["reason"] = "required candidate/delta Zindi metrics are absent or non-finite"
        return audit, None
    if not isinstance(candidate_zindi, (int, float)) or not math.isfinite(candidate_zindi):
        audit["reason"] = "required candidate/delta Zindi metrics are absent or non-finite"
        return audit, None

    audit["scope"] = matching[0]
    audit["eligible"] = True
    audit["reason"] = "strong pass and exact complete route cache"
    return audit, cache


def model_rank(audit: dict[str, Any]) -> tuple[float, float, str]:
    metrics = audit["metrics"]
    return (
        float(metrics["candidate_zindi"]),
        float(metrics["delta_zindi"]),
        str(audit["tag"]),
    )


def safe_tag(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def write_candidate(
    *,
    name: str,
    base: pd.DataFrame,
    overlays: list[tuple[dict[str, Any], pd.DataFrame]],
    route_by_id: dict[str, str],
    public_sensitive_rows: int,
    out_dir: Path,
) -> dict[str, Any]:
    candidate = base.copy()
    base_targets = dict(zip(base.ID, base.Target))
    applied: list[dict[str, Any]] = []
    claimed_ids: set[str] = set()

    for audit, cache in overlays:
        cache_ids = set(cache.ID)
        overlap = claimed_ids & cache_ids
        if overlap:
            raise RuntimeError(
                f"{name}: overlapping selected route caches, e.g. {sorted(overlap)[:5]}"
            )
        claimed_ids |= cache_ids
        values = dict(zip(cache.ID, cache.Target))
        candidate.loc[candidate.ID.isin(cache_ids), "Target"] = candidate.loc[
            candidate.ID.isin(cache_ids), "ID"
        ].map(values)
        delta = float(audit["metrics"]["delta_zindi"])
        projection = delta * len(cache_ids) / public_sensitive_rows
        applied.append(
            {
                "route": audit["route"],
                "model": audit["tag"],
                "scope": audit["scope"],
                "cache_rows": len(cache_ids),
                "offline_route_delta_zindi": delta,
                "candidate_route_zindi": audit["metrics"]["candidate_zindi"],
                "incumbent_route_zindi": audit["metrics"]["incumbent_zindi"],
                "crude_row_weighted_public_projection_delta": projection,
            }
        )

    candidate_targets = dict(zip(candidate.ID, candidate.Target))
    changed_ids = [uid for uid in base.ID if candidate_targets[uid] != base_targets[uid]]
    if not set(changed_ids) <= claimed_ids:
        raise RuntimeError(f"{name}: candidate changed IDs outside selected caches")
    changed_routes = Counter(route_by_id[uid] for uid in changed_ids)

    out_path = out_dir / f"submission_phase2_{name}.csv"
    candidate.to_csv(out_path, index=False)
    validation = check_phase2_submission(out_path, strict=True)
    if not validation["ok"]:
        raise RuntimeError(f"{name}: strict Phase2 validation failed: {validation['errors']}")
    roundtrip = read_submission(out_path, name)
    if roundtrip.ID.tolist() != base.ID.tolist():
        raise RuntimeError(f"{name}: output ID order changed")
    if roundtrip.Target.tolist() != candidate.Target.tolist():
        raise RuntimeError(f"{name}: CSV round-trip changed predictions")

    return {
        "name": name,
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "rows": len(candidate),
        "unique_ids": candidate.ID.nunique(),
        "empty_targets": int(candidate.Target.map(lambda value: not str(value).strip()).sum()),
        "id_order_matches_base": candidate.ID.tolist() == base.ID.tolist(),
        "id_set_matches_base": set(candidate.ID) == set(base.ID),
        "changed_rows_vs_base": len(changed_ids),
        "changed_route_counts": dict(sorted(changed_routes.items())),
        "changed_ids_sha256": sha256_lines(sorted(changed_ids)),
        "applied_routes": applied,
        "crude_row_weighted_public_projection_delta": sum(
            item["crude_row_weighted_public_projection_delta"] for item in applied
        ),
        "projection_warning": (
            "Crude validation-delta x utterance-share estimate only; leaderboard WER/CER "
            "is token-weighted and validation-to-test transfer is uncertain."
        ),
        "strict_phase2_validation": validation,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    base = read_submission(args.base, "base")
    base_validation = check_phase2_submission(args.base, strict=True)
    if not base_validation["ok"]:
        raise RuntimeError(f"base strict Phase2 validation failed: {base_validation['errors']}")

    route_index = pd.read_csv(
        args.index,
        usecols=["ID", "decode_lang", "split"],
        dtype={"ID": str, "decode_lang": str, "split": str},
        keep_default_na=False,
    )
    if route_index.ID.duplicated().any():
        raise RuntimeError("public-visible index contains duplicate IDs")
    if not set(route_index.ID) <= set(base.ID):
        raise RuntimeError("public-visible index contains IDs absent from the base")
    route_by_id = dict(zip(route_index.ID, route_index.decode_lang))
    public_sensitive_rows = len(route_index)

    report = json.loads(args.report.read_text())
    if report.get("protocol", {}).get("test_labels_read") is not False:
        raise RuntimeError("report does not explicitly attest test_labels_read=false")
    models = report.get("models")
    if not isinstance(models, dict):
        raise RuntimeError("report models section is missing or invalid")

    audits: list[dict[str, Any]] = []
    eligible: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for tag, model in sorted(models.items()):
        if not isinstance(model, dict):
            audits.append(
                {
                    "tag": tag,
                    "eligible": False,
                    "reason": "model report entry is not an object",
                }
            )
            continue
        audit, cache = inspect_model_cache(
            tag=tag,
            model=model,
            report_path=args.report,
            route_index=route_index,
        )
        audits.append(audit)
        if audit["eligible"] and cache is not None:
            eligible.append((audit, cache))

    if not eligible:
        raise RuntimeError("no model currently has both strong_pass=true and a complete route cache")

    selected: dict[str, tuple[dict[str, Any], pd.DataFrame]] = {}
    for audit, cache in eligible:
        route = str(audit["route"])
        incumbent = selected.get(route)
        if incumbent is None or model_rank(audit) > model_rank(incumbent[0]):
            selected[route] = (audit, cache)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    if "lin" in selected:
        candidates.append(
            write_candidate(
                name="lin_only",
                base=base,
                overlays=[selected["lin"]],
                route_by_id=route_by_id,
                public_sensitive_rows=public_sensitive_rows,
                out_dir=args.out_dir,
            )
        )
    if "sna" in selected:
        candidates.append(
            write_candidate(
                name="sna_only",
                base=base,
                overlays=[selected["sna"]],
                route_by_id=route_by_id,
                public_sensitive_rows=public_sensitive_rows,
                out_dir=args.out_dir,
            )
        )
    candidates.append(
        write_candidate(
            name="combined",
            base=base,
            overlays=[selected[route] for route in sorted(selected)],
            route_by_id=route_by_id,
            public_sensitive_rows=public_sensitive_rows,
            out_dir=args.out_dir,
        )
    )

    selected_tags = {route: pair[0]["tag"] for route, pair in sorted(selected.items())}
    for audit in audits:
        if audit.get("eligible") and selected_tags.get(audit.get("route")) != audit.get("tag"):
            audit["selected"] = False
            audit["reason"] = (
                "eligible but a stronger same-route candidate was selected by candidate "
                "Zindi score, then delta, then deterministic tag tie-break"
            )
        else:
            audit["selected"] = selected_tags.get(audit.get("route")) == audit.get("tag")

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": str(Path(__file__).resolve()),
        "safety": {
            "test_transcripts_or_targets_read": False,
            "uploads_performed": False,
            "producer_files_modified": False,
            "route_index_columns_read": ["ID", "decode_lang", "split"],
            "cache_columns_read": ["ID", "Target"],
        },
        "base": {
            "path": str(args.base),
            "sha256": sha256_file(args.base),
            "rows": len(base),
            "unique_ids": base.ID.nunique(),
            "empty_targets": int(base.Target.map(lambda value: not str(value).strip()).sum()),
            "strict_phase2_validation": base_validation,
        },
        "route_index": {
            "path": str(args.index),
            "sha256": sha256_file(args.index),
            "public_sensitive_rows": public_sensitive_rows,
            "route_counts": dict(sorted(Counter(route_index.decode_lang).items())),
            "new_route_counts": dict(
                sorted(Counter(route_index.loc[route_index.split == "new", "decode_lang"]).items())
            ),
        },
        "validation_report": {
            "path": str(args.report),
            "sha256": sha256_file(args.report),
            "test_labels_read": report.get("protocol", {}).get("test_labels_read"),
        },
        "selection_rule": (
            "Require metrics.strong_pass=true, finite candidate/delta Zindi metrics, and an "
            "exact cache ID-set match to either the complete public-visible route or the "
            "producer's expanded split=new route. Select the highest candidate Zindi score "
            "per route; break ties by delta then tag."
        ),
        "selected_models": selected_tags,
        "model_audits": audits,
        "candidates": candidates,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    manifest["manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    print(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=0.0,
        help="Poll until at least one strong-pass cache is complete (0 = do not wait)",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wait_seconds < 0 or args.poll_seconds <= 0:
        raise ValueError("wait-seconds must be nonnegative and poll-seconds must be positive")
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            build(args)
            return
        except RuntimeError as exc:
            if "no model currently has both" not in str(exc) or time.monotonic() >= deadline:
                raise
            remaining = max(0.0, deadline - time.monotonic())
            print(f"waiting for a complete strong-pass cache ({remaining:.0f}s remain)", flush=True)
            time.sleep(min(args.poll_seconds, remaining))


if __name__ == "__main__":
    main()
