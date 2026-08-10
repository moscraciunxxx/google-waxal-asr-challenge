from pathlib import Path

from scripts.validate_public_submission import validate


ROOT = Path(__file__).resolve().parents[1]


def test_curated_public_submission_is_uploadable():
    report = validate(ROOT / "submission" / "phase2_public_final.csv")
    assert report["ok"], report["errors"]
    assert report["rows"] == 2392
    assert report["empty_or_placeholder_targets"] == 0
