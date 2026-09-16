"""
tests/test_fixtures_synthetic.py
------------------------------------
Loads every fixture under tests/fixtures/prescriptions/ and runs it through
the real OCR-geometry -> parse_medicines pipeline. See
tests/fixtures/prescriptions/README.md for the honest status of these
fixtures: synthetic, hand-constructed stand-ins for real-world conditions,
NOT a real-world OCR accuracy benchmark (no real prescription image exists
in this repository).
"""

import json
from pathlib import Path

import pytest

from app.repositories import medicine_repository as repo
from app.services.extraction import parse_medicines
from app.services.ocr_geometry import OCRWord, group_words_into_lines

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "prescriptions"
FIXTURE_FILES = sorted(FIXTURES_DIR.glob("*/*.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("fixture_path", FIXTURE_FILES, ids=lambda p: p.stem)
def test_synthetic_fixture_produces_expected_medicines(db, fixture_path):
    fixture = _load(fixture_path)

    for record in fixture["catalog"]:
        repo.upsert_medicine(db, record)
    db.commit()

    words = [OCRWord(**w) for w in fixture["words"]]
    lines = group_words_into_lines(words)
    text = "\n".join(l.text for l in lines)

    results = parse_medicines(text, db, ocr_lines=lines)
    by_name = {r["name"]: r for r in results}

    expected = fixture["expected_medicines"]
    assert set(by_name) == {e["name"] for e in expected}, fixture_path

    for entry in expected:
        med = by_name[entry["name"]]
        if entry.get("strength") is not None:
            assert med["strength"] == entry["strength"], (fixture_path, entry["name"])
        assert med["dosage"] == entry["dosage"], (fixture_path, entry["name"])
        assert med["frequency"] == entry["frequency"], (fixture_path, entry["name"])
        if entry.get("duration") is not None:
            assert med["duration"] == entry["duration"], (fixture_path, entry["name"])

    if fixture.get("expect_overall_confidence_below_match_confidence"):
        for med in by_name.values():
            assert med["overall_confidence"] is not None
            assert med["overall_confidence"] < med["confidence"], fixture_path


def test_fixture_infrastructure_is_honestly_documented():
    """Guards against someone quietly deleting the honesty note this
    directory depends on."""
    readme = (FIXTURES_DIR / "README.md").read_text(encoding="utf-8")
    assert "does not contain any real prescription images" in readme
    assert "NOT a real-world accuracy benchmark" in readme or "real-world accuracy benchmark" in readme
