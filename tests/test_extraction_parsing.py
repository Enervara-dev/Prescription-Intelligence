"""
tests/test_extraction_parsing.py
------------------------------------
Regression tests for app/services/extraction/__init__.py: short medicine
names, multiple medicines per line/reconstructed-row, dosage/strength/
frequency/duration semantics, and the duration-hallucination removal. Each
test is written against a general mechanism (shape-based token filtering,
span scanning, windowed instruction extraction) rather than a name-specific
special case, per the "do not overfit" requirement -- "Pan 40" and "DOLO
650" are the regression examples, not the whole of what's being verified.
"""

import pytest

from app.repositories import medicine_repository as repo
from app.services.extraction import parse_medicines


@pytest.fixture()
def catalog(db):
    repo.upsert_medicine(db, {
        "external_id": "legacy:dolo-650",
        "brand_name": "Dolo 650", "generic_name": "Paracetamol",
        "aliases": ["dolo", "dolo650", "dolo 650", "dolo-650"],
        "strength": "650mg", "strength_value": 650, "strength_unit": "mg",
        "dosage_form": "tablet", "source": "legacy_manual",
    })
    repo.upsert_medicine(db, {
        "external_id": "legacy:pan-40",
        "brand_name": "Pan 40", "generic_name": "Pantoprazole",
        "aliases": ["pan", "pan40", "pan 40", "pan-40"],
        "strength": "40mg", "strength_value": 40, "strength_unit": "mg",
        "dosage_form": "tablet", "source": "legacy_manual",
    })
    repo.upsert_medicine(db, {
        "external_id": "legacy:amoxicillin",
        "generic_name": "Amoxicillin", "source": "legacy_manual",
    })
    db.commit()
    return db


# -- Mandatory regression: short medicine names --------------------------

def test_pan_40_is_detected_not_dropped(catalog):
    """
    Confirmed bug (audit section 3/4): a hard <4-character pre-filter
    rejected "Pan" before the "Pan"+"40" combination was ever tried, even
    though the resolver itself resolves "Pan 40" perfectly. Must be fixed
    generically (shape-based filtering, not length-based) -- not by special-
    casing this one name.
    """
    results = parse_medicines("Tab Pan 40 OD before food", catalog)
    names = {r["name"] for r in results}
    assert "Pan 40" in names


def test_pan_40_detected_as_only_line(catalog):
    results = parse_medicines("Pan 40 OD before food", catalog)
    assert {r["name"] for r in results} == {"Pan 40"}


def test_short_token_that_is_not_a_real_medicine_still_safely_unresolved(catalog):
    """The fix is catalog-aware acceptance, not "accept every short token" --
    a short token with nothing behind it in the catalog must still safely
    produce no result, never a forced guess."""
    results = parse_medicines("Zzq 99 OD 5 days", catalog)
    assert results == []


# -- Mandatory regression: DOLO 650 field semantics ------------------------

def test_dolo_650_full_field_semantics(catalog):
    results = parse_medicines("DOLO 650 1-0-1 5 days", catalog)
    assert len(results) == 1
    med = results[0]
    assert med["name"] == "Dolo 650"
    assert med["strength"] == "650mg"  # catalog-sourced structured strength
    assert med["frequency"] == "1-0-1 (M-A-N)"
    assert med["duration"] == "5 days"
    # The confirmed bug: "dosage" must never be the strength string.
    assert med["dosage"] != "650 mg"
    assert med["dosage"] == "N/A"  # no genuine administration-dose phrase here


def test_amoxicillin_with_explicit_strength(catalog):
    results = parse_medicines("AMOXICILLIN 500 mg 1-0-1 7 days", catalog)
    assert len(results) == 1
    med = results[0]
    assert med["strength_text"] == "500 mg"
    assert med["frequency"] == "1-0-1 (M-A-N)"
    assert med["duration"] == "7 days"


def test_genuine_administration_dosage_is_extracted_when_stated(catalog):
    results = parse_medicines("Dolo 650 1 tablet BD 5 days", catalog)
    assert len(results) == 1
    assert results[0]["dosage"] == "1 tablet"


# -- Mandatory regression: duration hallucination removed -----------------

def test_bracketed_number_never_becomes_days(catalog):
    """Confirmed hallucination (audit section 12): a bare "(5)" with no unit
    anywhere in the text must not become "5 days"."""
    results = parse_medicines("DOLO 650 (5)", catalog)
    assert len(results) == 1
    assert results[0]["duration"] == "N/A"


def test_explicit_duration_units_still_work(catalog):
    for phrase, expected in [
        ("Dolo 650 5 days", "5 days"),
        ("Dolo 650 1 week", "1 week"),
        ("Dolo 650 10 days", "10 days"),
    ]:
        results = parse_medicines(phrase, catalog)
        assert results[0]["duration"] == expected, phrase


# -- Mandatory regression: multiple medicines per line/prescription --------

def test_two_medicines_on_separate_lines_each_keep_their_own_instructions(catalog):
    text = "DOLO 650       1-0-1       5 days\nPAN 40         1-0-0       7 days"
    results = parse_medicines(text, catalog)
    names = {r["name"]: r for r in results}
    assert set(names) == {"Dolo 650", "Pan 40"}
    assert names["Dolo 650"]["frequency"] == "1-0-1 (M-A-N)"
    assert names["Dolo 650"]["duration"] == "5 days"
    assert names["Pan 40"]["frequency"] == "1-0-0 (M-A-N)"
    assert names["Pan 40"]["duration"] == "7 days"
    # Never cross-assigned.
    assert names["Dolo 650"]["duration"] != names["Pan 40"]["duration"]


def test_two_medicines_sharing_one_reconstructed_line(catalog):
    """
    Simulates OCR reconstruction merging two table rows into one line (a
    real, verified OCR-layer failure mode) -- the parser must still be able
    to recover both medicines, each with only ITS OWN nearby instructions,
    rather than being structurally limited to the first match on a line.

    Trailing newline is deliberate: real OCR output is always newline-joined
    (see ocr_service.py). A trailing "\n" makes this explicitly one real
    line followed by an empty (filtered-out) one, rather than relying on
    text.split("\n")'s single-element-list behavior for a bare string with
    no newline at all -- both produce the same one-line result here, but
    this spells out the intent.
    """
    text = "Dolo 650 1-0-1 5 days Pan 40 1-0-0 7 days\n"
    results = parse_medicines(text, catalog)
    names = {r["name"]: r for r in results}
    assert set(names) == {"Dolo 650", "Pan 40"}
    assert names["Dolo 650"]["duration"] == "5 days"
    assert names["Pan 40"]["duration"] == "7 days"


def test_medicine_name_alone_then_instructions_on_following_lines(catalog):
    """
    A common real layout: the medicine name on its own line, with dosage/
    frequency/duration printed on separate subsequent lines (e.g. a wrapped
    or vertically-stacked entry). Bounded lookahead must recover this
    without cross-assigning to the wrong medicine.
    """
    text = "Dolo 650\n1-0-1\n5 days"
    results = parse_medicines(text, catalog)
    assert len(results) == 1
    med = results[0]
    assert med["name"] == "Dolo 650"
    assert med["frequency"] == "1-0-1 (M-A-N)"
    assert med["duration"] == "5 days"


def test_cross_line_association_does_not_leak_into_the_next_medicine(catalog):
    text = "Dolo 650\n1-0-1\n5 days\nPan 40\n1-0-0\n7 days"
    results = parse_medicines(text, catalog)
    names = {r["name"]: r for r in results}
    assert set(names) == {"Dolo 650", "Pan 40"}
    assert names["Dolo 650"]["duration"] == "5 days"
    assert names["Pan 40"]["duration"] == "7 days"


# -- Frequency notation coverage -------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Dolo 650 OD 5 days", "Once Daily"),
    ("Dolo 650 BD 5 days", "Twice Daily"),
    ("Dolo 650 BID 5 days", "Twice Daily"),
    ("Dolo 650 TDS 5 days", "Thrice Daily"),
    ("Dolo 650 TID 5 days", "Thrice Daily"),
    ("Dolo 650 QID 5 days", "Four Times Daily"),
    ("Dolo 650 HS 5 days", "At Bedtime"),
    ("Dolo 650 SOS", "When Required"),
    ("Dolo 650 STAT", "Immediately"),
])
def test_frequency_abbreviations(catalog, text, expected):
    results = parse_medicines(text, catalog)
    assert results[0]["frequency"] == expected, text


def test_fractional_frequency_triad(catalog):
    results = parse_medicines("Dolo 650 1/2-0-1 5 days", catalog)
    assert results[0]["frequency"] == "1/2-0-1 (M-A-N)"


def test_plain_triad_frequency(catalog):
    for text, expected in [
        ("Dolo 650 1-0-1 5 days", "1-0-1 (M-A-N)"),
        ("Dolo 650 1-1-1 5 days", "1-1-1 (M-A-N)"),
        ("Dolo 650 0-0-1 5 days", "0-0-1 (M-A-N)"),
    ]:
        results = parse_medicines(text, catalog)
        assert results[0]["frequency"] == expected, text
