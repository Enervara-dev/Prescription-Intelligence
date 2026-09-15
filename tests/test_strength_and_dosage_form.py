"""
tests/test_strength_and_dosage_form.py
------------------------------------------
Pure unit tests (no DB) for the deterministic strength parser and
dosage-form normalizer -- kept as SEPARATE stages from name matching, per
the architecture's own rule.
"""

from decimal import Decimal

from app.services.dosage_form import dosage_forms_compatible, normalize_dosage_form, strip_form_words
from app.services.strength_parser import parse_strength, strip_strength_tokens


# -- strength parsing --

def test_parse_simple_strengths():
    assert parse_strength("500 mg").as_dict() == {"strength": 500.0, "unit": "mg"}
    assert parse_strength("500mg").as_dict() == {"strength": 500.0, "unit": "mg"}
    assert parse_strength("650 MG").as_dict() == {"strength": 650.0, "unit": "mg"}
    assert parse_strength("625 mg").as_dict() == {"strength": 625.0, "unit": "mg"}
    assert parse_strength("40mg").as_dict() == {"strength": 40.0, "unit": "mg"}
    assert parse_strength("5 ml").as_dict() == {"strength": 5.0, "unit": "ml"}


def test_parse_per_volume_strength():
    info = parse_strength("100 mg/5 ml")
    assert info.single.value == Decimal("100")
    assert info.single.unit == "mg"
    assert info.per_value == Decimal("5")
    assert info.per_unit == "ml"


def test_parse_combination_strength():
    info = parse_strength("250 mg + 125 mg")
    assert info.is_combination
    assert [c.value for c in info.components] == [Decimal("250"), Decimal("125")]
    assert all(c.unit == "mg" for c in info.components)


def test_parse_strength_not_found():
    assert parse_strength("nothing here").found is False
    assert parse_strength("").found is False
    assert parse_strength("Dolo 650").found is False  # bare number, no unit -- not a parseable strength


def test_parse_strength_never_mixed_with_name_matching():
    # Strength parsing works identically regardless of surrounding name text.
    a = parse_strength("Paracetamol 500 mg")
    b = parse_strength("500 mg")
    assert a.single.value == b.single.value == Decimal("500")
    assert a.single.unit == b.single.unit == "mg"


def test_strip_strength_tokens_only_removes_number_plus_unit():
    assert strip_strength_tokens("paracetamol 500 mg tablet") == "paracetamol tablet"
    # A bare number with no unit (part of a brand identity) is left alone.
    assert strip_strength_tokens("dolo 650") == "dolo 650"


# -- dosage form normalization --

def test_normalize_dosage_forms():
    for word, expected in [
        ("tab", "tablet"), ("tabs", "tablet"), ("tablet", "tablet"), ("tablets", "tablet"),
        ("cap", "capsule"), ("caps", "capsule"), ("capsule", "capsule"), ("capsules", "capsule"),
        ("syp", "syrup"), ("syr", "syrup"), ("syrup", "syrup"),
        ("inj", "injection"), ("injection", "injection"),
        ("susp", "suspension"), ("suspension", "suspension"),
        ("drop", "drops"), ("drops", "drops"),
    ]:
        assert normalize_dosage_form(word) == expected, word


def test_normalize_dosage_form_not_found():
    assert normalize_dosage_form("paracetamol 500mg") is None
    assert normalize_dosage_form("") is None


def test_dosage_forms_compatible():
    assert dosage_forms_compatible("tablet", "tablet") is True
    assert dosage_forms_compatible("tablet", "syrup") is False
    assert dosage_forms_compatible(None, "syrup") is True  # unknown -> don't penalize
    assert dosage_forms_compatible("tablet", None) is True


def test_strip_form_words():
    assert strip_form_words("paracetamol 500mg tablet") == "paracetamol 500mg"
    assert strip_form_words("dolo 650") == "dolo 650"  # nothing to strip
