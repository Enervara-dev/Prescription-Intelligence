"""
tests/test_medicine_resolver.py
-----------------------------------
End-to-end tests for the Indian medicine normalization pipeline
(app/services/medicine_resolver.py), against real Postgres.
"""

from app.repositories import medicine_repository as repo
from app.services.medicine_resolver import MatchStatus, resolve


def _seed_dolo(db):
    repo.upsert_medicine(db, {
        "external_id": "legacy:dolo-650",
        "brand_name": "Dolo 650",
        "generic_name": "Paracetamol",
        "aliases": ["dolo", "dolo650", "dolo 650", "dolo-650"],
        "strength": "650mg", "strength_value": 650, "strength_unit": "mg",
        "dosage_form": "tablet",
        "source": "legacy_manual",
    })


def _seed_crocin(db):
    repo.upsert_medicine(db, {
        "external_id": "legacy:crocin-650",
        "brand_name": "Crocin 650",
        "generic_name": "Paracetamol",
        "aliases": ["crocin", "crocin650", "crocin 650", "crocin-650"],
        "strength": "650mg", "strength_value": 650, "strength_unit": "mg",
        "dosage_form": "tablet",
        "source": "legacy_manual",
    })


def _seed_calpol_500_and_syrup(db):
    repo.upsert_medicine(db, {
        "external_id": "legacy:calpol-500",
        "brand_name": "Calpol 500", "generic_name": "Paracetamol",
        "aliases": ["calpol", "calpol500", "calpol 500"],
        "strength": "500mg", "strength_value": 500, "strength_unit": "mg",
        "dosage_form": "tablet", "source": "legacy_manual",
    })
    repo.upsert_medicine(db, {
        "external_id": "legacy:calpol-syrup",
        "brand_name": "Calpol Syrup", "generic_name": "Paracetamol",
        "aliases": ["calpol syrup"],
        "strength": "250mg/5ml", "strength_value": 250, "strength_unit": "mg",
        "dosage_form": "syrup", "source": "legacy_manual",
    })


def _seed_meftal_p(db):
    repo.upsert_medicine(db, {
        "external_id": "legacy:meftal-p",
        "brand_name": "Meftal-P", "generic_name": "Mefenamic Acid + Paracetamol",
        "aliases": ["meftal p", "meftalp", "meftal-p"],
        "strength": "250mg + 325mg",
        "combination_components": [
            {"strength": 250, "unit": "mg", "name": "Mefenamic Acid"},
            {"strength": 325, "unit": "mg", "name": "Paracetamol"},
        ],
        "dosage_form": "tablet", "source": "legacy_manual",
    })


# -- Exact matching (section 17) --

def test_exact_brand_match(db):
    _seed_dolo(db)
    db.commit()
    r = resolve(db, "Dolo 650")
    assert r.status == MatchStatus.EXACT
    assert r.matched.medicine.brand_name == "Dolo 650"
    assert r.source == "legacy_manual"


def test_exact_alias_match_combiflam_meftal(db):
    _seed_meftal_p(db)
    db.commit()
    r = resolve(db, "Meftal-P")
    assert r.status == MatchStatus.EXACT
    assert r.matched.medicine.brand_name == "Meftal-P"
    assert r.matched.medicine.combination_components == [
        {"strength": 250, "unit": "mg", "name": "Mefenamic Acid"},
        {"strength": 325, "unit": "mg", "name": "Paracetamol"},
    ]


# -- OCR variants (section 17) --

def test_ocr_variant_zero_for_o(db):
    _seed_dolo(db)
    db.commit()
    r = resolve(db, "D0LO 650")
    assert r.status in (MatchStatus.HIGH_CONFIDENCE, MatchStatus.EXACT)
    assert r.matched.medicine.brand_name == "Dolo 650"


def test_ocr_variant_hyphenated_and_concatenated(db):
    _seed_dolo(db)
    db.commit()
    for text in ("DOLO-650", "Dolo650"):
        r = resolve(db, text)
        assert r.status == MatchStatus.EXACT, text
        assert r.matched.medicine.brand_name == "Dolo 650", text


def test_ocr_variant_letter_for_letter(db):
    _seed_crocin(db)
    db.commit()
    r = resolve(db, "Crocln 650")  # l instead of i
    assert r.status == MatchStatus.HIGH_CONFIDENCE
    assert r.matched.medicine.brand_name == "Crocin 650"


# -- Strength parsing integration (section 17) --

def test_generic_plus_strength_disambiguates_uniquely(db):
    _seed_dolo(db)
    _seed_calpol_500_and_syrup(db)
    db.commit()
    r = resolve(db, "Paracetamol 500mg tablet")
    assert r.status == MatchStatus.EXACT
    assert r.matched.medicine.brand_name == "Calpol 500"  # the ONLY 500mg tablet


def test_strength_mismatch_never_silently_substitutes(db):
    # THE safety-rule example from the spec: "Dolo 680" must never silently
    # become "Dolo 650".
    _seed_dolo(db)
    db.commit()
    r = resolve(db, "Dolo 680mg")
    assert r.status != MatchStatus.EXACT
    assert r.matched is None
    if r.candidates:
        assert all(c.medicine.brand_name != "Dolo 650" or r.status != MatchStatus.EXACT for c in r.candidates)


# -- Dosage forms (section 17) --

def test_dosage_form_mismatch_prevents_false_match(db):
    _seed_calpol_500_and_syrup(db)
    db.commit()
    # Explicit tablet + 500mg must resolve to the tablet, not the syrup,
    # even though both are "Paracetamol".
    r = resolve(db, "Paracetamol 500mg tablet")
    assert r.matched.medicine.dosage_form == "tablet"

    # Asking for a strength/form combo that doesn't exist (500mg syrup) must
    # not fabricate a match against the 250mg syrup or the 500mg tablet.
    r2 = resolve(db, "Paracetamol 500mg syrup")
    assert r2.status != MatchStatus.EXACT


# -- Ambiguous candidates (section 17) --

def test_ambiguous_generic_without_strength_is_review_required(db):
    _seed_dolo(db)
    _seed_crocin(db)
    _seed_calpol_500_and_syrup(db)
    db.commit()
    r = resolve(db, "Paracetamol")  # no strength/form given -- 4 products exist
    assert r.status == MatchStatus.REVIEW_REQUIRED
    assert r.matched is None
    assert len(r.candidates) >= 2  # never forces a pick among genuinely tied options


# -- No-match (section 17) --

def test_unknown_medicine_is_no_match_not_fabricated(db):
    _seed_dolo(db)
    db.commit()
    r = resolve(db, "Xyzcompletelyfakemedicinename")
    assert r.status == MatchStatus.NO_MATCH
    assert r.matched is None
    assert r.candidates == []


def test_empty_input_is_no_match(db):
    r = resolve(db, "")
    assert r.status == MatchStatus.NO_MATCH
    r2 = resolve(db, "   ")
    assert r2.status == MatchStatus.NO_MATCH


# -- Null safety (section 17 / preserved from earlier fixes) --

def test_resolver_survives_null_name_catalog_row(db):
    repo.upsert_medicine(db, {
        "external_id": "bad:row",
        "generic_name": None,
        "brand_name": None,
        "aliases": ["weirdmed"],
        "source": "legacy_manual",
    })
    db.commit()
    r = resolve(db, "Weirdmed")
    assert r is not None  # must not raise


def test_resolver_survives_null_aliases(db):
    med = repo.upsert_medicine(db, {
        "external_id": "bad:aliases-none",
        "generic_name": "Testmed",
        "aliases": None,
        "source": "legacy_manual",
    })
    db.commit()
    assert med.aliases == []
    r = resolve(db, "Testmed")
    assert r.status == MatchStatus.EXACT


# -- RxNorm stays supplementary (section 17 / 13) --

def test_rxnorm_never_overrides_indian_brand_record(db):
    # Same generic name from both sources -- legacy_manual must always win
    # exact-match resolution, deterministically, regardless of insert order.
    repo.upsert_medicine(db, {"external_id": "rxnorm:723", "generic_name": "Amoxicillin", "source": "rxnorm"})
    repo.upsert_medicine(db, {"external_id": "legacy:amoxicillin", "generic_name": "Amoxicillin", "source": "legacy_manual"})
    db.commit()

    for _ in range(5):
        r = resolve(db, "Amoxicillin")
        assert r.status == MatchStatus.EXACT
        assert r.source == "legacy_manual"


def test_rxnorm_fills_gaps_legacy_manual_does_not_cover(db):
    # A generic RxNorm-only ingredient (no Indian brand curated for it here)
    # must still resolve -- RxNorm is supplementary, not excluded.
    repo.upsert_medicine(db, {"external_id": "rxnorm:6135", "generic_name": "Metoprolol", "source": "rxnorm"})
    db.commit()
    r = resolve(db, "Metoprolol")
    assert r.status == MatchStatus.EXACT
    assert r.source == "rxnorm"


def test_different_generic_naming_is_not_a_duplicate_error(db):
    # Dolo 650 resolves through legacy_manual -> paracetamol, while a plain
    # "Paracetamol" RxNorm-style query for the US name should not error out
    # just because the two sources spell the identity differently.
    _seed_dolo(db)
    repo.upsert_medicine(db, {"external_id": "rxnorm:161", "generic_name": "Acetaminophen", "source": "rxnorm"})
    db.commit()

    r1 = resolve(db, "Dolo 650")
    assert r1.status == MatchStatus.EXACT
    assert r1.matched.medicine.generic_name == "Paracetamol"

    r2 = resolve(db, "Acetaminophen")
    assert r2.status == MatchStatus.EXACT
    assert r2.source == "rxnorm"


# -- Auditability (section 12) --

def test_result_carries_full_audit_trail(db):
    _seed_dolo(db)
    db.commit()
    r = resolve(db, "DOLO-650")
    assert r.input_text == "DOLO-650"
    assert r.normalized_text == "dolo 650"
    assert r.matched is not None
    assert r.match_type is not None
    assert r.confidence is not None
    assert r.source is not None
    assert r.latency_ms >= 0
