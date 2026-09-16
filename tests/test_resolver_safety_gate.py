"""
tests/test_resolver_safety_gate.py
--------------------------------------
Regression tests for the identity-corroboration acceptance gate added to
medicine_resolver.py: a raw fuzzy/trigram string-similarity score is never,
on its own, sufficient evidence for HIGH_CONFIDENCE. These are deliberately
written against the REAL seeded catalog (not synthetic fixtures) so the
gate's generalization is exercised the same way it would be in production:
by evidence properties (exact-after-correction, strength/form agreement),
never by checking a specific medicine's name.

Mandatory regression (see forensic audit + hard safeguards):
    resolve("Asthalin") must NEVER return an accepted ("matched") medicine.
"""

import pytest

from app.repositories import medicine_repository as repo
from app.services.medicine_resolver import MatchStatus, resolve
from scripts.import_legacy_medicines import run as run_legacy_import
from scripts.seed_indian_brands import run as run_indian_brands_seed


@pytest.fixture()
def real_catalog(db):
    """The actual production catalog (legacy import + curated Indian brands),
    not a hand-picked synthetic fixture -- the safety gate must hold against
    real, unpredictable catalog contents, not just a scenario built to pass."""
    run_legacy_import()
    run_indian_brands_seed()
    db.commit()
    return db


# -- Mandatory regression: Asthalin must never become Aspirin -------------

def test_asthalin_never_becomes_accepted_aspirin(real_catalog):
    r = resolve(real_catalog, "Asthalin")
    assert r.matched is None
    assert r.status != MatchStatus.HIGH_CONFIDENCE
    assert r.status != MatchStatus.EXACT
    # A real, if uncertain, candidate is fine to surface for review -- an
    # ACCEPTED wrong medicine is what must never happen.
    if r.matched is not None:
        assert r.matched.medicine.generic_name != "Aspirin"


def test_asthalin_with_strength_still_not_accepted(real_catalog):
    # Adding a strength that doesn't correspond to any real catalog match
    # for this text must not manufacture corroboration out of nothing.
    r = resolve(real_catalog, "Asthalin 100mcg")
    assert r.matched is None


# -- Generalization: unrelated similarly-spelled real generics ------------
# These confirm the gate is a property of the EVIDENCE, not a name-specific
# rule -- exact matches on real, differently-spelled generics must resolve
# to THEMSELVES, never get hijacked by a fuzzy neighbor.

@pytest.mark.parametrize("name", ["Amlodipine", "Amoxicillin", "Metformin"])
def test_real_generics_resolve_via_exact_stage_not_fuzzy_lookalikes(real_catalog, name):
    r = resolve(real_catalog, name)
    # Either an unambiguous EXACT hit (only one product under that generic
    # identity) or a REVIEW_REQUIRED tie among real products sharing that
    # identity (e.g. bare "Amlodipine" vs the curated "Amlopres 5") -- both
    # are safe. What must NEVER happen is resolving to a DIFFERENT generic
    # identity than the one asked for.
    assert r.status in (MatchStatus.EXACT, MatchStatus.REVIEW_REQUIRED)
    if r.matched is not None:
        assert r.matched.medicine.generic_name == name
    else:
        assert all(c.medicine.generic_name == name for c in r.candidates)


# -- Fuzzy-only evidence is capped, corroborated fuzzy is not -------------

def test_pure_fuzzy_resemblance_alone_never_reaches_high_confidence(real_catalog):
    """
    Direct test of the acceptance-gate mechanism itself (not tied to any one
    medicine name): a candidate reached ONLY through the fuzzy stage, with
    no exact-after-correction basis and no strength/form corroboration from
    the query, must never be reported as HIGH_CONFIDENCE -- regardless of
    how high its raw score is. We force a high raw score via monkeypatching
    to isolate the gate from real-world score luck.
    """
    from app.services import medicine_resolver

    med = repo.upsert_medicine(real_catalog, {
        "external_id": "t:lookalike-target",
        "generic_name": "Zolpidem",
        "source": "legacy_manual",
    })
    real_catalog.commit()

    original_rerank = medicine_resolver._rerank

    def fake_rerank(term, candidates):
        # Force an artificially high score with no strength/form stated and
        # no exact-after-correction path available for this made-up query.
        return [(med, 99.0)]

    medicine_resolver._rerank = fake_rerank
    try:
        r = resolve(real_catalog, "Completelyunrelatedqueryword")
    finally:
        medicine_resolver._rerank = original_rerank

    assert r.status != MatchStatus.HIGH_CONFIDENCE
    assert r.matched is None


def test_fuzzy_with_strength_corroboration_can_reach_high_confidence(real_catalog):
    """The flip side: when the query states a strength that agrees with the
    top fuzzy candidate, that IS independent corroboration, and the gate
    must not block a genuinely well-evidenced match."""
    from app.services import medicine_resolver

    med = repo.upsert_medicine(real_catalog, {
        "external_id": "t:corroborated",
        "generic_name": "Somemedicinename",
        "strength": "250mg", "strength_value": 250, "strength_unit": "mg",
        "dosage_form": "tablet",
        "source": "legacy_manual",
    })
    real_catalog.commit()

    original_rerank = medicine_resolver._rerank

    def fake_rerank(term, candidates):
        return [(med, 90.0)]

    medicine_resolver._rerank = fake_rerank
    try:
        r = resolve(real_catalog, "Somemdicinename 250mg")
    finally:
        medicine_resolver._rerank = original_rerank

    assert r.status == MatchStatus.HIGH_CONFIDENCE
    assert r.matched is not None
    assert r.matched.medicine.id == med.id


# -- LOW_CONFIDENCE never asserts a medicine --------------------------------

def test_low_confidence_never_populates_matched(real_catalog):
    from app.services import medicine_resolver

    med = repo.upsert_medicine(real_catalog, {
        "external_id": "t:low-conf",
        "generic_name": "Somelowconfidencemed",
        "source": "legacy_manual",
    })
    real_catalog.commit()

    original_rerank = medicine_resolver._rerank

    def fake_rerank(term, candidates):
        return [(med, 65.0)]  # between LOW_CONFIDENCE and MATCH thresholds

    medicine_resolver._rerank = fake_rerank
    try:
        r = resolve(real_catalog, "Somequerytext")
    finally:
        medicine_resolver._rerank = original_rerank

    assert r.status == MatchStatus.LOW_CONFIDENCE
    assert r.matched is None
    assert len(r.candidates) >= 1  # still auditable/reviewable, just not asserted


# -- overall_confidence / ocr_confidence plumbing --------------------------

def test_overall_confidence_equals_confidence_when_ocr_confidence_absent(real_catalog):
    r = resolve(real_catalog, "Amoxicillin")
    assert r.ocr_confidence is None
    assert r.overall_confidence == r.confidence


def test_overall_confidence_dampens_but_never_inflates(real_catalog):
    r_high_ocr = resolve(real_catalog, "Amoxicillin", ocr_confidence=1.0)
    r_low_ocr = resolve(real_catalog, "Amoxicillin", ocr_confidence=0.2)
    assert r_high_ocr.overall_confidence == r_high_ocr.confidence  # perfect OCR: no dampening
    assert r_low_ocr.overall_confidence < r_low_ocr.confidence  # poor OCR: dampened down
    assert r_low_ocr.overall_confidence <= r_high_ocr.overall_confidence


# -- Brand/generic honesty for unclassified legacy rows --------------------

def test_unclassified_legacy_row_reports_honest_match_type(real_catalog):
    """
    A bare medicine_list.txt name with no curated brand/generic annotation
    (e.g. "Omez" -- a real brand, but the source data never says so) must
    not be reported as a confirmed EXACT_GENERIC identity it doesn't
    actually have.
    """
    r = resolve(real_catalog, "Omez")
    assert r.matched is not None
    assert r.matched.match_type == "EXACT_CATALOG_NAME"


def test_curated_brand_still_reports_real_brand_generic_split(real_catalog):
    """Curated rows (app/data/indian_brands.py) are unaffected -- they DO
    have an established brand/generic split and must keep reporting it."""
    r = resolve(real_catalog, "Dolo 650")
    assert r.matched is not None
    assert r.matched.match_type in ("EXACT_ALIAS", "EXACT_BRAND")
    assert r.matched.medicine.brand_name == "Dolo 650"
    assert r.matched.medicine.generic_name == "Paracetamol"
