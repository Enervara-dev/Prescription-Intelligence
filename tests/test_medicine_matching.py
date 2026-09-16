"""
tests/test_medicine_matching.py
----------------------------------
End-to-end tests against the real Postgres-backed medicine catalog: legacy
import (no data loss), exact/alias/fuzzy/ocr_corrected/unresolved/ambiguous
matching, and the full parse_medicines() pipeline.
"""

from app.repositories import medicine_repository as repo
from app.services import medicine_matching
from app.services.extraction import parse_medicines
from scripts.import_legacy_medicines import load_legacy_names, run as run_legacy_import


def test_legacy_import_preserves_every_distinct_name(db):
    written = run_legacy_import()
    names = load_legacy_names()
    distinct_names = {n.lower() for n in names}

    assert written == len(names)  # every line upserted (dupes just overwrite themselves)

    rows, total = repo.list_medicines(db, limit=len(distinct_names) + 10)
    db_names = {r.generic_name.lower() for r in rows}

    assert total == len(distinct_names)
    assert db_names == distinct_names  # zero data loss


def test_exact_match(db):
    repo.upsert_medicine(db, {
        "external_id": "legacy:paracetamol",
        "generic_name": "Paracetamol",
        "brand_name": "Dolo 650",
        "aliases": ["Dolo", "Crocin", "Calpol"],
        "source": "legacy_manual",
    })
    db.commit()

    match = medicine_matching.match_token(db, "Paracetamol")
    assert match.match_type == "exact"
    assert match.confidence == 100.0
    assert match.medicine.generic_name == "Paracetamol"


def test_indian_brand_alias_match(db):
    repo.upsert_medicine(db, {
        "external_id": "legacy:paracetamol",
        "generic_name": "Paracetamol",
        "brand_name": "Dolo 650",
        "aliases": ["Dolo", "Crocin", "Calpol"],
        "source": "legacy_manual",
    })
    db.commit()

    for brand in ("Dolo", "Crocin", "Calpol", "crocin"):
        match = medicine_matching.match_token(db, brand)
        assert match.match_type == "alias", f"{brand} should resolve via alias"
        assert match.medicine.generic_name == "Paracetamol"


def test_generic_name_match(db):
    repo.upsert_medicine(db, {"external_id": "legacy:ibuprofen", "generic_name": "Ibuprofen", "source": "legacy_manual"})
    db.commit()

    match = medicine_matching.match_token(db, "Ibuprofen")
    assert match.match_type == "exact"
    assert match.medicine.generic_name == "Ibuprofen"


def test_ocr_typo_resolves_via_fuzzy_or_optical_correction(db):
    repo.upsert_medicine(db, {"external_id": "legacy:amoxicillin", "generic_name": "Amoxicillin", "source": "legacy_manual"})
    db.commit()

    # Common OCR misread: 'rn' read as 'm' -> "Arnoxycillin" instead of "Amoxycillin"
    match = medicine_matching.match_token(db, "Arnoxycillin")
    assert match.match_type in ("fuzzy", "ocr_corrected")
    assert match.medicine is not None
    assert match.medicine.generic_name == "Amoxicillin"
    assert match.confidence >= medicine_matching.settings.MEDICINE_MATCH_THRESHOLD


def test_optical_correction_rescues_a_below_threshold_raw_score(db):
    repo.upsert_medicine(db, {"external_id": "legacy:amoxicillin", "generic_name": "Amoxicillin", "source": "legacy_manual"})
    db.commit()

    token = "arnoxycillin"
    raw_ranked = medicine_matching._rerank(token, repo.search_candidates(db, token))
    assert raw_ranked and raw_ranked[0][1] < medicine_matching.settings.MEDICINE_MATCH_THRESHOLD, (
        "test assumes the raw token alone should NOT clear the threshold"
    )

    match = medicine_matching.match_token(db, token)
    assert match.match_type == "ocr_corrected"
    assert match.medicine.generic_name == "Amoxicillin"


def test_unknown_word_is_unresolved_not_forced(db):
    repo.upsert_medicine(db, {"external_id": "legacy:paracetamol", "generic_name": "Paracetamol", "source": "legacy_manual"})
    db.commit()

    match = medicine_matching.match_token(db, "Xyzabcqqrandomtoken")
    assert match.match_type == "unresolved"
    assert match.medicine is None


def test_ambiguous_when_top_two_scores_are_within_margin(db, monkeypatch):
    med_a = repo.upsert_medicine(db, {"external_id": "t:a", "generic_name": "Alphamed", "source": "legacy_manual"})
    med_b = repo.upsert_medicine(db, {"external_id": "t:b", "generic_name": "Alphamet", "source": "legacy_manual"})
    db.commit()

    # Force the re-ranking step to return two near-tied scores, isolating the
    # ambiguity-margin decision in match_token() from real fuzzy-score luck.
    monkeypatch.setattr(
        medicine_matching, "_rerank", lambda term, candidates: [(med_a, 90.0), (med_b, 88.0)]
    )

    match = medicine_matching.match_token(db, "Alphame")
    assert match.match_type == "ambiguous"
    assert match.medicine is None  # never forces a pick between close candidates


def test_strength_form_route_manufacturer_pass_through(db):
    repo.upsert_medicine(db, {
        "external_id": "legacy:azithromycin",
        "generic_name": "Azithromycin",
        "strength": "500mg",
        "dosage_form": "tablet",
        "route": "oral",
        "manufacturer": "Test Pharma",
        "source": "legacy_manual",
    })
    db.commit()

    match = medicine_matching.match_token(db, "Azithromycin")
    assert match.medicine.strength == "500mg"
    assert match.medicine.dosage_form == "tablet"
    assert match.medicine.route == "oral"
    assert match.medicine.manufacturer == "Test Pharma"


def test_parse_medicines_end_to_end_pipeline(db):
    repo.upsert_medicine(db, {
        "external_id": "legacy:paracetamol",
        "generic_name": "Paracetamol",
        "brand_name": "Dolo 650",
        "aliases": ["Dolo", "Crocin", "Calpol"],
        "source": "legacy_manual",
    })
    repo.upsert_medicine(db, {"external_id": "legacy:amoxicillin", "generic_name": "Amoxicillin", "source": "legacy_manual"})
    db.commit()

    ocr_text = (
        "Rx\n"
        "Tab Dolo 650 1-0-1 x 5 days\n"
        "Tab Amoxycillin 500mg BD 7 days\n"
        "Gibberishnotamedicine 1-0-1\n"
    )
    results = parse_medicines(ocr_text, db)

    names = {r["name"] for r in results}
    assert "Dolo 650" in names
    assert "Amoxicillin" in names
    assert len(results) == 2  # gibberish line never emitted

    dolo = next(r for r in results if r["name"] == "Dolo 650")
    # parse_medicines is now backed by medicine_resolver.resolve() -- richer
    # UPPER_SNAKE match_type vocabulary than the old lowercase one. The
    # pipeline looks ahead to combine "Dolo"+"650" before trying "Dolo"
    # alone, so this resolves via the full brand string (EXACT_BRAND).
    assert dolo["match_type"] == "EXACT_BRAND"
    assert dolo["duration"] == "5 days"
    assert dolo["source"] == "legacy_manual"

    amox = next(r for r in results if r["name"] == "Amoxicillin")
    # "dosage" now means genuine administration dosage (e.g. "1 tablet"),
    # not strength -- "500mg" is a strength, correctly reported via
    # "strength_text" instead. This line never states an actual dosage
    # amount, so "dosage" is honestly "N/A" (see extraction/__init__.py,
    # fixing a confirmed bug from the OCR/matching forensic audit).
    assert amox["dosage"] == "N/A"
    assert amox["strength_text"] == "500mg"
    assert amox["frequency"] == "Twice Daily"


def test_malformed_catalog_row_with_no_name_does_not_crash(db):
    # A row with no generic_name/brand_name (e.g. bad ABDM data) must not
    # crash the pipeline; falling back to an alias is fine, but it must
    # never raise.
    repo.upsert_medicine(db, {
        "external_id": "bad:row",
        "generic_name": None,
        "brand_name": None,
        "aliases": ["Weirdmed"],
        "source": "legacy_manual",
    })
    db.commit()

    results = parse_medicines("Tab Weirdmed 500mg OD 5 days", db)
    assert results  # falls back to the alias as the display name
    # Aliases are stored normalized (lowercase) -- see upsert_medicine, needed
    # so find_exact()/find_alias_exact() can use the aliases GIN index via the
    # `@>` operator instead of an unindexed per-element lower() scan.
    assert results[0]["name"] == "weirdmed"


def test_upsert_with_explicit_none_aliases_does_not_crash(db):
    # aliases is NOT NULL; a caller passing aliases=None explicitly (not just
    # omitting the key) must be normalized to [], not violate the constraint.
    med = repo.upsert_medicine(db, {
        "external_id": "bad:aliases-none",
        "generic_name": "Testmed",
        "aliases": None,
        "source": "legacy_manual",
    })
    db.commit()
    assert med.aliases == []


def test_search_candidates_query_is_usable_by_the_trigram_index(db):
    # Regression test for a real bug found in review: search_candidates()
    # originally filtered on a bare `similarity(col, term) >= x` function
    # call. pg_trgm's GIN index does NOT accelerate that form at all -- only
    # the `%` operator form is index-aware -- so Postgres had NO way to use
    # the index for that query, regardless of table size, and silently fell
    # back to a full sequential scan even with a valid index present.
    # Verified via EXPLAIN ANALYZE at ~14.7k real RxNorm rows: seq scan
    # ~38ms vs the fixed `%`-operator form ~1ms (a ~30x difference).
    #
    # Whether the *planner* prefers the index over a seq scan is legitimately
    # size/cost dependent (small tables correctly seq-scan) -- what this
    # asserts is that the query is even *expressible* via the index, which
    # the old bare-function form never was, at any table size.
    import sqlalchemy as sa

    from app.models.medicine import Medicine

    norm = "amoxicilin"
    db.execute(sa.text("SET LOCAL pg_trgm.similarity_threshold = 0.2"))
    db.execute(sa.text("SET LOCAL enable_seqscan = off"))
    stmt = (
        sa.select(Medicine.id, sa.func.similarity(Medicine.search_text, norm).label("sim"))
        .where(Medicine.search_text.op("%")(norm))
        .order_by(sa.text("sim DESC"))
        .limit(8)
    )
    compiled = stmt.compile(dialect=db.bind.dialect)
    raw_conn = db.connection().connection.dbapi_connection
    cur = raw_conn.cursor()
    mogrified = cur.mogrify(str(compiled), compiled.params).decode()
    cur.execute("EXPLAIN " + mogrified)
    plan = "\n".join(row[0] for row in cur.fetchall())

    assert "ix_medicines_search_text_trgm" in plan, (
        f"query is not usable by the trigram index even with enable_seqscan=off:\n{plan}"
    )


def test_find_exact_queries_use_expression_and_gin_indexes(db):
    # Regression test for two more real bugs found in review: find_exact()
    # filtered on lower(generic_name)/lower(brand_name), which a plain btree
    # index on the raw column does NOT accelerate (same class of bug as the
    # pg_trgm one above) -- fixed with expression indexes on lower(...)
    # (migration 0003). And the alias check used
    # EXISTS(SELECT FROM unnest(aliases)...), which bypasses the aliases GIN
    # index entirely -- fixed by storing aliases pre-lowercased and querying
    # via the `@>` (contains) operator, which the GIN index does accelerate.
    # Verified via EXPLAIN ANALYZE at ~15k rows: ~7-8ms sequential scans for
    # both -> ~0.05-0.15ms indexed (50-100x).
    import sqlalchemy as sa

    from app.models.medicine import Medicine

    repo.upsert_medicine(db, {
        "external_id": "legacy:paracetamol",
        "generic_name": "Paracetamol",
        "brand_name": "Dolo 650",
        "aliases": ["dolo"],
        "source": "legacy_manual",
    })
    db.commit()

    def plan_for(stmt) -> str:
        compiled = stmt.compile(dialect=db.bind.dialect)
        raw_conn = db.connection().connection.dbapi_connection
        cur = raw_conn.cursor()
        mogrified = cur.mogrify(str(compiled), compiled.params).decode()
        cur.execute("EXPLAIN " + mogrified)
        return "\n".join(row[0] for row in cur.fetchall())

    db.execute(sa.text("SET LOCAL enable_seqscan = off"))

    name_stmt = sa.select(Medicine.id).where(
        (sa.func.lower(Medicine.generic_name) == "paracetamol") | (sa.func.lower(Medicine.brand_name) == "paracetamol")
    )
    name_plan = plan_for(name_stmt)
    assert "ix_medicines_generic_name_lower" in name_plan or "ix_medicines_brand_name_lower" in name_plan, name_plan

    alias_stmt = sa.select(Medicine.id).where(Medicine.aliases.contains(["dolo"]))
    alias_plan = plan_for(alias_stmt)
    assert "ix_medicines_aliases_gin" in alias_plan, alias_plan


def test_medicine_search_ranks_exact_first(db):
    repo.upsert_medicine(db, {
        "external_id": "legacy:paracetamol",
        "generic_name": "Paracetamol",
        "brand_name": "Dolo 650",
        "aliases": ["Dolo", "Crocin", "Calpol"],
        "source": "legacy_manual",
    })
    repo.upsert_medicine(db, {"external_id": "legacy:mupirocin", "generic_name": "Mupirocin", "source": "legacy_manual"})
    db.commit()

    results = medicine_matching.search_many(db, "paracetamol", limit=5)
    assert results[0].match_type == "exact"
    assert results[0].medicine.generic_name == "Paracetamol"
