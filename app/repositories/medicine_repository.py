"""
repositories/medicine_repository.py
-------------------------------------
All Postgres access for the medicine catalog. This is the ONLY place that
knows how `medicines` is queried — callers (matching logic, API endpoints,
sync jobs) never write raw SQL or scan rows themselves.

Retrieval is index-backed:
  - `find_exact()` hits the btree indexes on generic_name/brand_name and the
    GIN index on aliases.
  - `search_candidates()` hits the pg_trgm GIN index on `search_text` via the
    `%` similarity operator, so it stays fast as the catalog grows past the
    323 legacy rows — no linear Python scan.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import case as sa_case
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.medicine import DEFAULT_SOURCE_PRIORITY, SOURCE_PRIORITY, Medicine, build_search_text
from app.services.text_normalization import normalize_text


def _source_priority_order():
    """
    CASE expression ranking rows by source reliability (legacy_manual before
    abdm before rxnorm before anything else) — see SOURCE_PRIORITY. Used
    wherever more than one row could exactly match the same term: without an
    explicit order, `LIMIT 1` returns whichever row Postgres happens to scan
    first, which is a real, observed bug (e.g. "Amoxicillin" exists in both
    `legacy_manual` and `rxnorm`) — RxNorm is a supplementary layer and must
    never unpredictably shadow an Indian-catalog row.
    """
    whens = [(Medicine.source == src, prio) for src, prio in SOURCE_PRIORITY.items()]
    return sa_case(*whens, else_=DEFAULT_SOURCE_PRIORITY)


def _completeness_order():
    """
    Secondary tie-break, after source priority: prefer rows with more
    structured fields populated. Needed because the legacy flat catalog
    mixes bare, unstructured entries (e.g. medicine_list.txt's orphan "Dolo"
    line — just a generic_name, nothing else) with properly-modeled ones
    (e.g. the Indian-brands seed's "Dolo 650": brand_name, strength_value,
    aliases all populated) at the SAME source priority. Without this,
    find_exact()'s `LIMIT 1` can non-deterministically return the bare stub
    over an alias hit on the well-modeled row — observed for real: "Dolo
    680mg" resolved to the bare "Dolo" stub, silently discarding both the
    brand identity and the stated strength.
    """
    return (
        sa_case((Medicine.brand_name.is_(None), 1), else_=0)
        + sa_case((Medicine.strength_value.is_(None), 1), else_=0)
        + sa_case((func.coalesce(func.array_length(Medicine.aliases, 1), 0) == 0, 1), else_=0)
    )


@dataclass
class Candidate:
    medicine: Medicine
    similarity: float  # pg_trgm similarity, 0.0-1.0


def _normalize(text_: str) -> str:
    return " ".join(text_.strip().lower().split())


def find_exact(db: Session, term: str) -> Candidate | None:
    """
    Case-insensitive exact match against generic_name, brand_name, or any
    alias. Returns None if nothing matches exactly.
    """
    norm = _normalize(term)
    if not norm:
        return None

    stmt = (
        select(Medicine)
        .where(
            (func.lower(Medicine.generic_name) == norm)
            | (func.lower(Medicine.brand_name) == norm)
            | (Medicine.aliases.contains([norm]))
        )
        .order_by(_source_priority_order(), _completeness_order())
        .limit(1)
    )

    row = db.execute(stmt).scalars().first()
    return Candidate(medicine=row, similarity=1.0) if row else None


def find_alias_exact(db: Session, term: str) -> Candidate | None:
    """Exact match against aliases only (used to distinguish match_type='alias')."""
    norm = _normalize(term)
    if not norm:
        return None
    stmt = (
        select(Medicine)
        .where(Medicine.aliases.contains([norm]))
        .order_by(_source_priority_order(), _completeness_order())
        .limit(1)
    )
    row = db.execute(stmt).scalars().first()
    return Candidate(medicine=row, similarity=1.0) if row else None


def find_by_canonical_name(db: Session, canonical_name: str) -> list[Medicine]:
    """
    All rows sharing one generic identity (canonical_name), source-priority
    ordered — e.g. every strength/form of "paracetamol". Powers the
    "generic + strength" resolution stage: narrows via the
    ix_medicines_canonical_strength_form btree index before any fuzzy work.
    """
    norm = normalize_text(canonical_name)
    if not norm:
        return []
    stmt = (
        select(Medicine)
        .where(func.lower(Medicine.canonical_name) == norm)
        .order_by(_source_priority_order(), _completeness_order())
    )
    return list(db.execute(stmt).scalars().all())


def search_candidates(
    db: Session,
    term: str,
    limit: int | None = None,
    min_similarity: float | None = None,
) -> list[Candidate]:
    """
    Indexed fuzzy search: uses the pg_trgm GIN index on `search_text` to pull
    the top-N most similar rows for `term`, ordered by similarity desc.
    Cheap regardless of catalog size — this is what replaces the old
    "RapidFuzz over every row in a Python list" approach at the retrieval
    stage. Callers may still re-rank the (small) result with RapidFuzz for
    OCR-specific scoring.

    Uses the `%` operator (not a bare `similarity(...) >= x` predicate) in
    the WHERE clause: pg_trgm's GIN index only accelerates the operator form
    (`%`, `<->`, etc.) — a plain `similarity()` function call in WHERE is
    invisible to the planner as an index condition and silently falls back
    to a full sequential scan, even with the index present and valid.
    Verified via EXPLAIN ANALYZE: the function-call form scanned all rows
    (~38ms at ~14.7k rows); the `%` form uses a Bitmap Index Scan (~1ms).
    """
    norm = _normalize(term)
    if not norm:
        return []

    limit = limit or settings.MEDICINE_CANDIDATE_LIMIT
    min_similarity = min_similarity if min_similarity is not None else settings.MEDICINE_MIN_TRIGRAM_SIMILARITY

    # Transaction-scoped (resets automatically at commit/rollback / connection
    # checkin) — never leaks the threshold to another request on a pooled
    # connection.
    db.execute(text("SET LOCAL pg_trgm.similarity_threshold = :t"), {"t": min_similarity})

    stmt = (
        select(Medicine, func.similarity(Medicine.search_text, norm).label("sim"))
        .where(Medicine.search_text.op("%")(norm))
        .order_by(text("sim DESC"))
        .limit(limit)
    )
    rows = db.execute(stmt).all()
    return [Candidate(medicine=row[0], similarity=float(row[1])) for row in rows]


def list_medicines(db: Session, limit: int = 50, offset: int = 0) -> tuple[list[Medicine], int]:
    total = db.execute(select(func.count()).select_from(Medicine)).scalar_one()
    rows = (
        db.execute(
            select(Medicine)
            .order_by(func.coalesce(Medicine.generic_name, Medicine.brand_name))
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )
    return list(rows), total


def get_by_id(db: Session, medicine_id: uuid.UUID) -> Medicine | None:
    return db.get(Medicine, medicine_id)


def upsert_medicine(db: Session, data: dict, *, flush: bool = True) -> Medicine:
    """
    Insert or update a medicine, keyed on (source, external_id). Used by both
    the legacy import and the (future) ABDM sync — safe to re-run.
    """
    data = dict(data)
    # Normalize, not just default: a caller (e.g. a future ABDM normalizer)
    # may pass aliases=None explicitly, which `aliases` (NOT NULL) rejects.
    # Lowercased + deduped: find_exact()/find_alias_exact() match aliases via
    # the `@>` (contains) operator, which the GIN index accelerates ONLY for
    # an exact (case-sensitive) element match — storing aliases pre-
    # normalized means the index-backed query never needs a per-element
    # lower() call (which, like the pg_trgm/btree cases above, would make
    # the index unusable).
    raw_aliases = data.get("aliases") or []
    data["aliases"] = sorted({normalize_text(a) for a in raw_aliases if a and normalize_text(a)})
    data["combination_components"] = data.get("combination_components") or []
    data["search_text"] = build_search_text(data.get("generic_name"), data.get("brand_name"), data.get("aliases"))
    # Derived, like search_text — providers don't need to compute this
    # themselves. Prefers generic_name (the identity generic+strength
    # matching keys off) and falls back to brand_name only when no generic
    # name is known.
    if not data.get("canonical_name"):
        data["canonical_name"] = normalize_text(data.get("generic_name") or data.get("brand_name") or "") or None
    data.setdefault("id", uuid.uuid4())

    stmt = pg_insert(Medicine).values(**data)
    update_cols = {
        c: getattr(stmt.excluded, c)
        for c in data
        if c not in ("id", "source", "external_id", "created_at")
    }
    update_cols["updated_at"] = func.now()
    stmt = stmt.on_conflict_do_update(
        constraint="uq_medicines_source_external_id",
        set_=update_cols,
    ).returning(Medicine.id)

    result_id = db.execute(stmt).scalar_one()
    if flush:
        db.flush()
    result = db.get(Medicine, result_id)
    assert result is not None  # the row we just inserted/updated in this same transaction
    return result


def bulk_upsert(db: Session, records: list[dict]) -> int:
    """Upsert many records in one transaction. Returns the count written."""
    count = 0
    for record in records:
        upsert_medicine(db, record, flush=False)
        count += 1
    db.commit()
    if count:
        # Bulk writes (legacy import, ABDM/RxNorm sync) can shift row counts
        # enough to leave the planner's statistics stale, which is exactly
        # what causes it to skip the pg_trgm index in favour of a sequential
        # scan (verified via EXPLAIN ANALYZE). This is an out-of-band,
        # deliberately-triggered path — never called per OCR request — so
        # the extra ANALYZE cost here is negligible against that risk.
        db.execute(text("ANALYZE medicines"))
        db.commit()
    return count
