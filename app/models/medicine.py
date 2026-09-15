"""
models/medicine.py
--------------------
The local medicine terminology catalog. Replaces the old flat
`medicine_list.txt` + in-memory RapidFuzz-over-a-Python-list approach with an
indexed Postgres table (pg_trgm) that can hold real structured terminology
(brand/generic names, strength, dosage form, standard codes) from multiple
sources — hand-curated legacy data today, ABDM Drug Registry once its sync is
wired in.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base_class import Base

# Source reliability ordering, lowest number = highest priority. Used to make
# exact-match lookups deterministic when more than one source has a row for
# the same name (e.g. "Amoxicillin" exists in both `legacy_manual` and
# `rxnorm`): the Indian manual catalog must win, RxNorm is supplementary and
# must never silently override it. Unlisted/future sources sort last.
SOURCE_PRIORITY: dict[str, int] = {
    "legacy_manual": 0,
    "abdm": 1,
    "licensed_india": 1,
    "rxnorm": 2,
}
DEFAULT_SOURCE_PRIORITY = 99


class Medicine(Base):
    __tablename__ = "medicines"
    __table_args__ = (
        # Lets legacy import / ABDM sync upsert idempotently: re-running either
        # never duplicates a row, it just refreshes it.
        UniqueConstraint("source", "external_id", name="uq_medicines_source_external_id"),
        # NOTE: exact-match lookups (find_exact, find_by_canonical_name) all
        # filter on lower(column) = :term for case-insensitive comparison. A
        # plain btree index on the raw column is NOT used by the planner for
        # a lower(col) predicate — same class of gap as the pg_trgm `%` vs
        # bare similarity() issue. These are expression indexes on lower(...)
        # specifically (created via raw SQL in the Alembic migration, since
        # they're on function results, not columns) so they're documented
        # here but declared in the migration, not as plain Index() below.
        # See migration 0003_lowercase_expression_indexes.py.
        #
        # The pg_trgm GIN index on search_text and the GIN index on aliases are
        # created directly in the Alembic migration (raw SQL), since plain
        # SQLAlchemy Index() doesn't express `gin_trgm_ops`.
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Identity within its source catalog, e.g. an ABDM drug code or
    # "legacy:<slug>" for rows migrated from medicine_list.txt. Combined with
    # `source`, this is what upserts key off.
    external_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Normalized identity used for generic+strength / brand+strength matching
    # (stages 3 & 5 of the resolution hierarchy — see medicine_resolver.py).
    # Distinct from generic_name/brand_name (display strings): this is the
    # deterministically-normalized form (see text_normalization.normalize_text).
    canonical_name: Mapped[str | None] = mapped_column(Text, nullable=True)

    generic_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    brand_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default="{}")

    # `strength` is the raw/display string (e.g. "500mg + 125mg" for a
    # combination) as originally given by the source. `strength_value` /
    # `strength_unit` are the DETERMINISTICALLY PARSED structured form for a
    # single-component strength (see strength_parser.py) — nullable because
    # combination products don't have one strength/unit pair; their per-
    # component breakdown lives in `combination_components` instead.
    strength: Mapped[str | None] = mapped_column(Text, nullable=True)
    strength_value: Mapped[Decimal | None] = mapped_column(Numeric(10, 3), nullable=True)
    strength_unit: Mapped[str | None] = mapped_column(Text, nullable=True)
    # [{"strength": 500, "unit": "mg", "name": null}, {"strength": 125, "unit": "mg", "name": null}]
    combination_components: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")

    dosage_form: Mapped[str | None] = mapped_column(Text, nullable=True)
    route: Mapped[str | None] = mapped_column(Text, nullable=True)
    composition: Mapped[str | None] = mapped_column(Text, nullable=True)
    manufacturer: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Standard terminology codes, kept as their own columns for the common
    # ones (indexable, joinable) plus a JSONB bucket for anything source-
    # specific we don't want to keep adding columns for.
    atc_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    snomed_ct_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    rxcui: Mapped[str | None] = mapped_column(Text, nullable=True)
    codes: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")

    # Provenance. "legacy_manual" for the migrated medicine_list.txt entries,
    # "abdm" once the Drug Registry sync is implemented and run.
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Precomputed "generic_name + brand_name + aliases" blob that the pg_trgm
    # GIN index is built on, so a single indexed similarity() search covers
    # all name fields at once instead of one index per column.
    search_text: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


def build_search_text(generic_name: str | None, brand_name: str | None, aliases: list[str] | None) -> str:
    """Normalize the name fields into the blob the trigram index is built on."""
    parts = [generic_name or "", brand_name or "", *(aliases or [])]
    return " ".join(p.strip() for p in parts if p and p.strip()).lower()
