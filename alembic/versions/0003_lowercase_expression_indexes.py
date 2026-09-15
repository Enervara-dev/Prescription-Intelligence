"""fix exact-match indexes to be case-insensitive expression indexes

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-15

find_exact()/find_by_canonical_name() filter on lower(generic_name) = :term,
lower(brand_name) = :term, lower(canonical_name) = :term (case-insensitive
comparison). The plain btree indexes created in 0001/0002 are on the RAW
columns, which Postgres's planner does NOT use for a lower(col) predicate --
verified via EXPLAIN ANALYZE at ~15k rows: both queries fell back to a
sequential scan (~7ms) despite the "matching" index existing, the same class
of gap as the pg_trgm `%`-operator issue fixed earlier for search_candidates().

Replaces them with expression indexes on lower(...) specifically, which the
planner DOES use for these exact queries.
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_medicines_generic_name", table_name="medicines")
    op.drop_index("ix_medicines_brand_name", table_name="medicines")
    op.drop_index("ix_medicines_canonical_name", table_name="medicines")
    op.drop_index("ix_medicines_canonical_strength_form", table_name="medicines")

    op.execute("CREATE INDEX ix_medicines_generic_name_lower ON medicines (lower(generic_name))")
    op.execute("CREATE INDEX ix_medicines_brand_name_lower ON medicines (lower(brand_name))")
    op.execute("CREATE INDEX ix_medicines_canonical_name_lower ON medicines (lower(canonical_name))")
    op.execute(
        "CREATE INDEX ix_medicines_canonical_strength_form_lower "
        "ON medicines (lower(canonical_name), strength_value, dosage_form)"
    )


def downgrade() -> None:
    op.drop_index("ix_medicines_canonical_strength_form_lower", table_name="medicines")
    op.drop_index("ix_medicines_canonical_name_lower", table_name="medicines")
    op.drop_index("ix_medicines_brand_name_lower", table_name="medicines")
    op.drop_index("ix_medicines_generic_name_lower", table_name="medicines")

    op.create_index("ix_medicines_generic_name", "medicines", ["generic_name"])
    op.create_index("ix_medicines_brand_name", "medicines", ["brand_name"])
    op.create_index("ix_medicines_canonical_name", "medicines", ["canonical_name"])
    op.create_index(
        "ix_medicines_canonical_strength_form",
        "medicines",
        ["canonical_name", "strength_value", "dosage_form"],
    )
