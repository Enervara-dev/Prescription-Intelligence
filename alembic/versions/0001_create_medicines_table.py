"""create medicines table

Revision ID: 0001
Revises:
Create Date: 2026-09-15

Replaces the old medicine_list.txt + in-memory RapidFuzz scan with an
indexed Postgres catalog. Enables pg_trgm and builds a GIN trigram index over
a normalized search_text column (generic_name + brand_name + aliases), so
fuzzy lookups stay index-backed regardless of catalog size, instead of
scanning every row in Python.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "medicines",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("external_id", sa.String(length=128), nullable=True),
        sa.Column("generic_name", sa.Text(), nullable=True),
        sa.Column("brand_name", sa.Text(), nullable=True),
        sa.Column("aliases", postgresql.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("strength", sa.Text(), nullable=True),
        sa.Column("dosage_form", sa.Text(), nullable=True),
        sa.Column("route", sa.Text(), nullable=True),
        sa.Column("composition", sa.Text(), nullable=True),
        sa.Column("manufacturer", sa.Text(), nullable=True),
        sa.Column("atc_code", sa.Text(), nullable=True),
        sa.Column("snomed_ct_code", sa.Text(), nullable=True),
        sa.Column("rxcui", sa.Text(), nullable=True),
        sa.Column("codes", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_version", sa.Text(), nullable=True),
        sa.Column("source_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("source", "external_id", name="uq_medicines_source_external_id"),
    )

    op.create_index("ix_medicines_generic_name", "medicines", ["generic_name"])
    op.create_index("ix_medicines_brand_name", "medicines", ["brand_name"])

    # Trigram index powering fuzzy/typo-tolerant search over names+aliases.
    op.execute(
        "CREATE INDEX ix_medicines_search_text_trgm ON medicines USING GIN (search_text gin_trgm_ops)"
    )
    # GIN index for exact/containment lookups against the aliases array.
    op.execute("CREATE INDEX ix_medicines_aliases_gin ON medicines USING GIN (aliases)")


def downgrade() -> None:
    op.drop_index("ix_medicines_aliases_gin", table_name="medicines")
    op.drop_index("ix_medicines_search_text_trgm", table_name="medicines")
    op.drop_index("ix_medicines_brand_name", table_name="medicines")
    op.drop_index("ix_medicines_generic_name", table_name="medicines")
    op.drop_table("medicines")
    # Deliberately not dropping the pg_trgm extension — another table/DB user
    # may depend on it.
