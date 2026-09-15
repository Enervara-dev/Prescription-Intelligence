"""add indian medicine normalization fields

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-15

Additive, non-breaking: adds nullable columns + indexes for the Indian
medicine normalization layer (canonical_name, structured strength,
combination_components). Existing rows are unaffected; existing columns
(strength as raw text, generic_name, brand_name, etc.) are untouched.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("medicines", sa.Column("canonical_name", sa.Text(), nullable=True))
    op.add_column("medicines", sa.Column("strength_value", sa.Numeric(10, 3), nullable=True))
    op.add_column("medicines", sa.Column("strength_unit", sa.Text(), nullable=True))
    op.add_column(
        "medicines",
        sa.Column(
            "combination_components",
            postgresql.JSONB(),
            nullable=False,
            server_default="[]",
        ),
    )

    op.create_index("ix_medicines_canonical_name", "medicines", ["canonical_name"])
    op.create_index(
        "ix_medicines_canonical_strength_form",
        "medicines",
        ["canonical_name", "strength_value", "dosage_form"],
    )


def downgrade() -> None:
    op.drop_index("ix_medicines_canonical_strength_form", table_name="medicines")
    op.drop_index("ix_medicines_canonical_name", table_name="medicines")
    op.drop_column("medicines", "combination_components")
    op.drop_column("medicines", "strength_unit")
    op.drop_column("medicines", "strength_value")
    op.drop_column("medicines", "canonical_name")
