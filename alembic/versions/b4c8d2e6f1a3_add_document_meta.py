"""add document_meta table

Revision ID: b4c8d2e6f1a3
Revises: a3b7c9d1e5f2
Create Date: 2026-02-21

Adds a separate document_meta table for semantic metadata,
decoupled from the document_ledger (ingestion state).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision: str = "b4c8d2e6f1a3"
down_revision: Union[str, None] = "a3b7c9d1e5f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "document_meta",
        sa.Column("doc_hash", sa.String, sa.ForeignKey("document_ledger.doc_hash", ondelete="CASCADE"),
                  primary_key=True, index=True),
        # Identity
        sa.Column("company_name", sa.String, nullable=True),
        sa.Column("document_type", sa.String, nullable=True),
        sa.Column("project_code", sa.String, nullable=True),
        # Temporal
        sa.Column("as_of_date", sa.Date, nullable=True),
        sa.Column("period_label", sa.String, nullable=True),
        sa.Column("publish_date", sa.Date, nullable=True),
        # Classification
        sa.Column("sector", sa.String, nullable=True),
        sa.Column("geography", sa.String, nullable=True),
        sa.Column("currency", sa.String, nullable=True),
        sa.Column("confidentiality", sa.String, nullable=True),
        sa.Column("language", sa.String, server_default="en"),
        # Physical
        sa.Column("page_count", sa.Integer, server_default="0"),
        # User-managed
        sa.Column("tags", sa.JSON, server_default="[]"),
        sa.Column("notes", sa.Text, nullable=True),
        # Provenance
        sa.Column("extraction_confidence", sa.Float, server_default="0.0"),
        sa.Column("user_overrides", sa.JSON, server_default="{}"),
        sa.Column("last_edited_at", sa.DateTime, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("document_meta")
