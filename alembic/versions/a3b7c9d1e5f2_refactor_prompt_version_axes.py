"""refactor exec_mode to context_source and output_format

Revision ID: a3b7c9d1e5f2
Revises: 6e2e4d9c0fd6
Create Date: 2026-02-21

Replaces the monolithic exec_mode/retrieval_mode columns with orthogonal
context_source × output_format axes + search_strategy + temperature.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a3b7c9d1e5f2'
down_revision: Union[str, Sequence[str]] = '6e2e4d9c0fd6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns with safe defaults
    op.add_column("prompt_versions", sa.Column("context_source", sa.String, server_default="documents"))
    op.add_column("prompt_versions", sa.Column("output_format", sa.String, server_default="text"))
    op.add_column("prompt_versions", sa.Column("search_strategy", sa.JSON, server_default='["semantic"]'))
    op.add_column("prompt_versions", sa.Column("temperature", sa.Float, server_default="0.1"))

    # Migrate existing data from old columns to new ones
    op.execute("""
        UPDATE prompt_versions SET
            context_source = CASE exec_mode
                WHEN 'rag' THEN 'documents'
                WHEN 'structured' THEN 'documents'
                WHEN 'direct' THEN 'none'
                WHEN 'sql' THEN 'metrics_db'
                ELSE 'documents'
            END,
            output_format = CASE exec_mode
                WHEN 'rag' THEN 'text'
                WHEN 'structured' THEN 'json'
                WHEN 'direct' THEN 'text'
                WHEN 'sql' THEN 'table'
                ELSE 'text'
            END,
            search_strategy = CASE retrieval_mode
                WHEN 'quantitative' THEN '["numeric"]'::jsonb
                ELSE '["semantic"]'::jsonb
            END
    """)

    # Drop old columns
    op.drop_column("prompt_versions", "exec_mode")
    op.drop_column("prompt_versions", "retrieval_mode")


def downgrade() -> None:
    # Re-add old columns
    op.add_column("prompt_versions", sa.Column("exec_mode", sa.String, server_default="rag"))
    op.add_column("prompt_versions", sa.Column("retrieval_mode", sa.String, nullable=True))

    # Migrate back: context_source + output_format → exec_mode
    op.execute("""
        UPDATE prompt_versions SET
            exec_mode = CASE
                WHEN context_source = 'none' THEN 'direct'
                WHEN context_source = 'metrics_db' THEN 'sql'
                WHEN output_format = 'json' THEN 'structured'
                ELSE 'rag'
            END,
            retrieval_mode = CASE
                WHEN search_strategy::text LIKE '%numeric%' THEN 'quantitative'
                ELSE 'qualitative'
            END
    """)

    # Drop new columns
    op.drop_column("prompt_versions", "temperature")
    op.drop_column("prompt_versions", "search_strategy")
    op.drop_column("prompt_versions", "output_format")
    op.drop_column("prompt_versions", "context_source")
