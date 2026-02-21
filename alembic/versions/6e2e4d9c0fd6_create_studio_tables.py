"""create studio tables

Revision ID: 6e2e4d9c0fd6
Revises:
Create Date: 2026-02-20 22:29:31.733497

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '6e2e4d9c0fd6'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Prompts (master) ──────────────────────────────────────────
    op.create_table(
        "prompts",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("category", sa.String, server_default="custom"),
        sa.Column("tags", sa.JSON, server_default="[]"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("is_archived", sa.Boolean, server_default=sa.text("false")),
    )

    # ── Prompt Versions (immutable detail) ────────────────────────
    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("prompt_id", sa.String, sa.ForeignKey("prompts.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("template", sa.Text, nullable=False),
        sa.Column("variables", sa.JSON, server_default="[]"),
        sa.Column("exec_mode", sa.String, server_default="rag"),
        sa.Column("output_schema", sa.JSON, nullable=True),
        sa.Column("model_id", sa.String, nullable=True),
        sa.Column("retrieval_mode", sa.String, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("is_published", sa.Boolean, server_default=sa.text("false")),
        sa.UniqueConstraint("prompt_id", "version", name="uq_prompt_version"),
    )

    # ── Workflows ─────────────────────────────────────────────────
    op.create_table(
        "workflows",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("input_vars", sa.JSON, server_default="[]"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("is_archived", sa.Boolean, server_default=sa.text("false")),
    )

    # ── Workflow Steps (normalized, FK-enforced) ──────────────────
    op.create_table(
        "workflow_steps",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("workflow_id", sa.String, sa.ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("prompt_version_id", sa.String, sa.ForeignKey("prompt_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("step_order", sa.Integer, nullable=False),
        sa.Column("label", sa.String, nullable=False),
        sa.Column("input_mapping", sa.JSON, server_default="{}"),
        sa.Column("output_key", sa.String, nullable=False),
        sa.Column("condition", sa.Text, nullable=True),
        sa.Column("retry_count", sa.Integer, server_default="1"),
    )

    # ── Workflow Runs (execution history + checkpointing) ─────────
    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("workflow_id", sa.String, sa.ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("status", sa.String, server_default="running"),
        sa.Column("inputs", sa.JSON, server_default="{}"),
        sa.Column("step_outputs", sa.JSON, server_default="{}"),
        sa.Column("current_step", sa.Integer, server_default="0"),
        sa.Column("started_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
    )

    # ── Agent Tools ───────────────────────────────────────────────
    op.create_table(
        "agent_tools",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("icon", sa.String, server_default="wrench"),
        sa.Column("workflow_id", sa.String, sa.ForeignKey("workflows.id", ondelete="SET NULL"), nullable=True),
        sa.Column("prompt_version_id", sa.String, sa.ForeignKey("prompt_versions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("input_schema", sa.JSON, server_default="[]"),
        sa.Column("output_format", sa.String, server_default="text"),
        sa.Column("is_published", sa.Boolean, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("agent_tools")
    op.drop_table("workflow_runs")
    op.drop_table("workflow_steps")
    op.drop_table("workflows")
    op.drop_table("prompt_versions")
    op.drop_table("prompts")
