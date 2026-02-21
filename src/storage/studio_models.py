"""
Studio Models: Prompt Library, Workflows, and Agent Tools.

Uses a shared declarative Base that Alembic manages migrations for.
Existing tables (document_ledger, metric_facts) remain on create_all()
until they need schema changes.

Design decisions:
- prompt_versions are IMMUTABLE (append-only, publish-gated)
- workflow_steps are a NORMALIZED table (FK to prompt_versions, not JSON array)
- workflow_runs support HITL pause (status: paused_for_approval)
- All IDs are UUID strings for portability
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime, ForeignKey, JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _uuid() -> str:
    return str(uuid4())


# ==============================================================================
# PROMPT LIBRARY
# ==============================================================================

class Prompt(Base):
    """
    Master record for a saved prompt. Mutable fields: name, description, tags.
    The actual template content lives in prompt_versions (immutable).
    """
    __tablename__ = "prompts"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    category = Column(String, default="custom")  # extraction | analysis | comparison | summarization | custom
    tags = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_archived = Column(Boolean, default=False)

    # Relationship
    versions = relationship("PromptVersion", back_populates="prompt", order_by="PromptVersion.version")


class PromptVersion(Base):
    """
    Immutable snapshot of a prompt template. Created on each "Publish" action.
    Workflows pin to a specific prompt_version_id for reproducibility.
    """
    __tablename__ = "prompt_versions"

    id = Column(String, primary_key=True, default=_uuid)
    prompt_id = Column(String, ForeignKey("prompts.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(Integer, nullable=False)

    # Template content
    template = Column(Text, nullable=False)          # Jinja2 template with {{variables}}
    variables = Column(JSON, default=list)            # [{name, type, default, description}]

    # Execution config
    exec_mode = Column(String, default="rag")         # rag | structured | direct | sql
    output_schema = Column(JSON, nullable=True)       # Pydantic schema def (for structured mode)
    model_id = Column(String, nullable=True)          # Override model (null = system default)
    retrieval_mode = Column(String, nullable=True)    # qualitative | quantitative (for rag/structured)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    is_published = Column(Boolean, default=False)     # Draft vs published

    # Constraints
    __table_args__ = (
        UniqueConstraint("prompt_id", "version", name="uq_prompt_version"),
    )

    # Relationship
    prompt = relationship("Prompt", back_populates="versions")


# ==============================================================================
# WORKFLOWS
# ==============================================================================

class Workflow(Base):
    """
    A named sequence of prompt executions. Steps are normalized in workflow_steps.
    """
    __tablename__ = "workflows"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    input_vars = Column(JSON, default=list)  # [{name, type, default, description}]
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_archived = Column(Boolean, default=False)

    # Relationships
    steps = relationship("WorkflowStep", back_populates="workflow", order_by="WorkflowStep.step_order")
    runs = relationship("WorkflowRun", back_populates="workflow")


class WorkflowStep(Base):
    """
    A single step in a workflow. References a specific prompt version (FK-enforced).
    Deletion of a referenced prompt_version is blocked by FK constraint.
    """
    __tablename__ = "workflow_steps"

    id = Column(String, primary_key=True, default=_uuid)
    workflow_id = Column(String, ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False, index=True)
    prompt_version_id = Column(String, ForeignKey("prompt_versions.id", ondelete="RESTRICT"), nullable=False)
    step_order = Column(Integer, nullable=False)
    label = Column(String, nullable=False)
    input_mapping = Column(JSON, default=dict)   # Jinja2 variable resolution from inputs + prior steps
    output_key = Column(String, nullable=False)  # Reference as {{steps.<output_key>.text}}
    condition = Column(Text, nullable=True)      # Optional: skip condition expression
    retry_count = Column(Integer, default=1)     # Auto-retry on JSON parse failure

    # Relationships
    workflow = relationship("Workflow", back_populates="steps")
    prompt_version = relationship("PromptVersion")


class WorkflowRun(Base):
    """
    Execution record for a workflow run. Supports checkpointing and HITL pause.
    """
    __tablename__ = "workflow_runs"

    id = Column(String, primary_key=True, default=_uuid)
    workflow_id = Column(String, ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String, default="running")  # running | completed | failed | paused_for_approval
    inputs = Column(JSON, default=dict)
    step_outputs = Column(JSON, default=dict)   # {step_output_key: {text, structured, metadata}}
    current_step = Column(Integer, default=0)   # For checkpoint resume
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)

    # Relationship
    workflow = relationship("Workflow", back_populates="runs")


# ==============================================================================
# AGENT TOOLS
# ==============================================================================

class AgentTool(Base):
    """
    A published tool wrapping either a workflow or a single prompt version.
    Flat hierarchy: Tools → Workflows → Prompts. No recursive composition.
    """
    __tablename__ = "agent_tools"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    icon = Column(String, default="wrench")           # Lucide icon name
    workflow_id = Column(String, ForeignKey("workflows.id", ondelete="SET NULL"), nullable=True)
    prompt_version_id = Column(String, ForeignKey("prompt_versions.id", ondelete="SET NULL"), nullable=True)
    input_schema = Column(JSON, default=list)         # Exposed parameters [{name, type, required, default}]
    output_format = Column(String, default="text")    # text | table | json | markdown
    is_published = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    workflow = relationship("Workflow")
    prompt_version = relationship("PromptVersion")
