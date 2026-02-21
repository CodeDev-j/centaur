"""
Workflow API Routes.

CRUD for workflows and their steps.
Execution uses the SequentialExecutor (LangGraph-based).
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.storage.studio_driver import get_studio_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


# ══════════════════════════════════════════════════════════════════
# Request / Response Schemas
# ══════════════════════════════════════════════════════════════════

class WorkflowCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    input_vars: List[dict] = Field(default_factory=list)


class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    input_vars: Optional[List[dict]] = None


class StepCreate(BaseModel):
    prompt_version_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    output_key: str = Field(..., min_length=1)
    input_mapping: dict = Field(default_factory=dict)
    condition: Optional[str] = None
    retry_count: int = 1


class StepUpdate(BaseModel):
    label: Optional[str] = None
    input_mapping: Optional[dict] = None
    output_key: Optional[str] = None
    condition: Optional[str] = None
    retry_count: Optional[int] = None
    prompt_version_id: Optional[str] = None


class StepReorder(BaseModel):
    step_ids: List[str] = Field(..., min_length=1)


class WorkflowSummary(BaseModel):
    id: str
    name: str
    description: str
    input_vars: list
    created_at: Optional[str]
    updated_at: Optional[str]
    is_archived: bool
    step_count: int


class StepDetail(BaseModel):
    id: str
    step_order: int
    label: str
    prompt_version_id: str
    input_mapping: dict
    output_key: str
    condition: Optional[str]
    retry_count: int
    prompt_version: Optional[dict]


class WorkflowDetail(BaseModel):
    id: str
    name: str
    description: str
    input_vars: list
    created_at: Optional[str]
    updated_at: Optional[str]
    is_archived: bool
    steps: List[StepDetail]


class RunSummary(BaseModel):
    id: str
    status: str
    current_step: int
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


class RunDetail(BaseModel):
    id: str
    workflow_id: str
    status: str
    inputs: dict
    step_outputs: dict
    current_step: int
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


class WorkflowRunRequest(BaseModel):
    inputs: dict = Field(default_factory=dict)
    doc_filter: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# Workflow CRUD Routes
# ══════════════════════════════════════════════════════════════════

@router.post("", response_model=WorkflowSummary, status_code=201)
async def create_workflow(body: WorkflowCreate):
    """Create a new workflow (no steps yet)."""
    db = get_studio_db()
    wf = db.create_workflow(
        name=body.name,
        description=body.description,
        input_vars=body.input_vars,
    )
    return WorkflowSummary(
        id=wf.id,
        name=wf.name,
        description=wf.description,
        input_vars=wf.input_vars,
        created_at=wf.created_at.isoformat() if wf.created_at else None,
        updated_at=wf.updated_at.isoformat() if wf.updated_at else None,
        is_archived=wf.is_archived,
        step_count=0,
    )


@router.get("", response_model=List[WorkflowSummary])
async def list_workflows(include_archived: bool = False):
    """List all workflows."""
    db = get_studio_db()
    return db.list_workflows(include_archived=include_archived)


@router.get("/{workflow_id}", response_model=WorkflowDetail)
async def get_workflow(workflow_id: str):
    """Get a workflow with all its steps."""
    db = get_studio_db()
    result = db.get_workflow(workflow_id)
    if not result:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return result


@router.put("/{workflow_id}")
async def update_workflow(workflow_id: str, body: WorkflowUpdate):
    """Update workflow metadata."""
    db = get_studio_db()
    success = db.update_workflow(
        workflow_id=workflow_id,
        name=body.name,
        description=body.description,
        input_vars=body.input_vars,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {"status": "updated"}


@router.delete("/{workflow_id}")
async def archive_workflow(workflow_id: str):
    """Soft-delete (archive) a workflow."""
    db = get_studio_db()
    success = db.archive_workflow(workflow_id)
    if not success:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {"status": "archived"}


# ══════════════════════════════════════════════════════════════════
# Step Management Routes
# ══════════════════════════════════════════════════════════════════

@router.post("/{workflow_id}/steps", status_code=201)
async def add_step(workflow_id: str, body: StepCreate):
    """Add a step to a workflow."""
    db = get_studio_db()
    result = db.add_step(
        workflow_id=workflow_id,
        prompt_version_id=body.prompt_version_id,
        label=body.label,
        output_key=body.output_key,
        input_mapping=body.input_mapping,
        condition=body.condition,
        retry_count=body.retry_count,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Workflow or prompt version not found")
    return result


@router.put("/{workflow_id}/steps/{step_id}")
async def update_step(workflow_id: str, step_id: str, body: StepUpdate):
    """Update a workflow step."""
    db = get_studio_db()
    success = db.update_step(
        step_id=step_id,
        label=body.label,
        input_mapping=body.input_mapping,
        output_key=body.output_key,
        condition=body.condition,
        retry_count=body.retry_count,
        prompt_version_id=body.prompt_version_id,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Step not found")
    return {"status": "updated"}


@router.delete("/{workflow_id}/steps/{step_id}")
async def remove_step(workflow_id: str, step_id: str):
    """Remove a step from a workflow (auto-reorders remaining steps)."""
    db = get_studio_db()
    success = db.remove_step(step_id)
    if not success:
        raise HTTPException(status_code=404, detail="Step not found")
    return {"status": "removed"}


@router.put("/{workflow_id}/steps/reorder")
async def reorder_steps(workflow_id: str, body: StepReorder):
    """Reorder workflow steps."""
    db = get_studio_db()
    success = db.reorder_steps(workflow_id, body.step_ids)
    if not success:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {"status": "reordered"}


# ══════════════════════════════════════════════════════════════════
# Execution Routes
# ══════════════════════════════════════════════════════════════════

@router.post("/{workflow_id}/run", response_model=RunDetail)
async def run_workflow(workflow_id: str, body: WorkflowRunRequest):
    """Execute a workflow. Returns the completed run record."""
    from src.workflows.workflow_executor import execute_workflow

    result = await execute_workflow(
        workflow_id=workflow_id,
        inputs=body.inputs,
        doc_filter=body.doc_filter,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return result


@router.get("/{workflow_id}/runs", response_model=List[RunSummary])
async def list_runs(workflow_id: str, limit: int = 20):
    """List execution history for a workflow."""
    db = get_studio_db()
    return db.list_runs(workflow_id, limit=limit)


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: str):
    """Get a specific workflow run by ID."""
    db = get_studio_db()
    result = db.get_run(run_id)
    if not result:
        raise HTTPException(status_code=404, detail="Run not found")
    return result


@router.post("/runs/{run_id}/approve")
async def approve_run(run_id: str):
    """Resume a paused-for-approval workflow run."""
    from src.workflows.workflow_executor import resume_workflow

    result = await resume_workflow(run_id)
    if not result:
        raise HTTPException(status_code=404, detail="Run not found or not paused")
    return result
