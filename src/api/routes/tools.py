"""
Agent Tools API Routes.

CRUD for AgentTools — named wrappers around prompts/workflows
that can be exposed as MCP server endpoints (Phase 2).
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.storage.studio_driver import get_studio_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/tools", tags=["tools"])


# ══════════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════════

class ToolCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    icon: str = "wrench"
    workflow_id: Optional[str] = None
    prompt_version_id: Optional[str] = None
    input_schema: List[dict] = Field(default_factory=list)
    output_format: str = "text"


class ToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    is_published: Optional[bool] = None


class ToolSummary(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    workflow_id: Optional[str]
    prompt_version_id: Optional[str]
    input_schema: list
    output_format: str
    is_published: bool
    created_at: Optional[str]


# ══════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════

@router.get("", response_model=List[ToolSummary])
async def list_tools(include_unpublished: bool = True):
    """List all agent tools."""
    db = get_studio_db()
    return db.list_tools(include_unpublished=include_unpublished)


@router.post("", response_model=ToolSummary, status_code=201)
async def create_tool(body: ToolCreate):
    """Create an agent tool from a prompt version or workflow."""
    db = get_studio_db()
    result = db.create_tool(
        name=body.name,
        description=body.description,
        icon=body.icon,
        workflow_id=body.workflow_id,
        prompt_version_id=body.prompt_version_id,
        input_schema=body.input_schema,
        output_format=body.output_format,
    )
    return result


@router.put("/{tool_id}")
async def update_tool(tool_id: str, body: ToolUpdate):
    """Update agent tool metadata."""
    db = get_studio_db()
    success = db.update_tool(
        tool_id=tool_id,
        name=body.name,
        description=body.description,
        icon=body.icon,
        is_published=body.is_published,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"status": "updated"}


@router.delete("/{tool_id}")
async def archive_tool(tool_id: str):
    """Soft-delete (unpublish) an agent tool."""
    db = get_studio_db()
    success = db.archive_tool(tool_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"status": "archived"}
