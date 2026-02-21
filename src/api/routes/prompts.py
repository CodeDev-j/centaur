"""
Prompt Library API Routes.

CRUD for saved prompts and immutable prompt versions.
Execution endpoint uses the decoupled execution service (not chat router).
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.storage.studio_driver import get_studio_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/prompts", tags=["prompts"])


# ══════════════════════════════════════════════════════════════════
# Request / Response Schemas
# ══════════════════════════════════════════════════════════════════

class PromptCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    category: str = "custom"
    tags: List[str] = Field(default_factory=list)


class PromptUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


class VersionPublish(BaseModel):
    template: str = Field(..., min_length=1)
    variables: List[dict] = Field(default_factory=list)
    exec_mode: str = Field("rag", pattern="^(rag|structured|direct|sql)$")
    output_schema: Optional[dict] = None
    model_id: Optional[str] = None
    retrieval_mode: Optional[str] = Field(None, pattern="^(qualitative|quantitative)$")


class PromptSummary(BaseModel):
    id: str
    name: str
    description: str
    category: str
    tags: List[str]
    created_at: Optional[str]
    is_archived: bool
    latest_version: int


class VersionSummary(BaseModel):
    id: str
    version: int
    exec_mode: str
    created_at: Optional[str]
    is_published: bool


class VersionDetail(BaseModel):
    id: str
    prompt_id: str
    version: int
    template: str
    variables: list
    exec_mode: str
    output_schema: Optional[dict]
    model_id: Optional[str]
    retrieval_mode: Optional[str]
    created_at: Optional[str]
    is_published: bool


class PromptDetail(BaseModel):
    id: str
    name: str
    description: str
    category: str
    tags: List[str]
    created_at: Optional[str]
    is_archived: bool
    versions: List[VersionDetail]


# ══════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════

@router.post("", response_model=PromptSummary, status_code=201)
async def create_prompt(body: PromptCreate):
    """Create a new saved prompt (no versions yet)."""
    db = get_studio_db()
    prompt = db.create_prompt(
        name=body.name,
        description=body.description,
        category=body.category,
        tags=body.tags,
    )
    return PromptSummary(
        id=prompt.id,
        name=prompt.name,
        description=prompt.description,
        category=prompt.category,
        tags=prompt.tags,
        created_at=prompt.created_at.isoformat() if prompt.created_at else None,
        is_archived=prompt.is_archived,
        latest_version=0,
    )


@router.get("", response_model=List[PromptSummary])
async def list_prompts(category: Optional[str] = None, include_archived: bool = False):
    """List all prompts, optionally filtered by category."""
    db = get_studio_db()
    return db.list_prompts(category=category, include_archived=include_archived)


@router.get("/{prompt_id}", response_model=PromptDetail)
async def get_prompt(prompt_id: str):
    """Get a prompt with all its versions."""
    db = get_studio_db()
    result = db.get_prompt(prompt_id)
    if not result:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return result


@router.put("/{prompt_id}")
async def update_prompt(prompt_id: str, body: PromptUpdate):
    """Update prompt metadata (name, description, category, tags)."""
    db = get_studio_db()
    success = db.update_prompt(
        prompt_id=prompt_id,
        name=body.name,
        description=body.description,
        category=body.category,
        tags=body.tags,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"status": "updated"}


@router.delete("/{prompt_id}")
async def archive_prompt(prompt_id: str):
    """Soft-delete (archive) a prompt."""
    db = get_studio_db()
    success = db.archive_prompt(prompt_id)
    if not success:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"status": "archived"}


@router.post("/{prompt_id}/versions", response_model=VersionDetail, status_code=201)
async def publish_version(prompt_id: str, body: VersionPublish):
    """Publish a new immutable version of a prompt."""
    db = get_studio_db()
    result = db.publish_version(
        prompt_id=prompt_id,
        template=body.template,
        variables=body.variables,
        exec_mode=body.exec_mode,
        output_schema=body.output_schema,
        model_id=body.model_id,
        retrieval_mode=body.retrieval_mode,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return result


@router.get("/{prompt_id}/versions", response_model=List[VersionSummary])
async def list_versions(prompt_id: str):
    """List all versions of a prompt (newest first)."""
    db = get_studio_db()
    return db.list_versions(prompt_id)


@router.get("/versions/{version_id}", response_model=VersionDetail)
async def get_version(version_id: str):
    """Get a specific prompt version by ID."""
    db = get_studio_db()
    result = db.get_version(version_id)
    if not result:
        raise HTTPException(status_code=404, detail="Version not found")
    return result


# ══════════════════════════════════════════════════════════════════
# Execution (Test Run)
# ══════════════════════════════════════════════════════════════════

class PromptRunRequest(BaseModel):
    """Run a prompt template with variables. Does NOT require a published version."""
    template: str = Field(..., min_length=1)
    variables: dict = Field(default_factory=dict)
    exec_mode: str = Field("rag", pattern="^(rag|structured|direct|sql)$")
    output_schema: Optional[dict] = None
    retrieval_mode: str = "qualitative"
    doc_filter: Optional[str] = None
    model_id: Optional[str] = None
    force_retrieve: bool = Field(False, description="Bust retrieval cache and re-fetch chunks")


class PromptRunResponse(BaseModel):
    text: str
    structured: Optional[dict] = None
    metadata: dict = Field(default_factory=dict)


@router.post("/{prompt_id}/run", response_model=PromptRunResponse)
async def run_prompt(prompt_id: str, body: PromptRunRequest):
    """
    Execute a prompt template (test run). Ephemeral — not persisted.
    Uses the decoupled execution service, NOT the chat router.
    """
    from src.workflows.prompt_executor import execute_prompt, _retrieval_cache, _cache_key

    # Bust cache if requested
    if body.force_retrieve:
        key = _cache_key(prompt_id, body.doc_filter)
        _retrieval_cache.pop(key, None)

    result = await execute_prompt(
        template=body.template,
        variables=body.variables,
        exec_mode=body.exec_mode,
        output_schema=body.output_schema,
        retrieval_mode=body.retrieval_mode,
        doc_filter=body.doc_filter,
        model_id=body.model_id,
        prompt_id=prompt_id,
    )

    return PromptRunResponse(
        text=result.text,
        structured=result.structured,
        metadata=result.metadata,
    )
