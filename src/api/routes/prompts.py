"""
Prompt Library API Routes.

CRUD for saved prompts and immutable prompt versions.
Execution endpoint uses the decoupled execution service (not chat router).

Schemas use orthogonal axes: context_source × output_format.
Legacy exec_mode payloads are silently translated via @model_validator.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.storage.studio_driver import get_studio_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/prompts", tags=["prompts"])


# ══════════════════════════════════════════════════════════════════
# Legacy Translator
# ══════════════════════════════════════════════════════════════════

_EXEC_MODE_MAP = {
    "rag":        {"context_source": "documents",  "output_format": "text"},
    "structured": {"context_source": "documents",  "output_format": "json"},
    "direct":     {"context_source": "none",        "output_format": "text"},
    "sql":        {"context_source": "metrics_db",  "output_format": "table"},
}

_RETRIEVAL_MODE_MAP = {
    "qualitative":  ["semantic"],
    "quantitative": ["numeric"],
}


def _translate_legacy(data: dict) -> dict:
    """Convert old exec_mode/retrieval_mode payloads to new axes."""
    if "exec_mode" in data and "context_source" not in data:
        mapping = _EXEC_MODE_MAP.get(data.pop("exec_mode"), {})
        data.update(mapping)
    if "retrieval_mode" in data and "search_strategy" not in data:
        rm = data.pop("retrieval_mode")
        if rm in _RETRIEVAL_MODE_MAP:
            data["search_strategy"] = _RETRIEVAL_MODE_MAP[rm]
    return data


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
    context_source: str = Field("documents", pattern="^(documents|metrics_db|none)$")
    output_format: str = Field("text", pattern="^(text|json|chart|table)$")
    search_strategy: List[str] = Field(default_factory=lambda: ["semantic"])
    output_schema: Optional[dict] = None
    model_id: Optional[str] = None
    temperature: float = Field(0.1, ge=0.0, le=2.0)

    @model_validator(mode="before")
    @classmethod
    def translate_legacy(cls, data):
        if isinstance(data, dict):
            return _translate_legacy(data)
        return data


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
    context_source: str
    output_format: str
    search_strategy: Optional[List[str]] = None
    temperature: Optional[float] = None
    created_at: Optional[str]
    is_published: bool


class VersionDetail(BaseModel):
    id: str
    prompt_id: str
    version: int
    template: str
    variables: list
    context_source: str
    output_format: str
    search_strategy: Optional[List[str]] = None
    output_schema: Optional[dict] = None
    model_id: Optional[str] = None
    temperature: Optional[float] = None
    created_at: Optional[str] = None
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
        context_source=body.context_source,
        output_format=body.output_format,
        search_strategy=body.search_strategy,
        output_schema=body.output_schema,
        model_id=body.model_id,
        temperature=body.temperature,
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
    context_source: str = Field("documents", pattern="^(documents|metrics_db|none)$")
    output_format: str = Field("text", pattern="^(text|json|chart|table)$")
    search_strategy: List[str] = Field(default_factory=lambda: ["semantic"])
    output_schema: Optional[dict] = None
    doc_filter: Optional[str] = None
    model_id: Optional[str] = None
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    force_retrieve: bool = Field(False, description="Bust retrieval cache and re-fetch chunks")

    @model_validator(mode="before")
    @classmethod
    def translate_legacy(cls, data):
        if isinstance(data, dict):
            return _translate_legacy(data)
        return data

    @model_validator(mode="after")
    def validate_combination(self):
        if self.output_format in ("chart", "table") and self.context_source != "metrics_db":
            raise ValueError(f"{self.output_format} output requires metrics_db context source")
        if self.output_format == "json" and self.output_schema is None:
            pass  # Allow — schema can be optional for ad-hoc JSON
        return self


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
        key = _cache_key(prompt_id, body.doc_filter, body.search_strategy)
        _retrieval_cache.pop(key, None)

    try:
        result = await execute_prompt(
            template=body.template,
            variables=body.variables,
            context_source=body.context_source,
            output_format=body.output_format,
            search_strategy=body.search_strategy,
            output_schema=body.output_schema,
            doc_filter=body.doc_filter,
            model_id=body.model_id,
            temperature=body.temperature,
            prompt_id=prompt_id,
            force_retrieve=body.force_retrieve,
        )
    except Exception as e:
        logger.error(f"Prompt execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}")

    return PromptRunResponse(
        text=result.text,
        structured=result.structured,
        metadata=result.metadata,
    )
