"""
API Schemas: Request/Response models for the FastAPI endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from src.schemas.citation import Citation


class ChatMessage(BaseModel):
    """A single message in conversation history."""
    role: str = Field(..., pattern=r"^(user|assistant)$", description="'user' or 'assistant'")
    content: str = Field(..., min_length=1, max_length=50_000, description="Message text")


class ChatRequest(BaseModel):
    """Incoming chat query."""
    query: str = Field(..., min_length=1, max_length=10_000, description="The user's question")
    locale: Optional[str] = Field("en", pattern=r"^[a-z]{2}$", description="User locale: en, de, fr")
    doc_filter: Optional[str] = Field(None, pattern=r"^[a-fA-F0-9]{64}$", description="SHA-256 doc_hash")
    messages: Optional[List[ChatMessage]] = Field(None, max_length=50, description="Conversation history for multi-turn")


class ChatResponse(BaseModel):
    """Chat response with cited answer."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    query_route: str = "hybrid"
    query_expanded: str = ""
    sql_result: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0


class DocumentSummary(BaseModel):
    """Summary of an ingested document, including metadata."""
    doc_hash: str
    filename: str
    status: str
    upload_date: Optional[str] = None
    total_items: int = 0
    # Metadata fields (from document_meta table)
    company_name: Optional[str] = None
    document_type: Optional[str] = None
    project_code: Optional[str] = None
    as_of_date: Optional[str] = None
    period_label: Optional[str] = None
    sector: Optional[str] = None
    currency: Optional[str] = None
    page_count: int = 0
    tags: List[str] = Field(default_factory=list)
    extraction_confidence: float = 0.0


class RegionOverlay(BaseModel):
    """Bounding box overlay for the document viewer."""
    region_id: int
    x: float
    y: float
    width: float
    height: float
    archetype: str = ""
    title: str = ""


# ==============================================================================
# Chunk Inspector Schemas
# ==============================================================================

class ChunkDetail(BaseModel):
    """A single chunk from the vector database, for inspection."""
    chunk_id: str
    item_id: str
    item_type: str
    chunk_text: str
    chunk_role: Optional[str] = None
    page_number: int = 0
    bbox: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    value_bboxes_count: int = 0
    value_bboxes: Optional[Dict[str, List[List[float]]]] = None


class PageChunksResponse(BaseModel):
    """All chunks for a specific page."""
    doc_hash: str
    page_number: int
    total_chunks: int
    chunks: List[ChunkDetail]


class DocChunksResponse(BaseModel):
    """All chunks for a document, optionally filtered."""
    doc_hash: str
    total_chunks: int
    chunks: List[ChunkDetail]


class DocStatsResponse(BaseModel):
    """Document-level chunk statistics."""
    doc_hash: str
    total_chunks: int
    by_type: Dict[str, int]


# ==============================================================================
# Document Metadata Schemas
# ==============================================================================

class DocumentMetaResponse(BaseModel):
    """Full metadata for a single document."""
    doc_hash: str
    company_name: Optional[str] = None
    document_type: Optional[str] = None
    project_code: Optional[str] = None
    as_of_date: Optional[str] = None
    period_label: Optional[str] = None
    publish_date: Optional[str] = None
    sector: Optional[str] = None
    geography: Optional[str] = None
    currency: Optional[str] = None
    confidentiality: Optional[str] = None
    language: str = "en"
    page_count: int = 0
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    extraction_confidence: float = 0.0
    user_overrides: Dict[str, bool] = Field(default_factory=dict)
    last_edited_at: Optional[str] = None


class DocumentMetaUpdate(BaseModel):
    """User-editable metadata fields."""
    company_name: Optional[str] = None
    document_type: Optional[str] = None
    project_code: Optional[str] = None
    as_of_date: Optional[str] = None
    period_label: Optional[str] = None
    publish_date: Optional[str] = None
    sector: Optional[str] = None
    geography: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class FacetsResponse(BaseModel):
    """Distinct values for filter dropdowns."""
    companies: List[str]
    sectors: List[str]
    document_types: List[str]
    projects: List[str]


class BatchRequest(BaseModel):
    """Batch operation on multiple documents."""
    doc_hashes: List[str] = Field(..., max_length=100, description="Max 100 documents per batch")
    action: str  # "tag", "delete", "re_extract"
    tags: Optional[List[str]] = Field(None, max_length=50)
