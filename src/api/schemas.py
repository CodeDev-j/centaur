"""
API Schemas: Request/Response models for the FastAPI endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from src.schemas.citation import Citation


class ChatMessage(BaseModel):
    """A single message in conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    """Incoming chat query."""
    query: str = Field(..., min_length=1, description="The user's question")
    locale: Optional[str] = Field("en", description="User locale: en, de, fr")
    doc_filter: Optional[str] = Field(None, description="Filter by doc_hash")
    messages: Optional[List[ChatMessage]] = Field(None, description="Conversation history for multi-turn")


class ChatResponse(BaseModel):
    """Chat response with cited answer."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    query_route: str = "hybrid"
    query_expanded: str = ""
    sql_result: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0


class DocumentSummary(BaseModel):
    """Summary of an ingested document."""
    doc_hash: str
    filename: str
    status: str
    upload_date: Optional[str] = None
    total_items: int = 0


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
