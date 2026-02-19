"""
API Schemas: Request/Response models for the FastAPI endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from src.schemas.citation import Citation


class ChatRequest(BaseModel):
    """Incoming chat query."""
    query: str = Field(..., min_length=1, description="The user's question")
    locale: Optional[str] = Field("en", description="User locale: en, de, fr")
    doc_filter: Optional[str] = Field(None, description="Filter by doc_hash")


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
