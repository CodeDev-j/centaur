"""
Retrieval Schemas: The Read-Path Data Contract.

Defines the typed containers for search results flowing from
Qdrant → Reranker → Citation Sidecar → Generation LLM.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from src.schemas.citation import BoundingBox


class RetrievedChunk(BaseModel):
    """One result from hybrid search + reranking."""
    chunk_id: str
    text: str
    score: float = Field(default=0.0, description="RRF or reranker score")
    page_number: int
    source_file: str
    item_id: str
    item_type: str
    doc_hash: str = ""
    bbox: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Container for a complete retrieval cycle."""
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    query_original: str = ""
    query_expanded: str = ""
    search_latency_ms: float = 0.0
