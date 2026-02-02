"""
Schema definitions for Ingested Documents (The "Container" Phase).
Defines the structure of data chunks as they are stored in the vector database.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

# Ensure src.schemas.citation exists in your folder structure
from src.schemas.citation import BoundingBox


class IngestedChunk(BaseModel):
    """
    The Atomic Unit of Chiron.
    This is what gets embedded into Qdrant.
    """
    chunk_id: str = Field(..., description="Unique UUID for this text span")
    doc_hash: str = Field(..., description="Link to the Lineage Ledger")

    # The Token Firewall:
    # 'clean_text' strips headers/footers. This is what the Embedding Model sees.
    # 'raw_text' preserves the original context for the human.
    clean_text: str
    raw_text: str

    page_number: int
    token_count: int

    # Metadata for filtering (e.g., "Show me only 'Final' docs from 2024")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # The Sidecar Link:
    # If this chunk contains a table, we point to the Markdown file in Blobs.
    table_blob_path: Optional[str] = None

    # The Visual Link:
    # We store the bounding box here for immediate retrieval context,
    # though the full detailed layout map might be in a Blob.
    primary_bbox: Optional[BoundingBox] = None


class IngestionResult(BaseModel):
    """
    Summary report returned by the Pipeline after processing a file.
    """
    file_name: str
    doc_hash: str
    total_pages: int
    total_chunks: int
    processing_time_seconds: float
    cost_estimate_usd: float  # The "Latte Benchmark" metric
    status: str = "success"
    error_msg: Optional[str] = None