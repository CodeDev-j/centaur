from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Import the new Unified BoundingBox
from src.schemas.citation import BoundingBox

class IngestedChunk(BaseModel):
    """
    The Atomic Unit of Chiron (Centaur 2.0).
    This is what gets embedded into the Vector Database (Qdrant/Chroma).
    """
    chunk_id: str = Field(..., description="Unique UUID for this text span")
    doc_hash: str = Field(..., description="Link to the Lineage Ledger")
    
    # The Token Firewall:
    # 'clean_text': Stripped/Formatted. The Embedding Model sees this.
    # 'raw_text': Original content. The LLM sees this for context generation.
    clean_text: str 
    raw_text: str
    
    page_number: int
    token_count: Optional[int] = Field(default=0, description="Estimated token count")
    
    # Rich Metadata:
    # - For Visual Pages: contains "facts" (list of grounded numbers) and "blob_path" (image).
    # - For Tables: contains "rows" (e.g., "0-10") and table structure info.
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Visual Link (Optional Top-Level Fields):
    # If this chunk maps to a single visual region (like a paragraph), we store it here.
    # For complex charts with multiple facts, look inside metadata['facts'].
    primary_bbox: Optional[BoundingBox] = None
    
    # Sidecar Link:
    # If the table is too large for the context window, we store the full HTML/Markdown
    # in a blob and point to it here.
    table_blob_path: Optional[str] = None

class IngestionResult(BaseModel):
    """
    Summary report returned by the Pipeline after processing a file.
    Used for the "Latte Benchmark" (Cost/Performance tracking).
    """
    file_name: str
    doc_hash: str
    status: str = "success"
    
    # Metrics
    total_pages: int
    total_chunks: int
    processing_time_seconds: float
    cost_estimate_usd: float = 0.0 # Tracking the cost of VLM calls
    
    # Diagnostics
    error_msg: Optional[str] = None