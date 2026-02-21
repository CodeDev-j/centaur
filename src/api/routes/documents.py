"""
Document Routes: Serve document pages, region overlays, and chunk inspection.

GET /api/v1/documents/{hash}/pdf                    — Raw PDF for react-pdf
GET /api/v1/documents/{hash}/page/{n}/image         — Rendered page image (PyMuPDF)
GET /api/v1/documents/{hash}/page/{n}/regions       — Bbox overlays (stub)
GET /api/v1/documents/{hash}/page/{n}/chunks        — All chunks for a page
GET /api/v1/documents/{hash}/chunks                 — All chunks for a document (filtered)
GET /api/v1/documents/{hash}/stats                  — Document-level chunk counts
"""

import io
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ChunkDetail,
    DocChunksResponse,
    DocStatsResponse,
    PageChunksResponse,
    RegionOverlay,
)
from src.config import SystemPaths
from src.storage.db_driver import ledger_db, DocumentLedger
from src.storage.vector_driver import VectorDriver

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# ---------------------------------------------------------------------------
# Vector driver wiring (set during app startup in main.py)
# ---------------------------------------------------------------------------
_vector_driver: VectorDriver | None = None


def set_vector_driver(driver: VectorDriver) -> None:
    global _vector_driver
    _vector_driver = driver


def _get_vector_driver() -> VectorDriver:
    if _vector_driver is None:
        raise HTTPException(status_code=503, detail="Vector driver not initialized")
    return _vector_driver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_pdf_path(doc_hash: str) -> Path:
    """Finds the PDF file path from the ledger."""
    if not ledger_db:
        raise HTTPException(status_code=503, detail="Database not available")

    with ledger_db.session() as session:
        doc = session.query(DocumentLedger).filter_by(doc_hash=doc_hash).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        pdf_path = SystemPaths.INPUTS / doc.filename
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found on disk")
        return pdf_path


# Chunk type sort priority: tables first, then charts, visuals, narrative, headers
_TYPE_PRIORITY = {
    "financial_table": 0,
    "chart_table": 1,
    "visual": 2,
    "narrative": 3,
    "header": 4,
}

# Core payload fields that are NOT type-specific metadata
_CORE_FIELDS = {
    "doc_id", "doc_hash", "source_file", "page_number",
    "item_id", "item_type", "chunk_text", "chunk_role",
    "bbox_x", "bbox_y", "bbox_width", "bbox_height",
    "value_bboxes",
}


def _point_to_chunk(point) -> ChunkDetail:
    """Converts a Qdrant point to a ChunkDetail for the API response."""
    p = point.payload

    # Extract bbox if present
    bbox = None
    if all(k in p for k in ("bbox_x", "bbox_y", "bbox_width", "bbox_height")):
        bbox = {
            "x": float(p["bbox_x"]),
            "y": float(p["bbox_y"]),
            "width": float(p["bbox_width"]),
            "height": float(p["bbox_height"]),
        }

    # Extract value_bboxes (Dict[str, List[List[float]]]) for fine-grained highlighting
    vb = p.get("value_bboxes")
    vb_dict = vb if isinstance(vb, dict) else None
    vb_count = len(vb_dict) if vb_dict else 0

    # Everything not in _CORE_FIELDS is type-specific metadata
    metadata = {k: v for k, v in p.items() if k not in _CORE_FIELDS}

    return ChunkDetail(
        chunk_id=str(point.id),
        item_id=p.get("item_id", ""),
        item_type=p.get("item_type", "unknown"),
        chunk_text=p.get("chunk_text", ""),
        chunk_role=p.get("chunk_role"),
        page_number=p.get("page_number", 0),
        bbox=bbox,
        metadata=metadata,
        value_bboxes_count=vb_count,
        value_bboxes=vb_dict,
    )


def _chunk_sort_key(chunk: ChunkDetail) -> tuple:
    """Sort: type priority, then summary before series, then by series_label."""
    type_pri = _TYPE_PRIORITY.get(chunk.item_type, 99)
    role = chunk.chunk_role or ""
    role_pri = 0 if role == "summary" else 1
    label = chunk.metadata.get("series_label", "")
    return (type_pri, role_pri, label)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{doc_hash}/pdf")
async def get_pdf(doc_hash: str):
    """Serves the raw PDF file for react-pdf rendering."""
    pdf_path = _find_pdf_path(doc_hash)
    return StreamingResponse(
        open(str(pdf_path), "rb"),
        media_type="application/pdf",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{doc_hash}/page/{page_number}/image")
async def get_page_image(doc_hash: str, page_number: int):
    """
    Renders a PDF page as a PNG image using PyMuPDF.
    page_number is 1-based.
    """
    pdf_path = _find_pdf_path(doc_hash)

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        if page_number < 1 or page_number > len(doc):
            doc.close()
            raise HTTPException(status_code=404, detail=f"Page {page_number} not found")

        page = doc[page_number - 1]  # 0-indexed
        # Render at 2x for crisp display
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()

        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to render page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to render page")


@router.get("/{doc_hash}/page/{page_number}/regions", response_model=List[RegionOverlay])
async def get_page_regions(doc_hash: str, page_number: int):
    """
    Returns bounding box overlays for a given page.
    Currently returns empty — will be populated when layout cache is implemented.
    """
    return []


@router.get(
    "/{doc_hash}/page/{page_number}/chunks",
    response_model=PageChunksResponse,
)
async def get_page_chunks(doc_hash: str, page_number: int):
    """
    Returns all indexed chunks for a specific page, sorted by type priority.
    Uses Qdrant scroll with indexed payload filters — no embedding needed.
    value_bboxes are stripped from the response (only count is included).
    """
    vd = _get_vector_driver()
    points = vd.scroll_by_page(doc_hash, page_number)

    chunks = [_point_to_chunk(pt) for pt in points]
    chunks.sort(key=_chunk_sort_key)

    return PageChunksResponse(
        doc_hash=doc_hash,
        page_number=page_number,
        total_chunks=len(chunks),
        chunks=chunks,
    )


@router.get("/{doc_hash}/chunks", response_model=DocChunksResponse)
async def get_doc_chunks(
    doc_hash: str,
    item_type: Optional[str] = Query(None, description="Filter by item_type (e.g. visual)"),
    chunk_role: Optional[str] = Query(None, description="Filter by chunk_role (e.g. series)"),
):
    """
    Returns all indexed chunks for a document, optionally filtered.
    Used by: Metric Explorer (item_type=visual&chunk_role=series), TSV export (no filter).
    """
    vd = _get_vector_driver()
    points = vd.scroll_by_doc(doc_hash, item_type=item_type, chunk_role=chunk_role)

    chunks = [_point_to_chunk(pt) for pt in points]
    chunks.sort(key=_chunk_sort_key)

    return DocChunksResponse(
        doc_hash=doc_hash,
        total_chunks=len(chunks),
        chunks=chunks,
    )


@router.get("/{doc_hash}/stats", response_model=DocStatsResponse)
async def get_doc_stats(doc_hash: str):
    """
    Returns document-level chunk counts grouped by item_type.
    Uses Qdrant's Count API with indexed filters — sub-1ms per type.
    """
    vd = _get_vector_driver()
    counts = vd.count_by_doc(doc_hash)
    total = sum(counts.values())

    return DocStatsResponse(
        doc_hash=doc_hash,
        total_chunks=total,
        by_type=counts,
    )
