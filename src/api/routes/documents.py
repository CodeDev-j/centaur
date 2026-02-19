"""
Document Routes: Serve document pages and region overlays for the viewer.

GET /api/v1/documents/{hash}/page/{n}/image    — Rendered page image (PyMuPDF)
GET /api/v1/documents/{hash}/page/{n}/regions  — Bbox overlays for highlighting
"""

import io
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import RegionOverlay
from src.config import SystemPaths
from src.storage.db_driver import ledger_db, DocumentLedger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


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
        logger.error(f"Failed to render page: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{doc_hash}/page/{page_number}/regions", response_model=List[RegionOverlay])
async def get_page_regions(doc_hash: str, page_number: int):
    """
    Returns bounding box overlays for a given page.
    Reads from the shadow cache or reconstructs from layout data.
    Currently returns empty — will be populated when layout cache is implemented.
    """
    # TODO: Read from layout cache when available
    # For now, return empty list. The frontend can still display the page.
    return []
