"""
Export Routes: Excel workbook download.

GET /api/v1/documents/{doc_hash}/export/excel  — Single document export
"""

import logging
from pathlib import PurePosixPath

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.export.excel_builder import build_workbook
from src.storage.analytics_driver import AnalyticsDriver
from src.storage.db_driver import ledger_db, DocumentLedger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["export"])

# ---------------------------------------------------------------------------
# Driver wiring (set during app startup in main.py)
# ---------------------------------------------------------------------------
_analytics: AnalyticsDriver | None = None
_audit_engine = None  # Optional — for audit summary in workbook


def set_export_drivers(analytics: AnalyticsDriver, audit_engine=None) -> None:
    global _analytics, _audit_engine
    _analytics = analytics
    _audit_engine = audit_engine


def _get_analytics() -> AnalyticsDriver:
    if _analytics is None:
        raise HTTPException(status_code=503, detail="Analytics driver not initialized")
    return _analytics


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/documents/{doc_hash}/export/excel")
async def export_excel(doc_hash: str):
    """
    Generates and returns a .xlsx workbook for the given document.
    Every value cell contains a =HYPERLINK() back to the source page.
    """
    analytics = _get_analytics()

    # Get document metadata from ledger
    doc_metadata = None
    if ledger_db:
        try:
            with ledger_db.session() as session:
                doc = session.query(DocumentLedger).filter_by(doc_hash=doc_hash).first()
                if doc:
                    doc_metadata = {
                        "filename": doc.filename,
                        "doc_hash": doc_hash,
                    }
        except Exception as e:
            logger.warning(f"Could not fetch doc metadata: {e}")

    if not doc_metadata:
        doc_metadata = {"filename": "Unknown", "doc_hash": doc_hash}

    # Get audit summary if engine is available
    audit_summary = None
    if _audit_engine:
        try:
            audit_summary = _audit_engine.get_summary(doc_hash)
        except Exception as e:
            logger.warning(f"Could not fetch audit summary for export: {e}")

    try:
        buffer = build_workbook(
            doc_hash=doc_hash,
            analytics=analytics,
            doc_metadata=doc_metadata,
            audit_summary=audit_summary,
        )
    except Exception as e:
        logger.error(f"Excel export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Excel export failed")

    # Safe filename for Content-Disposition
    safe_name = PurePosixPath(doc_metadata.get("filename", "export")).stem
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "-_ ")[:50]

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}_metrics.xlsx"',
        },
    )
