"""
Audit Routes: Data quality findings API.

GET /api/v1/documents/{doc_hash}/audit          — All findings for a document
GET /api/v1/documents/{doc_hash}/audit/summary  — Severity counts
POST /api/v1/documents/{doc_hash}/audit/run     — Force re-run audit checks
"""

import logging
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException

from src.audit.engine import AuditEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["audit"])

# ---------------------------------------------------------------------------
# Engine wiring (set during app startup in main.py)
# ---------------------------------------------------------------------------
_audit_engine: AuditEngine | None = None


def set_audit_engine(engine: AuditEngine) -> None:
    global _audit_engine
    _audit_engine = engine


def _get_engine() -> AuditEngine:
    if _audit_engine is None:
        raise HTTPException(status_code=503, detail="Audit engine not initialized")
    return _audit_engine


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{doc_hash}/audit")
async def get_audit_findings(doc_hash: str) -> List[Dict[str, Any]]:
    """Returns all audit findings for a document, ordered by severity."""
    engine = _get_engine()
    try:
        return engine.get_findings(doc_hash)
    except Exception as e:
        logger.error(f"Failed to get audit findings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve audit findings")


@router.get("/{doc_hash}/audit/summary")
async def get_audit_summary(doc_hash: str) -> Dict[str, int]:
    """Returns severity counts: {error: N, warning: N, info: N}."""
    engine = _get_engine()
    try:
        return engine.get_summary(doc_hash)
    except Exception as e:
        logger.error(f"Failed to get audit summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve audit summary")


@router.post("/{doc_hash}/audit/run")
async def run_audit(doc_hash: str) -> List[Dict[str, Any]]:
    """Force re-run all audit checks for a document. Returns findings."""
    engine = _get_engine()
    try:
        return engine.run_all(doc_hash)
    except Exception as e:
        logger.error(f"Failed to run audit: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Audit run failed")
