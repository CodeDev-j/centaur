"""
Ingestion Routes: Upload and manage documents.

POST /api/v1/ingest                    — Upload a file for processing (async, returns 202)
GET  /api/v1/ingest/{doc_hash}/status  — SSE stream for ingestion progress
GET  /api/v1/documents                 — List all ingested documents
GET  /api/v1/documents/{hash}          — Get document details
"""

import asyncio
import hashlib
import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langsmith import traceable

from src.api.schemas import DocumentSummary
from src.config import SystemPaths
from src.storage.db_driver import ledger_db, DocumentLedger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["ingestion"])

# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
_ALLOWED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg",          # Visual stream
    ".xlsx", ".xls", ".xlsm", ".csv",         # Native stream
    ".docx", ".pptx", ".md", ".txt",          # Native stream
}
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

# Module-level pipeline reference — set by main.py
_pipeline = None


def set_pipeline(pipeline):
    global _pipeline
    _pipeline = pipeline


# ==========================================================================
# In-memory ingestion tracker (signaling between background task and SSE)
# ==========================================================================

@dataclass
class _IngestionTracker:
    status: str = "processing"
    error: Optional[str] = None
    phase: str = "starting"       # Current phase: parsing, indexing, etc.
    progress: str = ""            # Human-readable progress e.g. "Page 3 of 29"
    items_count: int = 0          # Total items extracted
    quarantine_count: int = 0     # Items that failed to parse
    done: asyncio.Event = field(default_factory=asyncio.Event)
    updated: asyncio.Event = field(default_factory=asyncio.Event)  # Fires on progress change


_trackers: Dict[str, _IngestionTracker] = {}

TRACKER_CLEANUP_DELAY_S = 60  # Remove tracker 60s after completion


async def _run_ingestion_background(file_path: Path, doc_hash: str):
    """
    Background task: runs the full ingestion pipeline, updates ledger,
    and signals the SSE tracker on completion.
    """
    tracker = _trackers.get(doc_hash)

    try:
        # Progress callback — updates tracker and signals SSE listeners
        def on_progress(phase: str, detail: str):
            if tracker:
                tracker.phase = phase
                tracker.progress = detail
                tracker.updated.set()
                tracker.updated = asyncio.Event()  # Reset for next update

        result = await _pipeline.run(file_path, on_progress=on_progress)

        # Check if pipeline returned an IngestionResult (internal failure)
        # vs a UnifiedDocument (success)
        from src.schemas.documents import IngestionResult
        from src.schemas.deal_stream import UnifiedDocument
        if isinstance(result, IngestionResult) and result.status != "completed":
            raise RuntimeError(
                f"Pipeline returned {result.status}: {result.error_msg or 'unknown error'}"
            )

        # Surface quarantine stats
        if tracker and isinstance(result, UnifiedDocument):
            tracker.items_count = len(result.items)
            tracker.quarantine_count = len(result.quarantined_items) if result.quarantined_items else 0

        # Ensure ledger status is "completed"
        if ledger_db:
            with ledger_db.session() as session:
                entry = session.query(DocumentLedger).filter_by(doc_hash=doc_hash).first()
                if entry:
                    entry.status = "completed"

        if tracker:
            tracker.status = "completed"
        logger.info(f"Background ingestion completed: {file_path.name}")

    except Exception as e:
        logger.error(f"Background ingestion failed for {file_path.name}: {e}", exc_info=True)

        # Clean up partial vectors so failed docs don't leave orphaned data
        try:
            await _pipeline.indexer.delete_by_doc_hash(doc_hash)
            logger.info(f"Cleaned up partial vectors for failed doc: {doc_hash[:12]}...")
        except Exception as cleanup_err:
            logger.warning(f"Vector cleanup failed (non-critical): {cleanup_err}")

        if ledger_db:
            try:
                with ledger_db.session() as session:
                    entry = session.query(DocumentLedger).filter_by(doc_hash=doc_hash).first()
                    if entry:
                        entry.status = "failed"
            except Exception as db_err:
                logger.error(f"Failed to update ledger status: {db_err}")

        if tracker:
            tracker.status = "failed"
            tracker.error = str(e)

    finally:
        # Signal any SSE listeners that ingestion is done
        if tracker:
            tracker.done.set()

        # Schedule cleanup so the tracker doesn't leak memory
        loop = asyncio.get_running_loop()
        loop.call_later(TRACKER_CLEANUP_DELAY_S, lambda: _trackers.pop(doc_hash, None))


# ==========================================================================
# Endpoints
# ==========================================================================

@router.post("/ingest")
@traceable(name="Ingest Endpoint", run_type="chain")
async def ingest(file: UploadFile = File(...), force: bool = False):
    """
    Upload a file for ingestion. Returns 202 immediately.
    Subscribe to GET /ingest/{doc_hash}/status for SSE progress updates.

    If the file has already been successfully ingested (same content hash),
    returns 200 with the existing doc_hash — no re-processing.
    Pass force=true to re-ingest (e.g., after pipeline/prompt changes).
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # 0. Validate file type and size at the boundary
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )
    if file.size and file.size > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file.size / 1024 / 1024:.0f} MB). Max: {_MAX_UPLOAD_BYTES // 1024 // 1024} MB",
        )

    # 1. Save uploaded file to inputs directory
    # Sanitize filename: strip path components to prevent directory traversal
    safe_name = PurePosixPath(file.filename).name if file.filename else "upload"
    safe_name = Path(safe_name).name  # Also handle Windows-style separators
    if not safe_name or safe_name.startswith("."):
        safe_name = "upload" + ext
    # UUID prefix prevents race conditions on identical filenames
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    dest = SystemPaths.INPUTS / unique_name
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

    # Verify actual file size on disk (handles missing Content-Length header)
    actual_size = dest.stat().st_size
    if actual_size > _MAX_UPLOAD_BYTES:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({actual_size / 1024 / 1024:.0f} MB). Max: {_MAX_UPLOAD_BYTES // 1024 // 1024} MB",
        )

    # 2. Compute doc_hash from file content (SHA256)
    doc_hash = hashlib.sha256(dest.read_bytes()).hexdigest()

    # 3. Skip if already successfully ingested (unless force=True)
    if not force and ledger_db:
        try:
            with ledger_db.session() as session:
                existing = session.query(DocumentLedger).filter_by(doc_hash=doc_hash).first()
                if existing and existing.status == "completed":
                    logger.info(
                        f"Skipping ingestion for {safe_name} — "
                        f"already completed (hash: {doc_hash[:12]}...)"
                    )
                    dest.unlink(missing_ok=True)  # Remove duplicate file
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "already_ingested",
                            "doc_hash": doc_hash,
                            "filename": safe_name,
                        },
                    )
        except Exception as db_err:
            logger.warning(f"Ledger check failed, proceeding with ingestion: {db_err}")

    # 4. Create/update ledger entry with status="processing"
    if ledger_db:
        try:
            with ledger_db.session() as session:
                existing = session.query(DocumentLedger).filter_by(doc_hash=doc_hash).first()
                if existing:
                    existing.status = "processing"
                else:
                    session.add(DocumentLedger(
                        doc_hash=doc_hash,
                        filename=safe_name,
                        status="processing",
                    ))
        except Exception as db_err:
            logger.warning(f"Ledger pre-registration failed: {db_err}")

    # 5. Create tracker and launch background task
    _trackers[doc_hash] = _IngestionTracker()
    asyncio.create_task(_run_ingestion_background(dest, doc_hash))

    return JSONResponse(
        status_code=202,
        content={
            "status": "processing",
            "doc_hash": doc_hash,
            "filename": safe_name,
        },
    )


@router.get("/ingest/{doc_hash}/status")
async def ingest_status_stream(doc_hash: str):
    """
    SSE stream for ingestion progress. Opens a persistent connection and
    pushes a status event when ingestion completes or fails.

    Sends heartbeat comments every 15s to keep the connection alive through
    proxies and load balancers.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        tracker = _trackers.get(doc_hash)

        # No active tracker — fall back to ledger for terminal status
        if tracker is None:
            status = "not_found"
            if ledger_db:
                try:
                    with ledger_db.session() as session:
                        entry = session.query(DocumentLedger).filter_by(doc_hash=doc_hash).first()
                        if entry:
                            status = entry.status
                except Exception:
                    pass
            yield f"data: {json.dumps({'status': status})}\n\n"
            return

        # Already done — send result immediately
        if tracker.done.is_set():
            yield f"data: {json.dumps({'status': tracker.status, 'error': tracker.error})}\n\n"
            return

        # Still processing — stream progress updates and heartbeats
        yield f"data: {json.dumps({'status': 'processing', 'phase': tracker.phase, 'progress': tracker.progress})}\n\n"

        while not tracker.done.is_set():
            # Wait for either: done, progress update, or 15s heartbeat
            done_task = asyncio.create_task(asyncio.shield(tracker.done.wait()))
            update_task = asyncio.create_task(asyncio.shield(tracker.updated.wait()))
            finished, pending = await asyncio.wait(
                {done_task, update_task},
                timeout=15,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

            if tracker.done.is_set():
                break

            if finished:
                # Progress update — send current phase/detail
                yield f"data: {json.dumps({'status': 'processing', 'phase': tracker.phase, 'progress': tracker.progress})}\n\n"
            else:
                # Timeout — heartbeat
                yield ": heartbeat\n\n"

        # Final status with summary stats
        yield f"data: {json.dumps({'status': tracker.status, 'error': tracker.error, 'items_count': tracker.items_count, 'quarantine_count': tracker.quarantine_count})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/documents", response_model=List[DocumentSummary])
async def list_documents():
    """Returns all documents in the ledger."""
    if not ledger_db:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        with ledger_db.session() as session:
            docs = session.query(DocumentLedger).all()
            return [
                DocumentSummary(
                    doc_hash=d.doc_hash,
                    filename=d.filename,
                    status=d.status,
                    upload_date=d.upload_date.isoformat() if d.upload_date else None,
                )
                for d in docs
            ]
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.get("/documents/{doc_hash}", response_model=DocumentSummary)
async def get_document(doc_hash: str):
    """Returns details for a specific document."""
    if not ledger_db:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        with ledger_db.session() as session:
            doc = session.query(DocumentLedger).filter_by(doc_hash=doc_hash).first()
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")
            return DocumentSummary(
                doc_hash=doc.doc_hash,
                filename=doc.filename,
                status=doc.status,
                upload_date=doc.upload_date.isoformat() if doc.upload_date else None,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get document")
