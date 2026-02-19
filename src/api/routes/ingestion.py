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
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langsmith import traceable

from src.api.schemas import DocumentSummary
from src.config import SystemPaths
from src.storage.db_driver import ledger_db, DocumentLedger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["ingestion"])

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
    done: asyncio.Event = field(default_factory=asyncio.Event)


_trackers: Dict[str, _IngestionTracker] = {}

TRACKER_CLEANUP_DELAY_S = 60  # Remove tracker 60s after completion


async def _run_ingestion_background(file_path: Path, doc_hash: str):
    """
    Background task: runs the full ingestion pipeline, updates ledger,
    and signals the SSE tracker on completion.
    """
    tracker = _trackers.get(doc_hash)

    try:
        await _pipeline.run(file_path)

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
        logger.error(f"Background ingestion failed for {file_path.name}: {e}")

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
async def ingest(file: UploadFile = File(...)):
    """
    Upload a file for ingestion. Returns 202 immediately.
    Subscribe to GET /ingest/{doc_hash}/status for SSE progress updates.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # 1. Save uploaded file to inputs directory
    dest = SystemPaths.INPUTS / file.filename
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # 2. Compute doc_hash for the ledger entry
    doc_hash = hashlib.sha256(dest.read_bytes()).hexdigest()

    # 3. Create ledger entry with status="processing" so it appears in the list
    if ledger_db:
        try:
            with ledger_db.session() as session:
                existing = session.query(DocumentLedger).filter_by(doc_hash=doc_hash).first()
                if existing:
                    existing.status = "processing"
                else:
                    session.add(DocumentLedger(
                        doc_hash=doc_hash,
                        filename=file.filename,
                        status="processing",
                    ))
        except Exception as db_err:
            logger.warning(f"Ledger pre-registration failed: {db_err}")

    # 4. Create tracker and launch background task
    _trackers[doc_hash] = _IngestionTracker()
    asyncio.create_task(_run_ingestion_background(dest, doc_hash))

    return JSONResponse(
        status_code=202,
        content={
            "status": "processing",
            "doc_hash": doc_hash,
            "filename": file.filename,
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

        # Still processing — wait with heartbeats
        yield f"data: {json.dumps({'status': 'processing'})}\n\n"

        while not tracker.done.is_set():
            try:
                await asyncio.wait_for(
                    asyncio.shield(tracker.done.wait()),
                    timeout=15,
                )
            except asyncio.TimeoutError:
                # Heartbeat: SSE comment line (colon prefix), not a data event
                yield ": heartbeat\n\n"

        # Final status
        yield f"data: {json.dumps({'status': tracker.status, 'error': tracker.error})}\n\n"

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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))
