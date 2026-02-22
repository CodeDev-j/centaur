"""
FastAPI Application: The Centaur API Server.

Serves the chat, ingestion, and document viewer endpoints.
Initializes the LangGraph workflow and all drivers on startup.
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import SystemConfig

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Centaur API",
    description="Financial Document Intelligence Platform",
    version="0.1.0",
)

# CORS for frontend
_CORS_ORIGINS = os.getenv("CENTAUR_CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["content-type", "authorization"],
)


# ---------------------------------------------------------------------------
# Global exception handler — prevents unhandled crashes from leaking tracebacks
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.on_event("startup")
async def startup():
    """Initialize the LangGraph workflow and wire into routes."""
    from src.workflows.graph import build_graph
    from src.ingestion.pipeline import IngestionPipeline
    from src.storage.vector_driver import VectorDriver
    from src.api.routes.chat import set_graph
    from src.api.routes.ingestion import set_pipeline
    from src.api.routes.documents import set_vector_driver
    from src.workflows.prompt_executor import (
        set_retrieval_pipeline as set_executor_pipeline,
        set_analytics_driver as set_executor_analytics,
    )

    from src.audit.engine import AuditEngine
    from src.api.routes.audit import set_audit_engine
    from src.api.routes.export import set_export_drivers

    logger.info("Initializing Centaur API...")

    # Build LangGraph workflow (initializes all drivers internally)
    graph = build_graph()
    set_graph(graph)

    # Initialize ingestion pipeline for upload endpoint
    pipeline = IngestionPipeline()
    set_pipeline(pipeline)

    # Initialize vector driver for chunk inspection endpoints
    vector_driver = VectorDriver()
    set_vector_driver(vector_driver)

    # Wire prompt executor with retrieval pipeline + analytics driver
    # (reuses the same instances created by build_graph)
    from src.workflows.nodes.retrieve import _retrieval_pipeline, _analytics_driver
    if _retrieval_pipeline:
        set_executor_pipeline(_retrieval_pipeline)
    if _analytics_driver:
        set_executor_analytics(_analytics_driver)

    # Initialize audit engine
    audit_engine = AuditEngine()
    set_audit_engine(audit_engine)

    # Wire export drivers (analytics for data, audit for summary sheet)
    if _analytics_driver:
        set_export_drivers(_analytics_driver, audit_engine)

    logger.info("Centaur API ready.")


@app.on_event("shutdown")
async def shutdown():
    """Gracefully close drivers and connection pools."""
    logger.info("Shutting down Centaur API...")
    from src.storage.db_driver import ledger_db
    from src.api.routes.documents import _vector_driver, _thread_pool

    if _thread_pool:
        _thread_pool.shutdown(wait=False)

    if _vector_driver and hasattr(_vector_driver, "client"):
        try:
            _vector_driver.client.close()
        except Exception:
            pass

    if ledger_db and hasattr(ledger_db, "engine"):
        try:
            ledger_db.engine.dispose()
        except Exception:
            pass

    logger.info("Centaur API shutdown complete.")


# Register routes
from src.api.routes import chat, ingestion, documents, prompts, workflows, audit, export, tools  # noqa: E402

app.include_router(chat.router)
app.include_router(ingestion.router)
app.include_router(documents.router)
app.include_router(prompts.router)
app.include_router(workflows.router)
app.include_router(audit.router)
app.include_router(export.router)
app.include_router(tools.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    SystemConfig._check_defaults()
    uvicorn.run(
        "src.api.main:app",
        host=SystemConfig.API_HOST,
        port=SystemConfig.API_PORT,
        reload=os.getenv("CENTAUR_DEV", "").lower() in ("1", "true"),
    )
