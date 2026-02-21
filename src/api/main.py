"""
FastAPI Application: The Centaur API Server.

Serves the chat, ingestion, and document viewer endpoints.
Initializes the LangGraph workflow and all drivers on startup.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import SystemConfig

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Centaur API",
    description="Financial Document Intelligence Platform",
    version="0.1.0",
)

# CORS for frontend (Next.js on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# Register routes
from src.api.routes import chat, ingestion, documents, prompts, workflows, audit, export  # noqa: E402

app.include_router(chat.router)
app.include_router(ingestion.router)
app.include_router(documents.router)
app.include_router(prompts.router)
app.include_router(workflows.router)
app.include_router(audit.router)
app.include_router(export.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import os
    import uvicorn
    SystemConfig._check_defaults()
    uvicorn.run(
        "src.api.main:app",
        host=SystemConfig.API_HOST,
        port=SystemConfig.API_PORT,
        reload=os.getenv("CENTAUR_DEV", "").lower() in ("1", "true"),
    )
