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
    from src.api.routes.chat import set_graph
    from src.api.routes.ingestion import set_pipeline

    logger.info("Initializing Centaur API...")

    # Build LangGraph workflow (initializes all drivers internally)
    graph = build_graph()
    set_graph(graph)

    # Initialize ingestion pipeline for upload endpoint
    pipeline = IngestionPipeline()
    set_pipeline(pipeline)

    logger.info("Centaur API ready.")


# Register routes
from src.api.routes import chat, ingestion, documents  # noqa: E402

app.include_router(chat.router)
app.include_router(ingestion.router)
app.include_router(documents.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=SystemConfig.API_HOST,
        port=SystemConfig.API_PORT,
        reload=True,
    )
