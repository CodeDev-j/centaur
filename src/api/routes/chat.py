"""
Chat Routes: Query the knowledge base with cited answers.

POST /api/v1/chat         — JSON response
POST /api/v1/chat/stream  — SSE stream for CopilotKit
"""

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langsmith import traceable

from src.api.schemas import ChatRequest, ChatResponse
from src.schemas.state import AgentState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Module-level graph reference — set by main.py
_graph = None


def set_graph(graph):
    global _graph
    _graph = graph


def _build_initial_state(request: ChatRequest) -> dict:
    """Constructs the initial AgentState for the graph."""
    return {
        "messages": [],
        "query": request.query,
        "query_locale": request.locale or "en",
        "query_route": "",
        "retrieval_chunks": [],
        "sql_result": [],
        "query_expanded": "",
        "sidecar_map": {},
        "context_str": "",
        "final_answer": "",
        "citations": [],
        "confidence": 0.0,
    }


@router.post("", response_model=ChatResponse)
@traceable(name="Chat Endpoint", run_type="chain")
async def chat(request: ChatRequest) -> ChatResponse:
    """Synchronous chat: runs full graph and returns complete response."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    initial_state = _build_initial_state(request)

    try:
        result = await _graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        answer=result.get("final_answer", ""),
        citations=result.get("citations", []),
        query_route=result.get("query_route", "hybrid"),
        query_expanded=result.get("query_expanded", ""),
        sql_result=result.get("sql_result", []),
        confidence=result.get("confidence", 0.0),
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    SSE stream for CopilotKit integration.
    Streams graph execution events as Server-Sent Events.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    initial_state = _build_initial_state(request)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for event in _graph.astream_events(initial_state, version="v2"):
                event_type = event.get("event", "")
                name = event.get("name", "")

                if event_type == "on_chain_end" and name == "generate_answer":
                    output = event.get("data", {}).get("output", {})
                    answer = output.get("final_answer", "")
                    if answer:
                        yield f"data: {json.dumps({'type': 'answer', 'content': answer})}\n\n"
                    citations = output.get("citations", [])
                    if citations:
                        yield f"data: {json.dumps({'type': 'citations', 'content': [c.model_dump() for c in citations]}, default=str)}\n\n"

                elif event_type == "on_chain_end" and name == "route_query":
                    output = event.get("data", {}).get("output", {})
                    yield f"data: {json.dumps({'type': 'route', 'content': output})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
