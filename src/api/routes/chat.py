"""
Chat Routes: Query the knowledge base with cited answers.

POST /api/v1/chat         — JSON response
POST /api/v1/chat/stream  — SSE stream with progress events
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
    # Build LangChain messages from conversation history (multi-turn)
    from langchain_core.messages import HumanMessage, AIMessage
    messages = []
    for msg in (request.messages or []):
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    return {
        "messages": messages,
        "query": request.query,
        "query_locale": request.locale or "en",
        "query_route": "",
        "doc_filter": request.doc_filter or None,
        "retrieval_chunks": [],
        "sql_result": [],
        "query_expanded": "",
        "sidecar_map": {},
        "context_str": "",
        "final_answer": "",
        "citations": [],
        "confidence": 0.0,
    }


def _serialize_citation(c) -> dict:
    """Safely serialize a citation — handles Pydantic objects, dicts, and unknowns."""
    if hasattr(c, "model_dump"):
        return c.model_dump()
    elif isinstance(c, dict):
        return c
    else:
        return {"blurb": str(c), "source_file": "", "page_number": 0}


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
    SSE stream using astream(stream_mode="updates").
    Each node yields {node_name: output_dict} — much simpler than astream_events.
    Events: route, answer, citations, done, error
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    initial_state = _build_initial_state(request)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in _graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if node_name == "route_query":
                        yield f"data: {json.dumps({'type': 'route', 'content': node_output}, default=str)}\n\n"

                    elif node_name == "generate_answer":
                        answer = node_output.get("final_answer", "")
                        if answer:
                            yield f"data: {json.dumps({'type': 'answer', 'content': answer})}\n\n"

                        # Serialize citations with isolated error handling
                        raw_citations = node_output.get("citations", [])
                        if raw_citations:
                            try:
                                serialized = [_serialize_citation(c) for c in raw_citations]
                                yield f"data: {json.dumps({'type': 'citations', 'content': serialized}, default=str)}\n\n"
                            except Exception as ce:
                                logger.error(f"Citation serialization failed: {ce}", exc_info=True)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
