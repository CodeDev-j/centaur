"""
Retrieval Nodes: LangGraph nodes for qualitative, quantitative, and hybrid retrieval.

Each node reads from AgentState and writes retrieval results back.
"""

import logging
from typing import Dict, Any, Optional

from langsmith import traceable
from qdrant_client import models

from src.schemas.state import AgentState
from src.retrieval.qdrant import RetrievalPipeline
from src.retrieval.sidecar import build_context_with_citations

logger = logging.getLogger(__name__)

# Module-level reference — set by graph.py during initialization
_retrieval_pipeline: RetrievalPipeline = None
_analytics_driver = None


def set_retrieval_pipeline(pipeline: RetrievalPipeline):
    global _retrieval_pipeline
    _retrieval_pipeline = pipeline


def set_analytics_driver(driver):
    global _analytics_driver
    _analytics_driver = driver


def _build_doc_filter(doc_hash: Optional[str]) -> Optional[models.Filter]:
    """Builds Qdrant filter for doc_hash if provided."""
    if not doc_hash:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="doc_hash",
                match=models.MatchValue(value=doc_hash),
            )
        ]
    )


@traceable(name="Retrieve Qualitative", run_type="chain")
async def retrieve_qualitative(state: AgentState) -> dict:
    """Qdrant hybrid search for narrative/qualitative queries."""
    result = await _retrieval_pipeline.retrieve(
        query=state["query"],
        locale=state.get("query_locale", "en"),
        top_k=10,
        filters=_build_doc_filter(state.get("doc_filter")),
    )

    context_str, sidecar_map = build_context_with_citations(result.chunks)

    return {
        "retrieval_chunks": result.chunks,
        "query_expanded": result.query_expanded,
        "context_str": context_str,
        "sidecar_map": sidecar_map,
        "sql_result": [],
    }


@traceable(name="Retrieve Quantitative", run_type="chain")
async def retrieve_quantitative(state: AgentState) -> dict:
    """Text-to-SQL over Postgres metric_facts for numeric queries."""
    from src.workflows.nodes.financial_math import generate_and_execute_sql

    sql_result = await generate_and_execute_sql(
        query=state["query"],
        analytics_driver=_analytics_driver,
    )

    # Also do a lightweight vector search for context around the numbers
    result = await _retrieval_pipeline.retrieve(
        query=state["query"],
        locale=state.get("query_locale", "en"),
        top_k=5,
        filters=_build_doc_filter(state.get("doc_filter")),
    )

    context_str, sidecar_map = build_context_with_citations(result.chunks)

    return {
        "retrieval_chunks": result.chunks,
        "sql_result": sql_result,
        "query_expanded": result.query_expanded,
        "context_str": context_str,
        "sidecar_map": sidecar_map,
    }


@traceable(name="Retrieve Hybrid", run_type="chain")
async def retrieve_hybrid(state: AgentState) -> dict:
    """Both qualitative and quantitative retrieval, merged."""
    from src.workflows.nodes.financial_math import generate_and_execute_sql

    # Run both in parallel-ish (sequential here, but both are fast)
    result = await _retrieval_pipeline.retrieve(
        query=state["query"],
        locale=state.get("query_locale", "en"),
        top_k=10,
        filters=_build_doc_filter(state.get("doc_filter")),
    )

    sql_result = await generate_and_execute_sql(
        query=state["query"],
        analytics_driver=_analytics_driver,
    )

    context_str, sidecar_map = build_context_with_citations(result.chunks)

    return {
        "retrieval_chunks": result.chunks,
        "sql_result": sql_result,
        "query_expanded": result.query_expanded,
        "context_str": context_str,
        "sidecar_map": sidecar_map,
    }
