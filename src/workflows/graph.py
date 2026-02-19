"""
LangGraph Workflow: The Brain.

Assembles the query processing pipeline:
  START → route_query → {qualitative|quantitative|hybrid} → generate_answer → verify → END

All nodes are async and traced via LangSmith.
"""

import logging

from langgraph.graph import StateGraph, END

from src.schemas.state import AgentState
from src.workflows.router import route_query
from src.workflows.nodes.retrieve import (
    retrieve_qualitative,
    retrieve_quantitative,
    retrieve_hybrid,
    set_retrieval_pipeline,
    set_analytics_driver,
)
from src.workflows.nodes.generate import generate_answer, verify_and_finalize

from src.storage.vector_driver import VectorDriver
from src.storage.analytics_driver import AnalyticsDriver
from src.retrieval.term_injector import TermInjector
from src.retrieval.qdrant import RetrievalPipeline

logger = logging.getLogger(__name__)


def _route_decision(state: AgentState) -> str:
    """Conditional edge: routes to the correct retrieval node."""
    route = state.get("query_route", "hybrid")
    if route == "qualitative":
        return "retrieve_qualitative"
    elif route == "quantitative":
        return "retrieve_quantitative"
    else:
        return "retrieve_hybrid"


def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph workflow.
    Initializes all drivers and wires them into the nodes.
    """
    # Initialize drivers
    vector_driver = VectorDriver()
    analytics_driver = AnalyticsDriver()
    term_injector = TermInjector(analytics_driver=analytics_driver)
    retrieval_pipeline = RetrievalPipeline(
        vector_driver=vector_driver,
        term_injector=term_injector,
    )

    # Wire drivers into retrieval nodes
    set_retrieval_pipeline(retrieval_pipeline)
    set_analytics_driver(analytics_driver)

    # Build graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("route_query", route_query)
    graph.add_node("retrieve_qualitative", retrieve_qualitative)
    graph.add_node("retrieve_quantitative", retrieve_quantitative)
    graph.add_node("retrieve_hybrid", retrieve_hybrid)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_and_finalize", verify_and_finalize)

    # Set entry point
    graph.set_entry_point("route_query")

    # Conditional routing
    graph.add_conditional_edges(
        "route_query",
        _route_decision,
        {
            "retrieve_qualitative": "retrieve_qualitative",
            "retrieve_quantitative": "retrieve_quantitative",
            "retrieve_hybrid": "retrieve_hybrid",
        },
    )

    # All retrieval paths converge to generation
    graph.add_edge("retrieve_qualitative", "generate_answer")
    graph.add_edge("retrieve_quantitative", "generate_answer")
    graph.add_edge("retrieve_hybrid", "generate_answer")

    # Generation → verification → END
    graph.add_edge("generate_answer", "verify_and_finalize")
    graph.add_edge("verify_and_finalize", END)

    logger.info("LangGraph workflow compiled successfully.")
    return graph.compile()
