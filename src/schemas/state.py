"""
Agent State: The Short-Term Memory of the Brain.

Tracks the full lifecycle of a query from routing through retrieval
to generation and citation verification.
"""

from typing import List, Annotated, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator

from src.schemas.retrieval import RetrievedChunk
from src.schemas.citation import Citation


class AgentState(TypedDict):
    """
    LangGraph state flowing through all nodes.
    Each node reads what it needs and writes its outputs.
    """
    # --- Conversation ---
    messages: Annotated[List[BaseMessage], operator.add]

    # --- Query Understanding ---
    query: str
    query_locale: str           # "en", "de", "fr"
    query_route: str            # "qualitative", "quantitative", "hybrid"

    # --- Retrieval Outputs ---
    retrieval_chunks: List[RetrievedChunk]
    sql_result: List[Dict[str, Any]]
    query_expanded: str

    # --- Citation Context ---
    sidecar_map: Dict[int, RetrievedChunk]
    context_str: str

    # --- Filters ---
    doc_filter: Optional[str]          # doc_hash to scope retrieval (None = all)

    # --- Generation Outputs ---
    final_answer: str
    citations: List[Citation]
    confidence: float
