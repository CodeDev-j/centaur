"""
Generation Node: Produces cited answers from retrieved context.

Uses GPT-4.1 with structured output to guarantee deterministic citations.
The LLM returns segments with source_ids; we control marker formatting.
"""

import json
import logging
from typing import List

from pydantic import BaseModel, Field
from langsmith import traceable

from src.config import SystemConfig
from src.schemas.state import AgentState
from src.retrieval.sidecar import assemble_cited_answer

logger = logging.getLogger(__name__)


# ==============================================================================
# Structured Output Schema
# ==============================================================================

class CitedSegment(BaseModel):
    """A sentence or short passage with its supporting source IDs."""
    text: str = Field(description="A sentence or short passage of the answer.")
    source_ids: List[int] = Field(
        default_factory=list,
        description=(
            "List of source IDs (the integer from the source id= attribute) "
            "that support this text. Empty list if no citation needed."
        ),
    )


class StructuredAnswer(BaseModel):
    """Answer broken into citable segments."""
    segments: List[CitedSegment] = Field(
        description=(
            "The answer broken into segments. Each segment is a sentence or "
            "short passage with optional source citations."
        )
    )


# ==============================================================================
# Prompts
# ==============================================================================

_GENERATION_PROMPT = """You are a senior financial analyst assistant. Answer the user's question using ONLY the provided sources.

RULES:
1. Break your answer into segments (sentences or short passages).
2. For each segment, provide the source IDs (the integer from the source id= attribute) that support that claim.
3. Only cite sources that directly support the specific claim. Leave source_ids empty for transitional text.
4. If the sources don't contain enough information, say so explicitly.
5. When SQL results are provided, incorporate the specific numbers into your answer.
6. Respond in the same language as the user's query.
7. Be precise with numbers — include currency, magnitude, and time period.

SOURCES:
{context}

{sql_section}"""


# ==============================================================================
# Nodes
# ==============================================================================

@traceable(name="Generate Answer", run_type="chain")
async def generate_answer(state: AgentState) -> dict:
    """
    Generates a cited answer using structured output.
    Returns both the assembled answer text and resolved citations.
    """
    context_str = state.get("context_str", "")
    sql_result = state.get("sql_result", [])
    sidecar_map = state.get("sidecar_map", {})
    query = state["query"]

    # Build SQL section if we have results
    sql_section = ""
    if sql_result:
        sql_section = "SQL QUERY RESULTS:\n"
        for row in sql_result[:20]:
            sql_section += json.dumps(row, default=str) + "\n"

    prompt = _GENERATION_PROMPT.format(
        context=context_str or "(No document sources retrieved)",
        sql_section=sql_section,
    )

    llm = SystemConfig.get_llm(
        model_name=SystemConfig.GENERATION_MODEL,
        temperature=0.1,
        max_tokens=2000,
    )

    try:
        # Use structured output for deterministic citations
        structured_llm = llm.with_structured_output(StructuredAnswer)
        result: StructuredAnswer = await structured_llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ])

        # Assemble answer text with validated, consecutively-numbered citations
        segments = [seg.model_dump() for seg in result.segments]
        answer, citations = assemble_cited_answer(segments, sidecar_map)

        logger.info(
            f"Structured generation: {len(result.segments)} segments → "
            f"{len(citations)} citations"
        )

    except Exception as e:
        logger.error(f"Structured generation failed, falling back to plain text: {e}")
        # Fallback: plain text generation without structured citations
        try:
            response = await llm.ainvoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ])
            answer = response.content.strip()
            citations = []
        except Exception as e2:
            logger.error(f"Fallback generation also failed: {e2}")
            answer = "I encountered an error generating the answer. Please try again."
            citations = []

    return {
        "final_answer": answer,
        "citations": citations,
    }


@traceable(name="Verify and Finalize", run_type="chain")
async def verify_and_finalize(state: AgentState) -> dict:
    """
    Post-generation finalization.
    Citations are already resolved by generate_answer via structured output.
    This node computes the confidence score.
    """
    citations = state.get("citations", [])
    return {
        "confidence": 0.8 if citations else 0.4,
    }
