"""
Query Router: Classifies incoming queries for optimal retrieval strategy.

Routes to one of three paths:
- qualitative:  Free-text Q&A → Qdrant hybrid search
- quantitative:  Numeric/comparison queries → Text-to-SQL over metric_facts
- hybrid:        Both paths, merged results

Also detects query locale (en/de/fr) for multilingual expansion.
"""

import logging
from langsmith import traceable

from src.config import SystemConfig
from src.schemas.state import AgentState

logger = logging.getLogger(__name__)

_ROUTER_PROMPT = """You are a financial query classifier. Given a user query, output exactly two lines:

Line 1 - ROUTE: one of [qualitative, quantitative, hybrid]
Line 2 - LOCALE: one of [en, de, fr]

Classification rules:
- "qualitative": Queries about strategies, narratives, trends, summaries, explanations.
  Examples: "What are the key risks?", "Summarize the investment highlights"
- "quantitative": Queries requesting specific numbers, comparisons, rankings, aggregations.
  Examples: "What was revenue in 2024?", "Which company has highest EBITDA margin?"
- "hybrid": Queries that need both numbers AND context/explanation.
  Examples: "Why did revenue decline in Q3?", "Compare margins and explain drivers"

Locale detection:
- "de" if the query is primarily in German
- "fr" if the query is primarily in French
- "en" for English or any other language

Output ONLY the two lines, no explanations."""


@traceable(name="Route Query", run_type="tool")
async def route_query(state: AgentState) -> dict:
    """
    Classifies the query and detects locale.
    Returns partial state update with query_route and query_locale.
    """
    query = state["query"]

    llm = SystemConfig.get_llm(
        model_name=SystemConfig.LAYOUT_MODEL,
        temperature=0.0,
        max_tokens=20,
    )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": _ROUTER_PROMPT},
            {"role": "user", "content": query},
        ])
        lines = response.content.strip().split('\n')

        route = "hybrid"
        locale = "en"
        for line in lines:
            line = line.strip().lower()
            if line.startswith("route:"):
                r = line.split(":", 1)[1].strip()
                if r in ("qualitative", "quantitative", "hybrid"):
                    route = r
            elif line.startswith("locale:"):
                loc = line.split(":", 1)[1].strip()
                if loc in ("en", "de", "fr"):
                    locale = loc

        logger.info(f"Routed query: route={route}, locale={locale}")
        return {"query_route": route, "query_locale": locale}

    except Exception as e:
        logger.warning(f"Router failed, defaulting to hybrid/en: {e}")
        return {"query_route": "hybrid", "query_locale": "en"}
