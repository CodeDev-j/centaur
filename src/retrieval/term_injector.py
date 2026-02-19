"""
Term Injector: Multilingual Query Expansion.

Fixes the cross-lingual BM25 gap: German "Umsatz" won't match English "Revenue"
in sparse search. This module expands the query with multilingual synonyms
and fuzzy-matches against known series labels from Postgres.

Flow: user query → LLM expansion (synonyms + translations) → label fuzzy match → expanded query
"""

import logging
from typing import List, Optional

from langsmith import traceable

from src.config import SystemConfig

logger = logging.getLogger(__name__)

# System prompt for query expansion
_EXPANSION_PROMPT = """You are a multilingual financial term expander.
Given a user query and locale, produce a single line of space-separated search terms
that includes:
1. The original query terms
2. English synonyms for any financial concepts
3. German translations if the query is in English (and vice versa)
4. French translations if relevant
5. Common abbreviations (EBITDA, Rev, OpEx, etc.)

Rules:
- Output ONLY the expanded terms on a single line, no explanations
- Include both formal and informal variants (Revenue, Umsatz, Erlöse, top-line, sales)
- For metrics, include the unit-less concept (revenue, growth, margin)
- Maximum 30 terms

Example:
Query: "Umsatzentwicklung Q3 2024"
Locale: de
Output: Umsatzentwicklung Umsatz Revenue sales top-line Erlöse growth Wachstum Q3 2024 quarterly third quarter drittes Quartal"""


class TermInjector:
    """Expands queries with multilingual synonyms and known label matches."""

    def __init__(self, analytics_driver=None):
        self.llm = SystemConfig.get_llm(
            model_name=SystemConfig.LAYOUT_MODEL,  # fast, cheap model
            temperature=0.0,
            max_tokens=200,
        )
        self.analytics_driver = analytics_driver
        self._label_cache: Optional[List[str]] = None

    def _get_known_labels(self) -> List[str]:
        """Cached fetch of distinct series labels from Postgres."""
        if self._label_cache is None and self.analytics_driver:
            try:
                self._label_cache = self.analytics_driver.get_distinct_labels()
            except Exception as e:
                logger.warning(f"Failed to fetch labels: {e}")
                self._label_cache = []
        return self._label_cache or []

    def _fuzzy_match_labels(self, query: str, top_n: int = 5) -> List[str]:
        """
        Simple substring matching of query terms against known labels.
        Not a full fuzzy search — just checks if any query word appears in a label.
        """
        labels = self._get_known_labels()
        if not labels:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())
        matched = []
        for label in labels:
            label_lower = label.lower()
            if any(w in label_lower for w in query_words):
                matched.append(label)
            if len(matched) >= top_n:
                break
        return matched

    @traceable(name="Query Expansion", run_type="tool")
    async def expand(self, query: str, locale: str = "en") -> str:
        """
        Expands a user query with multilingual synonyms and label matches.
        Returns a single string with all terms (original + expanded).
        """
        # 1. LLM expansion
        try:
            messages = [
                {"role": "system", "content": _EXPANSION_PROMPT},
                {"role": "user", "content": f"Query: \"{query}\"\nLocale: {locale}"},
            ]
            response = await self.llm.ainvoke(messages)
            expanded = response.content.strip()
        except Exception as e:
            logger.warning(f"LLM expansion failed, using original: {e}")
            expanded = query

        # 2. Label injection
        matched_labels = self._fuzzy_match_labels(query)
        if matched_labels:
            expanded = expanded + " " + " ".join(matched_labels)

        logger.info(f"Expanded query: '{query}' → '{expanded[:100]}...'")
        return expanded

    def invalidate_cache(self):
        """Clear the label cache (call after re-ingestion)."""
        self._label_cache = None
