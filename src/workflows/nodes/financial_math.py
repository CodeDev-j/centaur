"""
Financial Math: Text-to-SQL over Postgres metric_facts.

Converts natural language queries into SQL against the metric_facts table.
Key safety feature: queries ALWAYS use `resolved_value` (pre-computed float).
The LLM NEVER generates CASE statements for magnitude conversion.
"""

import logging
from typing import List, Dict, Any

from langsmith import traceable

from src.config import SystemConfig

logger = logging.getLogger(__name__)

_TEXT_TO_SQL_PROMPT = """You are a financial SQL analyst. Convert the user's question into a PostgreSQL query against the `metric_facts` table.

{ddl}

AVAILABLE SERIES LABELS (use these exact strings in WHERE clauses):
{labels}

CRITICAL RULES:
1. ALWAYS query `resolved_value` for numeric comparisons. It already incorporates magnitude.
   Example: "revenue > $1B" → WHERE resolved_value > 1000000000
2. NEVER write CASE statements for magnitude conversion.
3. Use `series_label` for filtering by metric name. Match EXACTLY against the labels above.
4. Use `period_date` for time-based filtering (it's a DATE column).
5. Use `source_file` to filter by document.
6. Limit results to 50 rows maximum.
7. Only SELECT columns that are useful for answering the question.

Output ONLY the SQL query, no explanations or markdown."""


@traceable(name="Text-to-SQL", run_type="tool")
async def generate_and_execute_sql(
    query: str,
    analytics_driver,
) -> List[Dict[str, Any]]:
    """
    Generates SQL from natural language and executes against metric_facts.
    Returns the result rows as a list of dicts.
    """
    if analytics_driver is None:
        logger.warning("No analytics driver available for SQL queries.")
        return []

    # Get schema and vocabulary
    ddl = analytics_driver.get_ddl()
    try:
        labels = analytics_driver.get_distinct_labels()
    except Exception:
        labels = []

    labels_str = ", ".join(f"'{l}'" for l in labels[:100]) if labels else "(none indexed yet)"

    prompt = _TEXT_TO_SQL_PROMPT.format(ddl=ddl, labels=labels_str)

    llm = SystemConfig.get_llm(
        model_name=SystemConfig.REASONING_MODEL,
        temperature=0.0,
        max_tokens=500,
    )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ])
        sql = response.content.strip()

        # Strip markdown code fences if present
        if sql.startswith("```"):
            lines = sql.split('\n')
            sql = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
            sql = sql.strip()

        # Safety: only allow SELECT queries
        if not sql.upper().lstrip().startswith("SELECT"):
            logger.warning(f"Non-SELECT SQL generated, refusing: {sql[:100]}")
            return []

        logger.info(f"Generated SQL: {sql[:200]}")
        result = analytics_driver.execute_sql(sql)
        logger.info(f"SQL returned {len(result)} rows")
        return result

    except Exception as e:
        logger.error(f"Text-to-SQL failed: {e}")
        return []
