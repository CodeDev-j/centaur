"""
Visualize Node: Natural language → SQL → Vega-Lite chart spec.

Terminal LangGraph node — the chart IS the answer (skips generate_answer).
Two LLM calls: (1) Text-to-SQL for data, (2) Vega-Lite spec generation.

Data guard: Auto-aggregates if SQL returns >500 rows.
"""

import json
import logging
from typing import List, Dict, Any

from langsmith import traceable

from src.config import SystemConfig
from src.schemas.state import AgentState

logger = logging.getLogger(__name__)

# Module-level driver ref — set by graph.py
_analytics_driver = None


def set_viz_analytics_driver(driver):
    global _analytics_driver
    _analytics_driver = driver


_VIZ_SQL_PROMPT = """You are a financial data visualization analyst. Convert the user's chart request into a PostgreSQL query against the `metric_facts` table.

{ddl}

AVAILABLE SERIES LABELS (use these exact strings in WHERE clauses):
{labels}

CRITICAL RULES:
1. ALWAYS query `resolved_value` for numeric values. It already incorporates magnitude.
2. NEVER write CASE statements for magnitude conversion.
3. Use `series_label` for filtering by metric name. Match EXACTLY against the labels above.
4. Use `period_date` for time-based filtering and ordering (it's a DATE column).
5. Use `source_file` to identify different documents.
6. Include columns needed for the visualization: typically series_label, label, resolved_value, period_date, source_file.
7. ORDER BY period_date or label for temporal charts.
8. Limit to 500 rows maximum.
{doc_filter_hint}

Output ONLY the SQL query, no explanations or markdown."""


MAX_VIZ_ROWS = 500


@traceable(name="Visualize Data", run_type="chain")
async def visualize(state: AgentState) -> dict:
    """
    Terminal node: generates a chart from metric_facts data.

    Steps:
    1. Text-to-SQL to retrieve data
    2. Execute SQL
    3. LLM generates Vega-Lite spec from results
    4. Returns viz_spec, viz_data, viz_sql, viz_title in state
    """
    query = state["query"]
    doc_filter = state.get("doc_filter")

    if _analytics_driver is None:
        logger.warning("No analytics driver for visualization")
        return {
            "final_answer": "Visualization unavailable — no data indexed yet.",
            "viz_spec": None,
        }

    # Step 1: Generate SQL
    ddl = _analytics_driver.get_ddl()
    try:
        labels = _analytics_driver.get_distinct_labels()
    except Exception:
        labels = []
    labels_str = (
        ", ".join(f"'{l}'" for l in labels[:100])
        if labels
        else "(none indexed yet)"
    )

    doc_filter_hint = ""
    if doc_filter:
        doc_filter_hint = (
            f"\nIMPORTANT: Filter to document with doc_hash = '{doc_filter}'"
        )

    sql_prompt = _VIZ_SQL_PROMPT.format(
        ddl=ddl, labels=labels_str, doc_filter_hint=doc_filter_hint
    )

    llm = SystemConfig.get_llm(
        model_name=SystemConfig.REASONING_MODEL,
        temperature=0.0,
        max_tokens=500,
    )

    try:
        sql_response = await llm.ainvoke([
            {"role": "system", "content": sql_prompt},
            {"role": "user", "content": query},
        ])
        sql = sql_response.content.strip()

        # Strip markdown fences
        if sql.startswith("```"):
            lines = sql.split('\n')
            sql = '\n'.join(
                lines[1:-1] if lines[-1].strip() == '```' else lines[1:]
            )
            sql = sql.strip()

        # Safety checks
        if not sql.upper().lstrip().startswith("SELECT"):
            return {
                "final_answer": "Could not generate a valid query for this visualization.",
                "viz_spec": None,
            }

        if ";" in sql.strip().rstrip(";"):
            return {
                "final_answer": "Could not generate a safe query for this visualization.",
                "viz_spec": None,
            }

        logger.info(f"Viz SQL: {sql[:200]}")
        data = _analytics_driver.execute_sql(sql)
        logger.info(f"Viz SQL returned {len(data)} rows")

    except Exception as e:
        logger.error(f"Viz SQL generation/execution failed: {e}")
        return {
            "final_answer": f"Failed to query data for visualization: {str(e)[:100]}",
            "viz_spec": None,
        }

    if not data:
        return {
            "final_answer": "No data found matching your visualization request.",
            "viz_spec": None,
        }

    # Data guard: truncate if too many rows
    if len(data) > MAX_VIZ_ROWS:
        logger.warning(f"Viz data too large ({len(data)} rows), truncating to {MAX_VIZ_ROWS}")
        data = data[:MAX_VIZ_ROWS]

    # Step 3: Generate Vega-Lite spec via shared helper
    from src.workflows.viz_helper import generate_vega_spec, clean_data_for_json

    try:
        spec = await generate_vega_spec(query, data, sql)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Vega-Lite spec: {e}")
        return {
            "final_answer": "Generated data but could not create chart. Here are the raw results.",
            "viz_spec": None,
            "viz_data": data,
            "viz_sql": sql,
        }
    except Exception as e:
        logger.error(f"Vega-Lite spec generation failed: {e}")
        return {
            "final_answer": "Failed to generate visualization specification.",
            "viz_spec": None,
            "viz_data": data,
            "viz_sql": sql,
        }

    viz_title = spec.get("title", query[:60])
    clean_data = clean_data_for_json(data)

    return {
        "final_answer": "",
        "viz_spec": spec,
        "viz_data": clean_data,
        "viz_sql": sql,
        "viz_title": viz_title if isinstance(viz_title, str) else str(viz_title),
    }
