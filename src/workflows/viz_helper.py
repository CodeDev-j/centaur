"""
Shared Vega-Lite Spec Generation.

Extracted from visualize.py for reuse by both the chat visualize node
and the Prompt Studio executor (output_format=chart).
"""

import json
import logging
from typing import Any, Dict, List, Optional

from src.config import SystemConfig

logger = logging.getLogger(__name__)

_VIZ_SPEC_PROMPT = """Given this SQL result data and the user's visualization request, generate a Vega-Lite v5 specification.

User request: {query}
SQL executed: {sql}
Data (first 10 rows): {sample_data}
Total rows: {row_count}
Column names and types: {column_info}

Rules:
- Use "$schema": "https://vega.github.io/schema/vega-lite/v5.json"
- Data will be injected separately via "data.values" — do NOT include a "data" field
- Choose appropriate mark type: bar (comparisons), line (trends), point (scatter), area (cumulative)
- Use proper temporal axis formatting if period_date is present
- Format resolved_value as numbers (not scientific notation)
- Include a descriptive title derived from the user's request
- Dark theme colors: background "#111111", text "#ededed", gridColor "#333333"
- Configure axis labels, tooltips, and legend as appropriate
- Width: 480, Height: 300 (container will resize)
- If multiple series, use color encoding on series_label or source_file

Return ONLY the Vega-Lite JSON spec, no explanation or markdown fences."""


def clean_data_for_json(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Serialize data values for JSON transport (handle dates etc.)."""
    clean = []
    for row in data:
        clean_row = {}
        for k, v in row.items():
            if hasattr(v, 'isoformat'):
                clean_row[k] = v.isoformat()
            else:
                clean_row[k] = v
        clean.append(clean_row)
    return clean


async def generate_vega_spec(
    user_query: str,
    data: List[Dict[str, Any]],
    sql: str,
    model_id: Optional[str] = None,
) -> dict:
    """
    Generate a Vega-Lite v5 spec from SQL result data and user intent.

    Returns the parsed spec dict.
    Raises json.JSONDecodeError if the LLM output isn't valid JSON.
    Raises Exception for other generation failures.
    """
    sample = data[:10]
    column_info = {}
    if data:
        for key, val in data[0].items():
            column_info[key] = type(val).__name__

    spec_prompt = _VIZ_SPEC_PROMPT.format(
        query=user_query,
        sql=sql,
        sample_data=json.dumps(sample, default=str, indent=2),
        row_count=len(data),
        column_info=json.dumps(column_info),
    )

    llm = SystemConfig.get_llm(
        model_name=model_id or SystemConfig.GENERATION_MODEL,
        temperature=0.0,
        max_tokens=2000,
    )

    response = await llm.ainvoke([
        {"role": "system", "content": spec_prompt},
        {"role": "user", "content": "Generate the Vega-Lite specification."},
    ])
    spec_text = response.content.strip()

    # Strip markdown fences
    if spec_text.startswith("```"):
        lines = spec_text.split('\n')
        spec_text = '\n'.join(
            lines[1:-1] if lines[-1].strip() == '```' else lines[1:]
        )
        spec_text = spec_text.strip()

    spec = json.loads(spec_text)
    logger.info("Vega-Lite spec generated successfully")
    return spec
