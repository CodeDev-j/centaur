"""
Prompt Executor: Composable 2-Step Execution Pipeline for Studio.

Two orthogonal axes:
  Context Source: documents | metrics_db | none
  Output Format:  text | json | chart | table

Pipeline: _fetch_context() → _generate_output() via ContextResult dataclass.

NEVER routes through the chat router. Each step returns a StepOutput.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jinja2 import TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment
from langsmith import traceable
from pydantic import BaseModel

from src.config import SystemConfig

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Output Types
# ══════════════════════════════════════════════════════════════════

@dataclass
class StepOutput:
    """
    Universal output from any execution pipeline.
    - text: raw LLM response (always available)
    - structured: parsed JSON dict (if output_schema was provided and parse succeeded)
    - metadata: execution telemetry (model, tokens, duration, citations, etc.)
    """
    text: str = ""
    structured: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextResult:
    """
    Intermediate result from _fetch_context().
    Carries both the LLM-injectable context string and raw data for audit/charts.
    """
    context_str: str = ""
    chunks_retrieved: int = 0
    cache_hit: bool = False
    raw_chunks: list = field(default_factory=list)
    sql_rows: Optional[list] = None
    sql_query: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# Template Rendering (Jinja2)
# ══════════════════════════════════════════════════════════════════

_sandbox_env = SandboxedEnvironment()


def render_template(template_str: str, variables: Dict[str, Any]) -> str:
    """
    Render a Jinja2 template with the given variables.
    Uses SandboxedEnvironment to prevent SSTI (Server-Side Template Injection).
    Returns the rendered string, or the original template if rendering fails.
    """
    try:
        tmpl = _sandbox_env.from_string(template_str)
        return tmpl.render(**variables)
    except TemplateSyntaxError as e:
        logger.warning(f"Jinja2 template syntax error: {e}")
        return template_str
    except Exception as e:
        logger.warning(f"Template rendering failed: {e}")
        return template_str


# ══════════════════════════════════════════════════════════════════
# Module-Level References & Retrieval Cache
# ══════════════════════════════════════════════════════════════════

_retrieval_pipeline = None
_analytics_driver = None

# Cache key includes search strategy: "{prompt_id}:{doc_filter}:{sorted_strategies}"
_retrieval_cache: Dict[str, tuple] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes
_MAX_CACHE_SIZE = 256


def _cache_key(prompt_id: str, doc_filter: Optional[str], search_strategy: List[str]) -> str:
    sorted_strats = ",".join(sorted(search_strategy))
    return f"{prompt_id}:{doc_filter or 'all'}:{sorted_strats}"


def _get_cached_context(key: str) -> Optional[ContextResult]:
    if key in _retrieval_cache:
        ctx, ts = _retrieval_cache[key]
        if time.time() - ts < _CACHE_TTL_SECONDS:
            return ctx
        del _retrieval_cache[key]
    return None


def _set_cached_context(key: str, context: ContextResult):
    if len(_retrieval_cache) >= _MAX_CACHE_SIZE:
        oldest_key = min(_retrieval_cache, key=lambda k: _retrieval_cache[k][1])
        del _retrieval_cache[oldest_key]
    _retrieval_cache[key] = (context, time.time())


def set_retrieval_pipeline(pipeline):
    global _retrieval_pipeline
    _retrieval_pipeline = pipeline


def set_analytics_driver(driver):
    global _analytics_driver
    _analytics_driver = driver


# ══════════════════════════════════════════════════════════════════
# Step 1: Fetch Context
# ══════════════════════════════════════════════════════════════════

_MAX_SQL_ROWS_FOR_LLM = 50  # Context window guard


@traceable(name="Fetch Context", run_type="chain")
async def _fetch_context(
    source: str,
    rendered: str,
    search_strategy: List[str],
    doc_filter: Optional[str] = None,
    prompt_id: Optional[str] = None,
    force_retrieve: bool = False,
) -> ContextResult:
    """
    Fetch context based on the source axis.
    - none: empty context
    - documents: multi-strategy retrieval (semantic, numeric, or both)
    - metrics_db: Text-to-SQL → format first 50 rows as markdown
    """
    if source == "none":
        return ContextResult()

    # Check cache (for documents and metrics_db)
    cache_k = _cache_key(prompt_id, doc_filter, search_strategy) if prompt_id else None
    if cache_k and not force_retrieve:
        cached = _get_cached_context(cache_k)
        if cached:
            logger.info(f"Context cache HIT for key={cache_k}")
            cached.cache_hit = True
            return cached

    if source == "documents":
        return await _fetch_documents_context(rendered, search_strategy, doc_filter, cache_k)
    elif source == "metrics_db":
        return await _fetch_metrics_context(rendered, doc_filter, cache_k)
    else:
        logger.warning(f"Unknown context_source '{source}', treating as none")
        return ContextResult()


async def _fetch_documents_context(
    rendered: str,
    search_strategy: List[str],
    doc_filter: Optional[str],
    cache_k: Optional[str],
) -> ContextResult:
    """Retrieval from Qdrant — hybrid search with deduplication."""
    if not _retrieval_pipeline:
        return ContextResult(context_str="Error: retrieval pipeline not initialized.")

    filters = None
    if doc_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        filters = Filter(must=[FieldCondition(key="doc_hash", match=MatchValue(value=doc_filter))])

    # RetrievalPipeline.retrieve() always runs hybrid (dense + sparse RRF).
    # search_strategy is captured in the cache key so toggling it busts cache,
    # but the actual search is always hybrid — the pipeline handles fusion internally.
    retrieval_result = await _retrieval_pipeline.retrieve(
        query=rendered,
        top_k=10,
        filters=filters,
    )
    all_chunks = retrieval_result.chunks

    # Build XML context string
    context_parts = []
    for i, r in enumerate(all_chunks):
        context_parts.append(
            f'<source id="{i+1}" file="{r.source_file}" '
            f'page="{r.page_number}">\n{r.text}\n</source>'
        )
    context_str = "\n\n".join(context_parts)

    result = ContextResult(
        context_str=context_str,
        chunks_retrieved=len(all_chunks),
        raw_chunks=[
            {
                "text": r.text,
                "source_file": r.source_file,
                "page_number": r.page_number,
                "item_type": r.item_type,
                "doc_hash": r.doc_hash,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in all_chunks
        ],
    )

    if cache_k:
        _set_cached_context(cache_k, result)

    return result


async def _fetch_metrics_context(
    rendered: str,
    doc_filter: Optional[str],
    cache_k: Optional[str],
) -> ContextResult:
    """Text-to-SQL against metric_facts. First 50 rows as markdown for LLM context."""
    if not _analytics_driver:
        return ContextResult(context_str="Error: analytics driver not initialized.")

    try:
        from src.workflows.nodes.financial_math import generate_and_execute_sql
        sql_result = await generate_and_execute_sql(
            query=rendered,
            analytics_driver=_analytics_driver,
        )

        # Format first 50 rows as markdown table for LLM context window
        if sql_result:
            cols = list(sql_result[0].keys())
            header = "| " + " | ".join(cols) + " |"
            separator = "| " + " | ".join(["---"] * len(cols)) + " |"
            rows = []
            for row in sql_result[:_MAX_SQL_ROWS_FOR_LLM]:
                rows.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
            context_str = f"SQL Results ({len(sql_result)} rows):\n\n{header}\n{separator}\n" + "\n".join(rows)
            if len(sql_result) > _MAX_SQL_ROWS_FOR_LLM:
                context_str += f"\n\n(Showing first {_MAX_SQL_ROWS_FOR_LLM} of {len(sql_result)} rows)"
        else:
            context_str = "SQL query returned no results."

        result = ContextResult(
            context_str=context_str,
            chunks_retrieved=len(sql_result),
            sql_rows=sql_result,
            sql_query=rendered,
        )

        if cache_k:
            _set_cached_context(cache_k, result)

        return result

    except Exception as e:
        logger.error(f"Metrics context fetch failed: {e}")
        return ContextResult(context_str=f"SQL execution error: {e}")


# ══════════════════════════════════════════════════════════════════
# Step 2: Generate Output
# ══════════════════════════════════════════════════════════════════

@traceable(name="Generate Output", run_type="chain")
async def _generate_output(
    rendered: str,
    context: ContextResult,
    output_format: str,
    output_schema: Optional[dict] = None,
    model_id: Optional[str] = None,
    temperature: float = 0.1,
) -> StepOutput:
    """
    Generate output based on the format axis.
    - text: standard LLM generation
    - json: structured output with Pydantic schema
    - chart: Vega-Lite spec from SQL results
    - table: bypass LLM — return raw SQL rows directly
    """
    t0 = time.time()

    if output_format == "table":
        return _output_table(context, t0)
    elif output_format == "chart":
        return await _output_chart(rendered, context, model_id, t0)
    elif output_format == "json":
        return await _output_json(rendered, context, output_schema, model_id, temperature, t0)
    else:  # text
        return await _output_text(rendered, context, model_id, temperature, t0)


def _output_table(context: ContextResult, t0: float) -> StepOutput:
    """TABLE: bypass LLM entirely — return raw SQL rows."""
    duration_ms = int((time.time() - t0) * 1000)
    rows = context.sql_rows or []
    return StepOutput(
        text=json.dumps(rows, indent=2, default=str) if rows else "No data returned.",
        structured={"rows": rows, "sql": context.sql_query},
        metadata={
            "output_format": "table",
            "row_count": len(rows),
            "duration_ms": duration_ms,
            "llm_bypassed": True,
        },
    )


async def _output_chart(
    rendered: str,
    context: ContextResult,
    model_id: Optional[str],
    t0: float,
) -> StepOutput:
    """CHART: generate Vega-Lite spec from SQL result data."""
    from src.workflows.viz_helper import generate_vega_spec, clean_data_for_json

    rows = context.sql_rows or []
    if not rows:
        return StepOutput(
            text="No data available for chart generation.",
            metadata={"output_format": "chart", "error": "no_data"},
        )

    try:
        spec = await generate_vega_spec(rendered, rows, context.sql_query or "", model_id)
        clean_data = clean_data_for_json(rows)
        duration_ms = int((time.time() - t0) * 1000)

        return StepOutput(
            text="",
            structured={
                "spec": spec,
                "data": clean_data,
                "sql": context.sql_query,
                "title": spec.get("title", rendered[:60]),
            },
            metadata={
                "output_format": "chart",
                "row_count": len(rows),
                "duration_ms": duration_ms,
                "model": model_id or SystemConfig.GENERATION_MODEL,
            },
        )
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        duration_ms = int((time.time() - t0) * 1000)
        return StepOutput(
            text=f"Chart generation failed: {e}",
            structured={"rows": rows, "sql": context.sql_query},
            metadata={
                "output_format": "chart",
                "error": str(e),
                "duration_ms": duration_ms,
                "fallback": "table",
            },
        )


async def _output_json(
    rendered: str,
    context: ContextResult,
    output_schema: Optional[dict],
    model_id: Optional[str],
    temperature: float,
    t0: float,
) -> StepOutput:
    """JSON: structured output with Pydantic schema validation."""
    if not output_schema:
        return StepOutput(
            text="Error: JSON output requires an output_schema.",
            metadata={"error": "no_schema"},
        )

    llm = SystemConfig.get_llm(
        model_name=model_id or SystemConfig.GENERATION_MODEL,
        temperature=0.0,
        max_tokens=4000,
    )

    system_prompt = (
        "You are a financial data extraction assistant. Extract the requested data "
        "from the sources into the specified JSON structure.\n\n"
        f"SOURCES:\n{context.context_str}"
    )

    try:
        structured_llm = llm.with_structured_output(output_schema)
        result = await structured_llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rendered},
        ])

        if isinstance(result, BaseModel):
            structured_data = result.model_dump()
        elif isinstance(result, dict):
            structured_data = result
        else:
            structured_data = {"result": str(result)}

        text = json.dumps(structured_data, indent=2, default=str)
    except Exception as e:
        logger.error(f"Structured extraction failed: {e}")
        structured_data = None
        text = f"Structured extraction error: {e}"

    duration_ms = int((time.time() - t0) * 1000)
    return StepOutput(
        text=text,
        structured=structured_data,
        metadata={
            "output_format": "json",
            "context_source": "documents" if context.raw_chunks else "metrics_db" if context.sql_rows else "none",
            "chunks_retrieved": context.chunks_retrieved,
            "cache_hit": context.cache_hit,
            "duration_ms": duration_ms,
            "model": model_id or SystemConfig.GENERATION_MODEL,
            "chunks": context.raw_chunks[:5] if context.raw_chunks else [],
        },
    )


async def _output_text(
    rendered: str,
    context: ContextResult,
    model_id: Optional[str],
    temperature: float,
    t0: float,
) -> StepOutput:
    """TEXT: standard LLM generation with context."""
    llm = SystemConfig.get_llm(
        model_name=model_id or SystemConfig.GENERATION_MODEL,
        temperature=temperature,
        max_tokens=2000,
    )

    if context.context_str:
        system_prompt = (
            "You are a senior financial analyst assistant. Answer the user's question "
            "using ONLY the provided sources. Be precise with numbers.\n\n"
            f"SOURCES:\n{context.context_str}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rendered},
        ]
    else:
        messages = [
            {"role": "user", "content": rendered},
        ]

    response = await llm.ainvoke(messages)

    duration_ms = int((time.time() - t0) * 1000)
    return StepOutput(
        text=response.content.strip(),
        metadata={
            "output_format": "text",
            "context_source": "documents" if context.raw_chunks else "metrics_db" if context.sql_rows else "none",
            "chunks_retrieved": context.chunks_retrieved,
            "cache_hit": context.cache_hit,
            "duration_ms": duration_ms,
            "model": model_id or SystemConfig.GENERATION_MODEL,
            "chunks": context.raw_chunks[:5] if context.raw_chunks else [],
        },
    )


# ══════════════════════════════════════════════════════════════════
# Top-Level Dispatcher
# ══════════════════════════════════════════════════════════════════

@traceable(name="Execute Prompt", run_type="chain")
async def execute_prompt(
    template: str,
    variables: Dict[str, Any],
    context_source: str = "documents",
    output_format: str = "text",
    search_strategy: Optional[List[str]] = None,
    output_schema: Optional[dict] = None,
    doc_filter: Optional[str] = None,
    model_id: Optional[str] = None,
    temperature: float = 0.1,
    prompt_id: Optional[str] = None,
    force_retrieve: bool = False,
) -> StepOutput:
    """
    Top-level dispatcher. Renders the Jinja2 template, then runs
    the 2-step pipeline: fetch context → generate output.
    """
    search_strategy = search_strategy or ["semantic"]

    # Validate combination
    if output_format in ("chart", "table") and context_source != "metrics_db":
        return StepOutput(
            text=f"Error: {output_format} output requires metrics_db context source.",
            metadata={"error": "invalid_combination"},
        )

    # Render template
    rendered = render_template(template, variables)

    # Step 1: Fetch context
    context = await _fetch_context(
        source=context_source,
        rendered=rendered,
        search_strategy=search_strategy,
        doc_filter=doc_filter,
        prompt_id=prompt_id,
        force_retrieve=force_retrieve,
    )

    # Step 2: Generate output
    return await _generate_output(
        rendered=rendered,
        context=context,
        output_format=output_format,
        output_schema=output_schema,
        model_id=model_id,
        temperature=temperature,
    )
