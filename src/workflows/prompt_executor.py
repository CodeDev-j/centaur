"""
Prompt Executor: Decoupled Execution Service for Studio.

Four explicit modes — NEVER routes through the chat router:
  1. rag         — forced retrieval → generate (hardcoded qual/quant)
  2. structured  — forced retrieval → generate with Pydantic output schema
  3. direct      — LLM only, no retrieval
  4. sql         — text-to-SQL against metric_facts

Each mode returns a StepOutput dataclass with text, structured, and metadata.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jinja2 import TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment
from langsmith import traceable
from pydantic import BaseModel, create_model

from src.config import SystemConfig

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# StepOutput — the universal return type for all execution modes
# ══════════════════════════════════════════════════════════════════

@dataclass
class StepOutput:
    """
    Universal output from any execution mode.
    - text: raw LLM response (always available)
    - structured: parsed JSON dict (if output_schema was provided and parse succeeded)
    - metadata: execution telemetry (model, tokens, duration, citations, etc.)
    """
    text: str = ""
    structured: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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
# Execution Modes
# ══════════════════════════════════════════════════════════════════

# Module-level references set during startup (same pattern as retrieve.py)
_retrieval_pipeline = None
_analytics_driver = None

# Retrieval cache: keyed by "{prompt_id}:{doc_filter}" → (context_str, timestamp)
# Used during prompt refinement to skip re-retrieval when iterating on template
_retrieval_cache: Dict[str, tuple] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes
_MAX_CACHE_SIZE = 256  # Prevent unbounded memory growth


def _cache_key(prompt_id: str, doc_filter: Optional[str]) -> str:
    return f"{prompt_id}:{doc_filter or 'all'}"


def _get_cached_context(key: str) -> Optional[str]:
    if key in _retrieval_cache:
        ctx, ts = _retrieval_cache[key]
        if time.time() - ts < _CACHE_TTL_SECONDS:
            return ctx
        del _retrieval_cache[key]
    return None


def _set_cached_context(key: str, context_str: str):
    # Evict oldest entry if cache is at capacity
    if len(_retrieval_cache) >= _MAX_CACHE_SIZE:
        oldest_key = min(_retrieval_cache, key=lambda k: _retrieval_cache[k][1])
        del _retrieval_cache[oldest_key]
    _retrieval_cache[key] = (context_str, time.time())


def set_retrieval_pipeline(pipeline):
    global _retrieval_pipeline
    _retrieval_pipeline = pipeline


def set_analytics_driver(driver):
    global _analytics_driver
    _analytics_driver = driver


@traceable(name="Execute Prompt (RAG)", run_type="chain")
async def execute_rag(
    rendered_prompt: str,
    retrieval_mode: str = "qualitative",
    doc_filter: Optional[str] = None,
    model_id: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> StepOutput:
    """
    Mode: rag — forced retrieval (hardcoded qual/quant) → generate.
    Bypasses the chat router entirely.
    If cache_key is provided, tries to reuse cached retrieval context.
    """
    t0 = time.time()
    cache_hit = False

    # Try cache first
    context_str = _get_cached_context(cache_key) if cache_key else None
    num_chunks = 0

    if context_str:
        cache_hit = True
        logger.info(f"Retrieval cache HIT for key={cache_key}")
    else:
        if not _retrieval_pipeline:
            return StepOutput(
                text="Error: retrieval pipeline not initialized.",
                metadata={"error": "no_pipeline"},
            )

        # Retrieve chunks
        from src.retrieval.qdrant import SearchResult
        filters = None
        if doc_filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            filters = Filter(must=[FieldCondition(key="doc_hash", match=MatchValue(value=doc_filter))])

        results: List[SearchResult] = await _retrieval_pipeline.search(
            query_text=rendered_prompt,
            limit=10,
            filters=filters,
        )
        num_chunks = len(results)

        # Build context string from retrieved chunks
        context_parts = []
        for i, r in enumerate(results):
            context_parts.append(
                f'<source id="{i+1}" file="{r.metadata.get("source_file", "")}" '
                f'page="{r.metadata.get("page_number", 0)}">\n{r.text}\n</source>'
            )
        context_str = "\n\n".join(context_parts)

        # Cache for refinement
        if cache_key:
            _set_cached_context(cache_key, context_str)

    # Generate answer
    llm = SystemConfig.get_llm(
        model_name=model_id or SystemConfig.GENERATION_MODEL,
        temperature=0.1,
        max_tokens=2000,
    )

    system_prompt = (
        "You are a senior financial analyst assistant. Answer the user's question "
        "using ONLY the provided sources. Be precise with numbers.\n\n"
        f"SOURCES:\n{context_str}"
    )

    response = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rendered_prompt},
    ])

    duration_ms = int((time.time() - t0) * 1000)
    return StepOutput(
        text=response.content.strip(),
        metadata={
            "mode": "rag",
            "retrieval_mode": retrieval_mode,
            "chunks_retrieved": num_chunks,
            "cache_hit": cache_hit,
            "duration_ms": duration_ms,
            "model": model_id or SystemConfig.GENERATION_MODEL,
        },
    )


@traceable(name="Execute Prompt (Structured)", run_type="chain")
async def execute_structured(
    rendered_prompt: str,
    output_schema: dict,
    retrieval_mode: str = "qualitative",
    doc_filter: Optional[str] = None,
    model_id: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> StepOutput:
    """
    Mode: structured — forced retrieval → generate with Pydantic output schema.
    Uses .with_structured_output() for deterministic JSON.
    If cache_key is provided, tries to reuse cached retrieval context.
    """
    t0 = time.time()
    cache_hit = False

    # Try cache first
    context_str = _get_cached_context(cache_key) if cache_key else None
    num_chunks = 0

    if context_str:
        cache_hit = True
        logger.info(f"Retrieval cache HIT for key={cache_key}")
    else:
        if not _retrieval_pipeline:
            return StepOutput(
                text="Error: retrieval pipeline not initialized.",
                metadata={"error": "no_pipeline"},
            )

        # Retrieve (same as rag mode)
        filters = None
        if doc_filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            filters = Filter(must=[FieldCondition(key="doc_hash", match=MatchValue(value=doc_filter))])

        results = await _retrieval_pipeline.search(
            query_text=rendered_prompt,
            limit=10,
            filters=filters,
        )
        num_chunks = len(results)

        context_parts = []
        for i, r in enumerate(results):
            context_parts.append(
                f'<source id="{i+1}" file="{r.metadata.get("source_file", "")}" '
                f'page="{r.metadata.get("page_number", 0)}">\n{r.text}\n</source>'
            )
        context_str = "\n\n".join(context_parts)

        # Cache for refinement
        if cache_key:
            _set_cached_context(cache_key, context_str)

    # Build dynamic Pydantic model from schema
    llm = SystemConfig.get_llm(
        model_name=model_id or SystemConfig.GENERATION_MODEL,
        temperature=0.0,
        max_tokens=4000,
    )

    system_prompt = (
        "You are a financial data extraction assistant. Extract the requested data "
        "from the sources into the specified JSON structure.\n\n"
        f"SOURCES:\n{context_str}"
    )

    try:
        structured_llm = llm.with_structured_output(output_schema)
        result = await structured_llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rendered_prompt},
        ])

        # Convert Pydantic model or dict to plain dict
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
            "mode": "structured",
            "retrieval_mode": retrieval_mode,
            "chunks_retrieved": num_chunks,
            "cache_hit": cache_hit,
            "duration_ms": duration_ms,
            "model": model_id or SystemConfig.GENERATION_MODEL,
        },
    )


@traceable(name="Execute Prompt (Direct)", run_type="chain")
async def execute_direct(
    rendered_prompt: str,
    model_id: Optional[str] = None,
) -> StepOutput:
    """
    Mode: direct — LLM only, no retrieval. For synthesis/analysis/reformatting.
    """
    t0 = time.time()

    llm = SystemConfig.get_llm(
        model_name=model_id or SystemConfig.GENERATION_MODEL,
        temperature=0.2,
        max_tokens=4000,
    )

    response = await llm.ainvoke([
        {"role": "user", "content": rendered_prompt},
    ])

    duration_ms = int((time.time() - t0) * 1000)
    return StepOutput(
        text=response.content.strip(),
        metadata={
            "mode": "direct",
            "duration_ms": duration_ms,
            "model": model_id or SystemConfig.GENERATION_MODEL,
        },
    )


@traceable(name="Execute Prompt (SQL)", run_type="chain")
async def execute_sql(
    rendered_prompt: str,
    doc_filter: Optional[str] = None,
    model_id: Optional[str] = None,
) -> StepOutput:
    """
    Mode: sql — text-to-SQL against metric_facts table.
    Reuses the existing financial_math pipeline.
    """
    t0 = time.time()

    if not _analytics_driver:
        return StepOutput(
            text="Error: analytics driver not initialized.",
            metadata={"error": "no_analytics"},
        )

    try:
        from src.workflows.nodes.financial_math import generate_and_execute_sql
        sql_result = await generate_and_execute_sql(
            query=rendered_prompt,
            analytics_driver=_analytics_driver,
            doc_filter=doc_filter,
        )

        text = json.dumps(sql_result, indent=2, default=str)
        structured = {"rows": sql_result}
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        text = f"SQL execution error: {e}"
        structured = None

    duration_ms = int((time.time() - t0) * 1000)
    return StepOutput(
        text=text,
        structured=structured,
        metadata={
            "mode": "sql",
            "duration_ms": duration_ms,
            "model": model_id or SystemConfig.REASONING_MODEL,
        },
    )


# ══════════════════════════════════════════════════════════════════
# Dispatcher — routes to the correct execution mode
# ══════════════════════════════════════════════════════════════════

@traceable(name="Execute Prompt", run_type="chain")
async def execute_prompt(
    template: str,
    variables: Dict[str, Any],
    exec_mode: str = "rag",
    output_schema: Optional[dict] = None,
    retrieval_mode: str = "qualitative",
    doc_filter: Optional[str] = None,
    model_id: Optional[str] = None,
    prompt_id: Optional[str] = None,
) -> StepOutput:
    """
    Top-level dispatcher. Renders the Jinja2 template, then routes
    to the correct execution mode.

    When prompt_id is provided, enables retrieval caching for refinement loops:
    first run retrieves and caches; subsequent runs reuse cached context.
    """
    # Render template
    rendered = render_template(template, variables)

    # Build cache key for retrieval modes
    cache_key = _cache_key(prompt_id, doc_filter) if prompt_id else None

    if exec_mode == "rag":
        return await execute_rag(rendered, retrieval_mode, doc_filter, model_id, cache_key)
    elif exec_mode == "structured":
        if not output_schema:
            return StepOutput(
                text="Error: structured mode requires output_schema.",
                metadata={"error": "no_schema"},
            )
        return await execute_structured(rendered, output_schema, retrieval_mode, doc_filter, model_id, cache_key)
    elif exec_mode == "direct":
        return await execute_direct(rendered, model_id)
    elif exec_mode == "sql":
        return await execute_sql(rendered, doc_filter, model_id)
    else:
        return StepOutput(
            text=f"Error: unknown exec_mode '{exec_mode}'.",
            metadata={"error": "unknown_mode"},
        )
