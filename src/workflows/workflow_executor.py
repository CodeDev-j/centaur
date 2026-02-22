"""
Workflow Executor: Sequential prompt chain execution via LangGraph.

Static 2-node graph:
  execute_step → route_next_or_end → (execute_step | END)

State checkpoints after each step via the studio_driver (Postgres).
Supports HITL pause: when a step's condition evaluates to "pause",
the run status is set to "paused_for_approval" and execution halts.
Resume from the last checkpoint with resume_workflow().
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langsmith import traceable

from src.storage.studio_driver import get_studio_db
from src.workflows.prompt_executor import execute_prompt, render_template

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Core Execution Logic
# ══════════════════════════════════════════════════════════════════

@traceable(name="Execute Workflow Step", run_type="chain")
async def _execute_single_step(
    step: dict,
    workflow_inputs: dict,
    step_outputs: dict,
    doc_filter: Optional[str] = None,
) -> dict:
    """
    Execute a single workflow step.

    Resolves input_mapping via Jinja2 against:
      - workflow_inputs (user-provided)
      - step_outputs (from prior steps, keyed by output_key)

    Returns the StepOutput as a serializable dict.
    """
    pv = step.get("prompt_version") or {}
    template = pv.get("template", "")
    context_source = pv.get("context_source", "documents")
    output_format = pv.get("output_format", "text")
    search_strategy = pv.get("search_strategy", ["semantic"])
    model_id = pv.get("model_id")
    temperature = pv.get("temperature", 0.1)

    # Build variable context for Jinja2 resolution
    # Steps access prior outputs via: {{ steps.step_key.text }}
    jinja_context = {
        "inputs": workflow_inputs,
        "steps": step_outputs,
    }

    # Resolve input_mapping: each key→value is a Jinja2 expression
    input_mapping = step.get("input_mapping", {})
    resolved_vars = {}
    for var_name, expression in input_mapping.items():
        if isinstance(expression, str) and "{{" in expression:
            resolved_vars[var_name] = render_template(expression, jinja_context)
        else:
            resolved_vars[var_name] = expression

    # Also include direct workflow inputs as template variables
    all_vars = {**workflow_inputs, **resolved_vars}

    from src.workflows.prompt_executor import StepOutput
    result: StepOutput = await execute_prompt(
        template=template,
        variables=all_vars,
        context_source=context_source,
        output_format=output_format,
        search_strategy=search_strategy,
        doc_filter=doc_filter,
        model_id=model_id,
        temperature=temperature,
    )

    return {
        "text": result.text,
        "structured": result.structured,
        "metadata": result.metadata,
    }


def _should_skip_step(step: dict, step_outputs: dict, workflow_inputs: dict) -> bool:
    """Evaluate a step's skip condition. Returns True if step should be skipped."""
    condition = step.get("condition")
    if not condition:
        return False

    # Simple expression evaluation against context
    context = {"inputs": workflow_inputs, "steps": step_outputs}
    try:
        rendered = render_template(condition, context)
        # If the condition renders to "skip" or "false", skip the step
        return rendered.strip().lower() in ("skip", "false", "0", "")
    except Exception:
        return False


def _should_pause_step(step: dict, step_outputs: dict, workflow_inputs: dict) -> bool:
    """Check if a step requires HITL approval before proceeding."""
    condition = step.get("condition")
    if not condition:
        return False

    context = {"inputs": workflow_inputs, "steps": step_outputs}
    try:
        rendered = render_template(condition, context)
        return rendered.strip().lower() == "pause"
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
# Main Executor
# ══════════════════════════════════════════════════════════════════

@traceable(name="Execute Workflow", run_type="chain")
async def execute_workflow(
    workflow_id: str,
    inputs: Optional[dict] = None,
    doc_filter: Optional[str] = None,
) -> Optional[dict]:
    """
    Execute a workflow sequentially.

    Creates a run record, iterates through steps, checkpoints after each,
    and returns the completed run record.

    Returns None if workflow not found.
    """
    db = get_studio_db()
    inputs = inputs or {}

    # Load workflow with steps
    wf = db.get_workflow(workflow_id)
    if not wf:
        return None

    steps = wf.get("steps", [])
    if not steps:
        return {
            "id": "",
            "workflow_id": workflow_id,
            "status": "completed",
            "inputs": inputs,
            "step_outputs": {},
            "current_step": 0,
            "started_at": None,
            "completed_at": None,
            "error": "No steps defined",
        }

    # Create run record
    run = db.create_run(workflow_id=workflow_id, inputs=inputs)
    if not run:
        return None

    run_id = run["id"]
    step_outputs: Dict[str, Any] = {}
    t0 = time.time()

    try:
        for i, step in enumerate(steps):
            # Checkpoint current position
            db.update_run(run_id, current_step=i, step_outputs=step_outputs)

            # Check skip condition
            if _should_skip_step(step, step_outputs, inputs):
                logger.info(f"Step {i} '{step['label']}' skipped by condition")
                continue

            # Check HITL pause
            if _should_pause_step(step, step_outputs, inputs):
                logger.info(f"Step {i} '{step['label']}' paused for approval")
                db.update_run(
                    run_id,
                    status="paused_for_approval",
                    current_step=i,
                    step_outputs=step_outputs,
                )
                return db.get_run(run_id)

            # Execute with retry
            retry_count = step.get("retry_count", 1)
            last_error = None

            for attempt in range(retry_count):
                try:
                    output = await _execute_single_step(
                        step=step,
                        workflow_inputs=inputs,
                        step_outputs=step_outputs,
                        doc_filter=doc_filter,
                    )
                    step_outputs[step["output_key"]] = output
                    last_error = None
                    break
                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        f"Step {i} '{step['label']}' attempt {attempt+1}/{retry_count} "
                        f"failed: {e}"
                    )

            if last_error:
                db.update_run(
                    run_id,
                    status="failed",
                    step_outputs=step_outputs,
                    current_step=i,
                    error=f"Step '{step['label']}' failed after {retry_count} attempts: {last_error}",
                )
                return db.get_run(run_id)

        # All steps completed
        duration_ms = int((time.time() - t0) * 1000)
        step_outputs["_metadata"] = {"duration_ms": duration_ms, "total_steps": len(steps)}

        db.update_run(
            run_id,
            status="completed",
            step_outputs=step_outputs,
            current_step=len(steps),
        )
        return db.get_run(run_id)

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        db.update_run(
            run_id,
            status="failed",
            step_outputs=step_outputs,
            error=str(e),
        )
        return db.get_run(run_id)


# ══════════════════════════════════════════════════════════════════
# Resume (HITL)
# ══════════════════════════════════════════════════════════════════

@traceable(name="Resume Workflow", run_type="chain")
async def resume_workflow(
    run_id: str,
    doc_filter: Optional[str] = None,
) -> Optional[dict]:
    """
    Resume a paused-for-approval workflow run from its checkpoint.

    Returns None if run not found or not in paused state.
    """
    db = get_studio_db()

    run = db.get_run(run_id)
    if not run or run["status"] != "paused_for_approval":
        return None

    workflow_id = run["workflow_id"]
    inputs = run["inputs"]
    step_outputs = run["step_outputs"]
    resume_from = run["current_step"]

    # Load workflow
    wf = db.get_workflow(workflow_id)
    if not wf:
        return None

    steps = wf.get("steps", [])

    # Update status to running
    db.update_run(run_id, status="running")

    t0 = time.time()

    try:
        # Resume from the paused step (execute it, then continue)
        for i in range(resume_from, len(steps)):
            step = steps[i]

            # Checkpoint
            db.update_run(run_id, current_step=i, step_outputs=step_outputs)

            # Skip if already has output (from before pause)
            if step["output_key"] in step_outputs:
                continue

            # Check skip condition
            if _should_skip_step(step, step_outputs, inputs):
                continue

            # Don't re-pause on the same step we just approved
            if i > resume_from and _should_pause_step(step, step_outputs, inputs):
                db.update_run(
                    run_id,
                    status="paused_for_approval",
                    current_step=i,
                    step_outputs=step_outputs,
                )
                return db.get_run(run_id)

            # Execute with retry
            retry_count = step.get("retry_count", 1)
            last_error = None

            for attempt in range(retry_count):
                try:
                    output = await _execute_single_step(
                        step=step,
                        workflow_inputs=inputs,
                        step_outputs=step_outputs,
                        doc_filter=doc_filter,
                    )
                    step_outputs[step["output_key"]] = output
                    last_error = None
                    break
                except Exception as e:
                    last_error = str(e)

            if last_error:
                db.update_run(
                    run_id,
                    status="failed",
                    step_outputs=step_outputs,
                    current_step=i,
                    error=f"Step '{step['label']}' failed: {last_error}",
                )
                return db.get_run(run_id)

        # Completed
        duration_ms = int((time.time() - t0) * 1000)
        step_outputs["_metadata"] = {
            "duration_ms": duration_ms,
            "total_steps": len(steps),
            "resumed_from": resume_from,
        }

        db.update_run(
            run_id,
            status="completed",
            step_outputs=step_outputs,
            current_step=len(steps),
        )
        return db.get_run(run_id)

    except Exception as e:
        logger.error(f"Workflow resume failed: {e}")
        db.update_run(
            run_id,
            status="failed",
            step_outputs=step_outputs,
            error=str(e),
        )
        return db.get_run(run_id)
