"""
Studio Driver: CRUD operations for Prompt Library, Workflows, and Agent Tools.

Uses the same Postgres instance as analytics_driver (SystemConfig connection).
Tables are managed by Alembic migrations (not create_all).
"""

import logging
from contextlib import contextmanager
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from langsmith import traceable

from src.config import SystemConfig
from src.storage.studio_models import (
    Base, Prompt, PromptVersion, Workflow, WorkflowStep,
    WorkflowRun, AgentTool,
)

logger = logging.getLogger(__name__)


class StudioDriver:
    """
    Manages CRUD for Studio tables.
    Singleton instance created at module level (same pattern as ledger_db).
    """

    def __init__(self):
        db_url = (
            f"postgresql://{SystemConfig.POSTGRES_USER}:{SystemConfig.POSTGRES_PASSWORD}"
            f"@{SystemConfig.POSTGRES_HOST}:{SystemConfig.POSTGRES_PORT}"
            f"/{SystemConfig.POSTGRES_DB}"
        )
        self.engine = create_engine(db_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine,
        )
        logger.info("Studio Driver initialized.")

    @contextmanager
    def session(self):
        """Provide a transactional scope."""
        s: Session = self.SessionLocal()
        try:
            yield s
            s.commit()
        except Exception as e:
            s.rollback()
            logger.error(f"Studio DB session error: {e}")
            raise
        finally:
            s.close()

    # ══════════════════════════════════════════════════════════════
    # PROMPTS
    # ══════════════════════════════════════════════════════════════

    @traceable(name="Create Prompt", run_type="tool")
    def create_prompt(
        self,
        name: str,
        description: str = "",
        category: str = "custom",
        tags: Optional[List[str]] = None,
    ) -> Prompt:
        with self.session() as s:
            prompt = Prompt(
                id=str(uuid4()),
                name=name,
                description=description,
                category=category,
                tags=tags or [],
            )
            s.add(prompt)
            s.flush()
            # Detach before commit closes session
            s.expunge(prompt)
            return prompt

    @traceable(name="List Prompts", run_type="tool")
    def list_prompts(
        self,
        category: Optional[str] = None,
        include_archived: bool = False,
    ) -> List[dict]:
        with self.session() as s:
            q = s.query(Prompt)
            if not include_archived:
                q = q.filter(Prompt.is_archived == False)
            if category:
                q = q.filter(Prompt.category == category)
            q = q.order_by(Prompt.created_at.desc())
            return [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "category": p.category,
                    "tags": p.tags,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "is_archived": p.is_archived,
                    "latest_version": (
                        p.versions[-1].version if p.versions else 0
                    ),
                }
                for p in q.all()
            ]

    @traceable(name="Get Prompt", run_type="tool")
    def get_prompt(self, prompt_id: str) -> Optional[dict]:
        with self.session() as s:
            p = s.query(Prompt).filter(Prompt.id == prompt_id).first()
            if not p:
                return None
            return {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "category": p.category,
                "tags": p.tags,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "is_archived": p.is_archived,
                "versions": [
                    {
                        "id": v.id,
                        "version": v.version,
                        "template": v.template,
                        "variables": v.variables,
                        "exec_mode": v.exec_mode,
                        "output_schema": v.output_schema,
                        "model_id": v.model_id,
                        "retrieval_mode": v.retrieval_mode,
                        "created_at": v.created_at.isoformat() if v.created_at else None,
                        "is_published": v.is_published,
                    }
                    for v in p.versions
                ],
            }

    @traceable(name="Update Prompt", run_type="tool")
    def update_prompt(
        self,
        prompt_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        with self.session() as s:
            p = s.query(Prompt).filter(Prompt.id == prompt_id).first()
            if not p:
                return False
            if name is not None:
                p.name = name
            if description is not None:
                p.description = description
            if category is not None:
                p.category = category
            if tags is not None:
                p.tags = tags
            return True

    @traceable(name="Archive Prompt", run_type="tool")
    def archive_prompt(self, prompt_id: str) -> bool:
        with self.session() as s:
            p = s.query(Prompt).filter(Prompt.id == prompt_id).first()
            if not p:
                return False
            p.is_archived = True
            return True

    # ══════════════════════════════════════════════════════════════
    # PROMPT VERSIONS
    # ══════════════════════════════════════════════════════════════

    @traceable(name="Publish Prompt Version", run_type="tool")
    def publish_version(
        self,
        prompt_id: str,
        template: str,
        variables: Optional[List[dict]] = None,
        exec_mode: str = "rag",
        output_schema: Optional[dict] = None,
        model_id: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Create an immutable prompt version. Auto-increments version number.
        """
        with self.session() as s:
            prompt = s.query(Prompt).filter(Prompt.id == prompt_id).first()
            if not prompt:
                return None

            # Get next version number
            latest = (
                s.query(PromptVersion)
                .filter(PromptVersion.prompt_id == prompt_id)
                .order_by(PromptVersion.version.desc())
                .first()
            )
            next_version = (latest.version + 1) if latest else 1

            pv = PromptVersion(
                id=str(uuid4()),
                prompt_id=prompt_id,
                version=next_version,
                template=template,
                variables=variables or [],
                exec_mode=exec_mode,
                output_schema=output_schema,
                model_id=model_id,
                retrieval_mode=retrieval_mode,
                is_published=True,
            )
            s.add(pv)
            s.flush()

            return {
                "id": pv.id,
                "prompt_id": prompt_id,
                "version": next_version,
                "template": template,
                "variables": pv.variables,
                "exec_mode": exec_mode,
                "output_schema": output_schema,
                "model_id": model_id,
                "retrieval_mode": retrieval_mode,
                "is_published": True,
            }

    @traceable(name="Get Prompt Version", run_type="tool")
    def get_version(self, version_id: str) -> Optional[dict]:
        with self.session() as s:
            pv = s.query(PromptVersion).filter(PromptVersion.id == version_id).first()
            if not pv:
                return None
            return {
                "id": pv.id,
                "prompt_id": pv.prompt_id,
                "version": pv.version,
                "template": pv.template,
                "variables": pv.variables,
                "exec_mode": pv.exec_mode,
                "output_schema": pv.output_schema,
                "model_id": pv.model_id,
                "retrieval_mode": pv.retrieval_mode,
                "created_at": pv.created_at.isoformat() if pv.created_at else None,
                "is_published": pv.is_published,
            }

    @traceable(name="List Prompt Versions", run_type="tool")
    def list_versions(self, prompt_id: str) -> List[dict]:
        with self.session() as s:
            versions = (
                s.query(PromptVersion)
                .filter(PromptVersion.prompt_id == prompt_id)
                .order_by(PromptVersion.version.desc())
                .all()
            )
            return [
                {
                    "id": v.id,
                    "version": v.version,
                    "exec_mode": v.exec_mode,
                    "created_at": v.created_at.isoformat() if v.created_at else None,
                    "is_published": v.is_published,
                }
                for v in versions
            ]


    # ══════════════════════════════════════════════════════════════
    # WORKFLOWS
    # ══════════════════════════════════════════════════════════════

    @traceable(name="Create Workflow", run_type="tool")
    def create_workflow(
        self,
        name: str,
        description: str = "",
        input_vars: Optional[List[dict]] = None,
    ) -> Workflow:
        with self.session() as s:
            wf = Workflow(
                id=str(uuid4()),
                name=name,
                description=description,
                input_vars=input_vars or [],
            )
            s.add(wf)
            s.flush()
            s.expunge(wf)
            return wf

    @traceable(name="List Workflows", run_type="tool")
    def list_workflows(self, include_archived: bool = False) -> List[dict]:
        with self.session() as s:
            q = s.query(Workflow)
            if not include_archived:
                q = q.filter(Workflow.is_archived == False)
            q = q.order_by(Workflow.created_at.desc())
            return [
                {
                    "id": w.id,
                    "name": w.name,
                    "description": w.description,
                    "input_vars": w.input_vars,
                    "created_at": w.created_at.isoformat() if w.created_at else None,
                    "updated_at": w.updated_at.isoformat() if w.updated_at else None,
                    "is_archived": w.is_archived,
                    "step_count": len(w.steps),
                }
                for w in q.all()
            ]

    @traceable(name="Get Workflow", run_type="tool")
    def get_workflow(self, workflow_id: str) -> Optional[dict]:
        with self.session() as s:
            w = s.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not w:
                return None
            return {
                "id": w.id,
                "name": w.name,
                "description": w.description,
                "input_vars": w.input_vars,
                "created_at": w.created_at.isoformat() if w.created_at else None,
                "updated_at": w.updated_at.isoformat() if w.updated_at else None,
                "is_archived": w.is_archived,
                "steps": [
                    {
                        "id": st.id,
                        "step_order": st.step_order,
                        "label": st.label,
                        "prompt_version_id": st.prompt_version_id,
                        "input_mapping": st.input_mapping,
                        "output_key": st.output_key,
                        "condition": st.condition,
                        "retry_count": st.retry_count,
                        # Include prompt version details for display
                        "prompt_version": {
                            "id": st.prompt_version.id,
                            "prompt_id": st.prompt_version.prompt_id,
                            "version": st.prompt_version.version,
                            "template": st.prompt_version.template,
                            "exec_mode": st.prompt_version.exec_mode,
                            "model_id": st.prompt_version.model_id,
                        } if st.prompt_version else None,
                    }
                    for st in w.steps
                ],
            }

    @traceable(name="Update Workflow", run_type="tool")
    def update_workflow(
        self,
        workflow_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_vars: Optional[List[dict]] = None,
    ) -> bool:
        with self.session() as s:
            w = s.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not w:
                return False
            if name is not None:
                w.name = name
            if description is not None:
                w.description = description
            if input_vars is not None:
                w.input_vars = input_vars
            return True

    @traceable(name="Archive Workflow", run_type="tool")
    def archive_workflow(self, workflow_id: str) -> bool:
        with self.session() as s:
            w = s.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not w:
                return False
            w.is_archived = True
            return True

    # ══════════════════════════════════════════════════════════════
    # WORKFLOW STEPS
    # ══════════════════════════════════════════════════════════════

    @traceable(name="Add Workflow Step", run_type="tool")
    def add_step(
        self,
        workflow_id: str,
        prompt_version_id: str,
        label: str,
        output_key: str,
        input_mapping: Optional[dict] = None,
        condition: Optional[str] = None,
        retry_count: int = 1,
        step_order: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Add a step to a workflow. If step_order is None, appends to the end.
        """
        with self.session() as s:
            wf = s.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not wf:
                return None

            # Verify prompt version exists
            pv = s.query(PromptVersion).filter(PromptVersion.id == prompt_version_id).first()
            if not pv:
                return None

            if step_order is None:
                step_order = len(wf.steps)

            step = WorkflowStep(
                id=str(uuid4()),
                workflow_id=workflow_id,
                prompt_version_id=prompt_version_id,
                step_order=step_order,
                label=label,
                input_mapping=input_mapping or {},
                output_key=output_key,
                condition=condition,
                retry_count=retry_count,
            )
            s.add(step)
            s.flush()

            return {
                "id": step.id,
                "workflow_id": workflow_id,
                "step_order": step_order,
                "label": label,
                "prompt_version_id": prompt_version_id,
                "input_mapping": step.input_mapping,
                "output_key": output_key,
                "condition": condition,
                "retry_count": retry_count,
            }

    @traceable(name="Remove Workflow Step", run_type="tool")
    def remove_step(self, step_id: str) -> bool:
        with self.session() as s:
            step = s.query(WorkflowStep).filter(WorkflowStep.id == step_id).first()
            if not step:
                return False
            wf_id = step.workflow_id
            s.delete(step)
            s.flush()

            # Reorder remaining steps
            remaining = (
                s.query(WorkflowStep)
                .filter(WorkflowStep.workflow_id == wf_id)
                .order_by(WorkflowStep.step_order)
                .all()
            )
            for i, st in enumerate(remaining):
                st.step_order = i

            return True

    @traceable(name="Reorder Workflow Steps", run_type="tool")
    def reorder_steps(self, workflow_id: str, step_ids: List[str]) -> bool:
        """Set step order based on provided list of step IDs."""
        with self.session() as s:
            for i, step_id in enumerate(step_ids):
                step = s.query(WorkflowStep).filter(WorkflowStep.id == step_id).first()
                if step and step.workflow_id == workflow_id:
                    step.step_order = i
            return True

    @traceable(name="Update Workflow Step", run_type="tool")
    def update_step(
        self,
        step_id: str,
        label: Optional[str] = None,
        input_mapping: Optional[dict] = None,
        output_key: Optional[str] = None,
        condition: Optional[str] = None,
        retry_count: Optional[int] = None,
        prompt_version_id: Optional[str] = None,
    ) -> bool:
        with self.session() as s:
            step = s.query(WorkflowStep).filter(WorkflowStep.id == step_id).first()
            if not step:
                return False
            if label is not None:
                step.label = label
            if input_mapping is not None:
                step.input_mapping = input_mapping
            if output_key is not None:
                step.output_key = output_key
            if condition is not None:
                step.condition = condition
            if retry_count is not None:
                step.retry_count = retry_count
            if prompt_version_id is not None:
                step.prompt_version_id = prompt_version_id
            return True

    # ══════════════════════════════════════════════════════════════
    # WORKFLOW RUNS
    # ══════════════════════════════════════════════════════════════

    @traceable(name="Create Workflow Run", run_type="tool")
    def create_run(
        self,
        workflow_id: str,
        inputs: Optional[dict] = None,
    ) -> Optional[dict]:
        with self.session() as s:
            wf = s.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not wf:
                return None

            run = WorkflowRun(
                id=str(uuid4()),
                workflow_id=workflow_id,
                status="running",
                inputs=inputs or {},
                step_outputs={},
                current_step=0,
            )
            s.add(run)
            s.flush()
            s.expunge(run)
            return {
                "id": run.id,
                "workflow_id": workflow_id,
                "status": "running",
                "inputs": run.inputs,
                "step_outputs": {},
                "current_step": 0,
                "started_at": run.started_at.isoformat() if run.started_at else None,
            }

    @traceable(name="Update Workflow Run", run_type="tool")
    def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        step_outputs: Optional[dict] = None,
        current_step: Optional[int] = None,
        error: Optional[str] = None,
    ) -> bool:
        with self.session() as s:
            run = s.query(WorkflowRun).filter(WorkflowRun.id == run_id).first()
            if not run:
                return False
            if status is not None:
                run.status = status
                if status in ("completed", "failed"):
                    from datetime import datetime
                    run.completed_at = datetime.utcnow()
            if step_outputs is not None:
                run.step_outputs = step_outputs
            if current_step is not None:
                run.current_step = current_step
            if error is not None:
                run.error = error
            return True

    @traceable(name="Get Workflow Run", run_type="tool")
    def get_run(self, run_id: str) -> Optional[dict]:
        with self.session() as s:
            run = s.query(WorkflowRun).filter(WorkflowRun.id == run_id).first()
            if not run:
                return None
            return {
                "id": run.id,
                "workflow_id": run.workflow_id,
                "status": run.status,
                "inputs": run.inputs,
                "step_outputs": run.step_outputs,
                "current_step": run.current_step,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "error": run.error,
            }

    @traceable(name="List Workflow Runs", run_type="tool")
    def list_runs(self, workflow_id: str, limit: int = 20) -> List[dict]:
        with self.session() as s:
            runs = (
                s.query(WorkflowRun)
                .filter(WorkflowRun.workflow_id == workflow_id)
                .order_by(WorkflowRun.started_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "status": r.status,
                    "current_step": r.current_step,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                    "error": r.error,
                }
                for r in runs
            ]


# Singleton — lazily initialized (DB might not be up at import time)
_studio_db: Optional[StudioDriver] = None


def get_studio_db() -> StudioDriver:
    global _studio_db
    if _studio_db is None:
        _studio_db = StudioDriver()
    return _studio_db
