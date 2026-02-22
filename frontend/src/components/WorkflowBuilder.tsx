"use client";

import { useEffect, useState } from "react";
import {
  ArrowLeft,
  Plus,
  Play,
  Loader2,
  GitBranch,
  Trash2,
  GripVertical,
  ChevronRight,
  CheckCircle,
  XCircle,
  PauseCircle,
  Clock,
} from "lucide-react";
import { useDocStore } from "@/stores/useDocStore";
import {
  listWorkflows,
  createWorkflow,
  getWorkflow,
  archiveWorkflow,
  addWorkflowStep,
  removeWorkflowStep,
  runWorkflow,
  listPrompts,
  approveWorkflowRun,
  WorkflowSummary,
  WorkflowDetail,
  WorkflowRunDetail,
  PromptSummary,
  PromptVersionDetail,
  getPrompt,
} from "@/lib/api";

// ─── Status icon ─────────────────────────────────────────────────────

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <CheckCircle size={12} className="text-green-500" />;
    case "failed":
      return <XCircle size={12} className="text-red-500" />;
    case "paused_for_approval":
      return <PauseCircle size={12} className="text-yellow-500" />;
    case "running":
      return <Loader2 size={12} className="animate-spin text-[var(--accent)]" />;
    default:
      return <Clock size={12} className="text-[var(--text-secondary)]" />;
  }
}

// ─── Workflow List ───────────────────────────────────────────────────

function WorkflowList({
  onSelect,
}: {
  onSelect: (id: string) => void;
}) {
  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    loadWorkflows();
  }, []);

  const loadWorkflows = async () => {
    setLoading(true);
    try {
      const data = await listWorkflows();
      setWorkflows(data);
    } catch (err) {
      console.error("Failed to load workflows:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    setCreating(true);
    try {
      const wf = await createWorkflow({ name: "Untitled Workflow" });
      await loadWorkflows();
      onSelect(wf.id);
    } catch (err) {
      console.error("Failed to create workflow:", err);
    } finally {
      setCreating(false);
    }
  };

  const handleArchive = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await archiveWorkflow(id);
      await loadWorkflows();
    } catch (err) {
      console.error("Failed to archive:", err);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-[var(--border-subtle)]">
        <span className="text-h2 text-[var(--text-primary)]">WORKFLOWS</span>
        <button
          onClick={handleCreate}
          disabled={creating}
          className="flex items-center gap-1 px-2 py-1 rounded text-[11px]
            bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)]
            active:bg-[var(--accent-pressed)] transition-colors duration-75
            disabled:opacity-50"
        >
          {creating ? <Loader2 size={12} className="animate-spin" /> : <Plus size={12} />}
          New
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 size={16} className="animate-spin text-[var(--text-secondary)]" />
          </div>
        ) : workflows.length === 0 ? (
          <div className="px-4 py-8 text-center">
            <p className="text-[12px] text-[var(--text-secondary)]">No workflows yet</p>
            <p className="text-[11px] text-[var(--text-secondary)] mt-1 opacity-60">
              Chain prompts into multi-step workflows
            </p>
          </div>
        ) : (
          <div className="py-1">
            {workflows.map((w) => (
              <div
                key={w.id}
                role="button"
                tabIndex={0}
                onClick={() => onSelect(w.id)}
                className="w-full flex items-start gap-2.5 px-3 py-2 text-left cursor-pointer
                  hover:bg-[var(--bg-surface)] transition-colors duration-75 group"
              >
                <GitBranch size={14} className="text-[var(--text-secondary)] mt-0.5 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-[12px] text-[var(--text-primary)] truncate">{w.name}</div>
                  <div className="text-[10px] text-[var(--text-secondary)] opacity-60 mt-0.5">
                    {w.step_count} step{w.step_count !== 1 ? "s" : ""}
                  </div>
                </div>
                <button
                  onClick={(e) => handleArchive(e, w.id)}
                  className="opacity-0 group-hover:opacity-60 hover:!opacity-100
                    p-1 rounded hover:bg-[var(--bg-tertiary)] transition-opacity duration-75"
                >
                  <Trash2 size={12} className="text-[var(--text-secondary)]" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Step Picker (select a published prompt version) ─────────────────

function StepPicker({
  onAdd,
  onCancel,
}: {
  onAdd: (versionId: string, label: string) => void;
  onCancel: () => void;
}) {
  const [prompts, setPrompts] = useState<PromptSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedPromptId, setExpandedPromptId] = useState<string | null>(null);
  const [versions, setVersions] = useState<PromptVersionDetail[]>([]);
  const [versionsLoading, setVersionsLoading] = useState(false);

  useEffect(() => {
    listPrompts()
      .then(setPrompts)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleExpandPrompt = async (promptId: string) => {
    if (expandedPromptId === promptId) {
      setExpandedPromptId(null);
      return;
    }
    setExpandedPromptId(promptId);
    setVersionsLoading(true);
    try {
      const detail = await getPrompt(promptId);
      setVersions(detail.versions);
    } catch (err) {
      console.error("Failed to load versions:", err);
      setVersions([]);
    } finally {
      setVersionsLoading(false);
    }
  };

  return (
    <div className="border border-[var(--border)] rounded-md bg-[var(--bg-desk)] overflow-hidden">
      <div className="flex items-center justify-between px-2.5 py-1.5 bg-[var(--bg-surface)]">
        <span className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider">
          Select a published prompt
        </span>
        <button
          onClick={onCancel}
          className="text-[10px] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
        >
          Cancel
        </button>
      </div>

      <div className="max-h-[200px] overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-4">
            <Loader2 size={14} className="animate-spin text-[var(--text-secondary)]" />
          </div>
        ) : prompts.length === 0 ? (
          <div className="px-3 py-4 text-[11px] text-[var(--text-secondary)] text-center">
            No prompts available. Create and publish a prompt first.
          </div>
        ) : (
          prompts.map((p) => (
            <div key={p.id}>
              <button
                onClick={() => handleExpandPrompt(p.id)}
                className="w-full flex items-center gap-2 px-2.5 py-1.5 text-left
                  hover:bg-[var(--bg-surface)] transition-colors duration-75"
              >
                <ChevronRight
                  size={10}
                  className={`text-[var(--text-secondary)] transition-transform duration-100
                    ${expandedPromptId === p.id ? "rotate-90" : ""}`}
                />
                <span className="text-[11px] text-[var(--text-primary)] truncate">{p.name}</span>
                <span className="text-[10px] text-[var(--text-secondary)] ml-auto opacity-60">
                  {p.latest_version > 0 ? `v${p.latest_version}` : "no versions"}
                </span>
              </button>

              {expandedPromptId === p.id && (
                <div className="pl-6 pb-1">
                  {versionsLoading ? (
                    <Loader2 size={12} className="animate-spin text-[var(--text-secondary)] my-1" />
                  ) : versions.length === 0 ? (
                    <div className="text-[10px] text-[var(--text-secondary)] py-1">
                      No published versions
                    </div>
                  ) : (
                    versions.map((v) => (
                      <button
                        key={v.id}
                        onClick={() => onAdd(v.id, `${p.name} v${v.version}`)}
                        className="w-full flex items-center gap-2 px-2 py-1 rounded text-left
                          hover:bg-[var(--bg-surface)] transition-colors duration-75"
                      >
                        <span className="text-[10px] font-mono text-[var(--accent)]">
                          v{v.version}
                        </span>
                        <span className="text-[10px] text-[var(--text-secondary)]">
                          {v.context_source} / {v.output_format}
                        </span>
                      </button>
                    ))
                  )}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// ─── Workflow Editor ─────────────────────────────────────────────────

function WorkflowEditor({
  workflowId,
  onBack,
}: {
  workflowId: string;
  onBack: () => void;
}) {
  const [workflow, setWorkflow] = useState<WorkflowDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [showPicker, setShowPicker] = useState(false);
  const [runLoading, setRunLoading] = useState(false);
  const [runResult, setRunResult] = useState<WorkflowRunDetail | null>(null);
  const selectedDocHash = useDocStore((s) => s.selectedDocHash);

  const loadWorkflow = async () => {
    setLoading(true);
    try {
      const detail = await getWorkflow(workflowId);
      setWorkflow(detail);
    } catch (err) {
      console.error("Failed to load workflow:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadWorkflow();
  }, [workflowId]);

  const handleAddStep = async (versionId: string, label: string) => {
    setShowPicker(false);
    const stepNum = (workflow?.steps.length ?? 0) + 1;
    const outputKey = `step_${stepNum}`;
    try {
      await addWorkflowStep(workflowId, {
        prompt_version_id: versionId,
        label,
        output_key: outputKey,
      });
      await loadWorkflow();
    } catch (err) {
      console.error("Failed to add step:", err);
    }
  };

  const handleRemoveStep = async (stepId: string) => {
    try {
      await removeWorkflowStep(workflowId, stepId);
      await loadWorkflow();
    } catch (err) {
      console.error("Failed to remove step:", err);
    }
  };

  const handleRun = async () => {
    setRunLoading(true);
    setRunResult(null);
    try {
      const result = await runWorkflow(workflowId, {
        doc_filter: selectedDocHash ?? undefined,
      });
      setRunResult(result);
    } catch (err) {
      console.error("Failed to run workflow:", err);
    } finally {
      setRunLoading(false);
    }
  };

  const handleApprove = async () => {
    if (!runResult) return;
    setRunLoading(true);
    try {
      const result = await approveWorkflowRun(runResult.id);
      setRunResult(result);
    } catch (err) {
      console.error("Failed to approve:", err);
    } finally {
      setRunLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 size={20} className="animate-spin text-[var(--text-secondary)]" />
      </div>
    );
  }

  if (!workflow) {
    return (
      <div className="flex items-center justify-center h-full text-[12px] text-[var(--text-secondary)]">
        Workflow not found
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-[var(--border-subtle)] shrink-0">
        <button
          onClick={onBack}
          className="p-1 rounded hover:bg-[var(--bg-surface)] transition-colors duration-75"
        >
          <ArrowLeft size={14} className="text-[var(--text-secondary)]" />
        </button>
        <span className="flex-1 text-[12px] text-[var(--text-primary)] truncate">
          {workflow.name}
        </span>
        <span className="text-[10px] text-[var(--text-secondary)]">
          {workflow.steps.length} step{workflow.steps.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto">
        {/* Step list */}
        <div className="px-3 py-2">
          <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider mb-1.5 block">
            Steps
          </label>

          {workflow.steps.length === 0 ? (
            <div className="text-[11px] text-[var(--text-secondary)] py-3 text-center opacity-60">
              No steps yet. Add a published prompt to start.
            </div>
          ) : (
            <div className="space-y-1">
              {workflow.steps.map((step, idx) => (
                <div
                  key={step.id}
                  className="flex items-center gap-2 px-2.5 py-2 rounded
                    bg-[var(--bg-desk)] border border-[var(--border)] group"
                >
                  <GripVertical size={12} className="text-[var(--text-secondary)] opacity-40 shrink-0" />
                  <span className="text-[10px] font-mono text-[var(--accent)] shrink-0 w-5">
                    {idx + 1}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="text-[11px] text-[var(--text-primary)] truncate">
                      {step.label}
                    </div>
                    <div className="text-[10px] text-[var(--text-secondary)] opacity-60">
                      {step.prompt_version ? `${step.prompt_version.context_source} / ${step.prompt_version.output_format}` : "?"} → {step.output_key}
                    </div>
                  </div>
                  <button
                    onClick={() => handleRemoveStep(step.id)}
                    className="opacity-0 group-hover:opacity-60 hover:!opacity-100
                      p-0.5 rounded transition-opacity duration-75"
                  >
                    <Trash2 size={11} className="text-[var(--text-secondary)]" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Add step */}
          <div className="mt-2">
            {showPicker ? (
              <StepPicker
                onAdd={handleAddStep}
                onCancel={() => setShowPicker(false)}
              />
            ) : (
              <button
                onClick={() => setShowPicker(true)}
                className="w-full flex items-center justify-center gap-1.5 py-2 rounded
                  border border-dashed border-[var(--border)] text-[11px]
                  text-[var(--text-secondary)] hover:text-[var(--text-primary)]
                  hover:border-[var(--border-sharp)] transition-colors duration-75"
              >
                <Plus size={12} />
                Add Step
              </button>
            )}
          </div>
        </div>

        {/* Run action */}
        <div className="px-3 py-2 border-t border-[var(--border-subtle)] flex items-center gap-2">
          <button
            onClick={handleRun}
            disabled={runLoading || workflow.steps.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded text-[11px]
              bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)]
              active:bg-[var(--accent-pressed)] transition-colors duration-75
              disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {runLoading ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
            Run Workflow
          </button>

          {runResult?.status === "paused_for_approval" && (
            <button
              onClick={handleApprove}
              disabled={runLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-[11px]
                border border-yellow-500/50 text-yellow-400
                hover:bg-yellow-500/10 transition-colors duration-75
                disabled:opacity-40"
            >
              Approve & Continue
            </button>
          )}
        </div>

        {/* Run output */}
        {runResult && (
          <div className="px-3 py-2 border-t border-[var(--border-subtle)]">
            <div className="flex items-center gap-2 mb-2">
              <StatusIcon status={runResult.status} />
              <span className="text-[11px] text-[var(--text-primary)] capitalize">
                {runResult.status.replace(/_/g, " ")}
              </span>
              {runResult.error && (
                <span className="text-[10px] text-red-400 truncate ml-auto">
                  {runResult.error}
                </span>
              )}
            </div>

            {/* Step outputs */}
            {Object.entries(runResult.step_outputs)
              .filter(([k]) => k !== "_metadata")
              .map(([key, output]) => {
                const stepOutput = output as { text?: string; structured?: unknown };
                return (
                  <div
                    key={key}
                    className="mb-2 bg-[var(--bg-desk)] border border-[var(--border)] rounded-md p-2"
                  >
                    <div className="text-[10px] font-mono text-[var(--accent)] mb-1">
                      {key}
                    </div>
                    <pre className="text-[10px] text-[var(--text-primary)] font-mono whitespace-pre-wrap max-h-[150px] overflow-y-auto leading-relaxed">
                      {stepOutput?.structured
                        ? JSON.stringify(stepOutput.structured, null, 2)
                        : stepOutput?.text ?? String(output)
                      }
                    </pre>
                  </div>
                );
              })}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────

export default function WorkflowBuilderPanel() {
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null);

  if (selectedWorkflowId) {
    return (
      <WorkflowEditor
        workflowId={selectedWorkflowId}
        onBack={() => setSelectedWorkflowId(null)}
      />
    );
  }

  return <WorkflowList onSelect={setSelectedWorkflowId} />;
}
