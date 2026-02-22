"use client";

import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  Plus,
  Play,
  Save,
  Loader2,
  FileText,
  Trash2,
  RefreshCw,
  Zap,
} from "lucide-react";
import { useStudioStore, isValidCombination } from "@/stores/useStudioStore";
import { useDocStore } from "@/stores/useDocStore";
import WorkflowBuilderPanel from "./WorkflowBuilder";
import ToolsPanel from "./ToolsPanel";
import SegmentedControl from "./SegmentedControl";
import ChatVizBlock from "./ChatVizBlock";
import {
  listPrompts,
  createPrompt,
  getPrompt,
  archivePrompt,
  publishVersion,
  runPrompt,
  PromptSummary,
  ContextSource,
  OutputFormat,
} from "@/lib/api";

// ─── Jinja2 variable extraction ──────────────────────────────────────

const JINJA_VAR_RE = /\{\{\s*([\w.]+)\s*\}\}/g;

function extractVariables(template: string): string[] {
  const vars = new Set<string>();
  let match;
  while ((match = JINJA_VAR_RE.exec(template)) !== null) {
    // Skip step references like steps.1.text — those are workflow-level
    if (!match[1].startsWith("steps.")) {
      vars.add(match[1]);
    }
  }
  return Array.from(vars);
}

// ─── Axis options ───────────────────────────────────────────────────

const CONTEXT_SOURCES = [
  { value: "documents", label: "Documents" },
  { value: "metrics_db", label: "Metrics DB" },
  { value: "none", label: "None" },
];

const OUTPUT_FORMATS = [
  { value: "text", label: "Text" },
  { value: "json", label: "JSON" },
  { value: "chart", label: "Chart" },
  { value: "table", label: "Table" },
];

const SEARCH_STRATEGIES = [
  { value: "semantic", label: "Semantic" },
  { value: "numeric", label: "Numeric" },
];

// ─── Library View ────────────────────────────────────────────────────

function PromptLibrary() {
  const prompts = useStudioStore((s) => s.prompts);
  const loading = useStudioStore((s) => s.promptsLoading);
  const [creating, setCreating] = useState(false);

  // Fetch prompts on mount
  useEffect(() => {
    loadPrompts();
  }, []);

  const loadPrompts = async () => {
    useStudioStore.getState().setPromptsLoading(true);
    try {
      const data = await listPrompts();
      useStudioStore.getState().setPrompts(data);
    } catch (err) {
      console.error("Failed to load prompts:", err);
    } finally {
      useStudioStore.getState().setPromptsLoading(false);
    }
  };

  const handleCreate = async () => {
    setCreating(true);
    try {
      const prompt = await createPrompt({
        name: "Untitled Prompt",
        category: "custom",
      });
      await loadPrompts();
      // Open the new prompt in editor
      handleSelect(prompt.id);
    } catch (err) {
      console.error("Failed to create prompt:", err);
    } finally {
      setCreating(false);
    }
  };

  const handleSelect = async (id: string) => {
    const store = useStudioStore.getState();
    store.selectPrompt(id);
    store.setView("editor");
    try {
      const detail = await getPrompt(id);
      useStudioStore.getState().setSelectedPrompt(detail);
      // Load latest version into draft (or empty for new prompts)
      if (detail.versions.length > 0) {
        const latest = detail.versions[detail.versions.length - 1];
        useStudioStore.getState().loadDraftFromVersion(latest);
      } else {
        useStudioStore.getState().loadDraftFromVersion({
          template: "",
          variables: [],
          context_source: "documents",
          output_format: "text",
          search_strategy: ["semantic"],
          output_schema: null,
          model_id: null,
          temperature: 0.1,
        });
      }
    } catch (err) {
      console.error("Failed to load prompt:", err);
      // Return to library on error instead of showing "Prompt not found"
      useStudioStore.getState().goBackToLibrary();
    } finally {
      useStudioStore.getState().setSelectedPromptLoading(false);
    }
  };

  const handleArchive = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await archivePrompt(id);
      await loadPrompts();
    } catch (err) {
      console.error("Failed to archive prompt:", err);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-[var(--border-subtle)]">
        <span className="text-h2 text-[var(--text-primary)]">PROMPT LIBRARY</span>
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

      {/* Prompt list */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 size={16} className="animate-spin text-[var(--text-secondary)]" />
          </div>
        ) : prompts.length === 0 ? (
          <div className="px-4 py-8 text-center">
            <p className="text-[12px] text-[var(--text-secondary)]">No prompts yet</p>
            <p className="text-[11px] text-[var(--text-secondary)] mt-1 opacity-60">
              Create a prompt to get started
            </p>
          </div>
        ) : (
          <div className="py-1">
            {prompts.map((p) => (
              <PromptRow
                key={p.id}
                prompt={p}
                onSelect={() => handleSelect(p.id)}
                onArchive={(e) => handleArchive(e, p.id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function PromptRow({
  prompt,
  onSelect,
  onArchive,
}: {
  prompt: PromptSummary;
  onSelect: () => void;
  onArchive: (e: React.MouseEvent) => void;
}) {
  return (
    <div
      onClick={onSelect}
      role="button"
      tabIndex={0}
      className="w-full flex items-start gap-2.5 px-3 py-2 text-left cursor-pointer
        hover:bg-[var(--bg-surface)] transition-colors duration-75 group"
    >
      <FileText
        size={14}
        className="text-[var(--text-secondary)] mt-0.5 shrink-0"
      />
      <div className="flex-1 min-w-0">
        <div className="text-[12px] text-[var(--text-primary)] truncate">
          {prompt.name}
        </div>
        {prompt.description && (
          <div className="text-[11px] text-[var(--text-secondary)] truncate mt-0.5">
            {prompt.description}
          </div>
        )}
        <div className="flex items-center gap-2 mt-1">
          <span className="text-[10px] text-[var(--text-secondary)] opacity-60">
            {prompt.category}
          </span>
          {prompt.latest_version > 0 && (
            <span className="text-[10px] text-[var(--text-secondary)] opacity-60">
              v{prompt.latest_version}
            </span>
          )}
        </div>
      </div>
      <button
        onClick={onArchive}
        className="opacity-0 group-hover:opacity-60 hover:!opacity-100
          p-1 rounded hover:bg-[var(--bg-tertiary)] transition-opacity duration-75"
        title="Archive"
      >
        <Trash2 size={12} className="text-[var(--text-secondary)]" />
      </button>
    </div>
  );
}

// ─── Editor View ─────────────────────────────────────────────────────

function PromptEditor() {
  const selectedPrompt = useStudioStore((s) => s.selectedPrompt);
  const selectedPromptLoading = useStudioStore((s) => s.selectedPromptLoading);
  const draft = useStudioStore((s) => s.draft);
  const draftDirty = useStudioStore((s) => s.draftDirty);
  const runResult = useStudioStore((s) => s.runResult);
  const runLoading = useStudioStore((s) => s.runLoading);
  const runVariables = useStudioStore((s) => s.runVariables);
  const hasRetrievalCache = useStudioStore((s) => s.hasRetrievalCache);
  const selectedDocHash = useDocStore((s) => s.selectedDocHash);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [nameEditing, setNameEditing] = useState(false);
  const [nameValue, setNameValue] = useState("");
  const [publishing, setPublishing] = useState(false);

  // Extract Jinja2 variables from template
  const detectedVars = extractVariables(draft.template);

  // Build output format options with disabled states
  const outputFormatOptions = OUTPUT_FORMATS.map((opt) => ({
    ...opt,
    disabled: !isValidCombination(draft.context_source, opt.value as OutputFormat),
    tooltip: !isValidCombination(draft.context_source, opt.value as OutputFormat)
      ? `${opt.label} requires Metrics DB as context source`
      : undefined,
  }));

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = `${Math.max(ta.scrollHeight, 200)}px`;
    }
  }, [draft.template]);

  const handleBack = () => {
    useStudioStore.getState().goBackToLibrary();
  };

  const handleNameSave = async () => {
    if (!selectedPrompt || !nameValue.trim()) return;
    try {
      const { updatePrompt } = await import("@/lib/api");
      await updatePrompt(selectedPrompt.id, { name: nameValue.trim() });
      // Refresh prompt
      const detail = await getPrompt(selectedPrompt.id);
      useStudioStore.getState().setSelectedPrompt(detail);
    } catch (err) {
      console.error("Failed to update name:", err);
    }
    setNameEditing(false);
  };

  const handlePublish = async () => {
    if (!selectedPrompt) return;
    setPublishing(true);
    try {
      await publishVersion(selectedPrompt.id, {
        template: draft.template,
        variables: draft.variables,
        context_source: draft.context_source,
        output_format: draft.output_format,
        search_strategy: draft.search_strategy,
        output_schema: draft.output_schema,
        model_id: draft.model_id,
        temperature: draft.temperature,
      });
      // Refresh prompt to get new version
      const detail = await getPrompt(selectedPrompt.id);
      useStudioStore.getState().setSelectedPrompt(detail);
      useStudioStore.getState().markDraftClean();
      // Refresh library list
      const prompts = await listPrompts();
      useStudioStore.getState().setPrompts(prompts);
    } catch (err) {
      console.error("Failed to publish:", err);
    } finally {
      setPublishing(false);
    }
  };

  const handleRun = async (forceRetrieve = false) => {
    if (!selectedPrompt) return;
    useStudioStore.getState().setRunLoading(true);
    useStudioStore.getState().setRunResult(null);
    try {
      const result = await runPrompt(selectedPrompt.id, {
        template: draft.template,
        variables: runVariables,
        context_source: draft.context_source,
        output_format: draft.output_format,
        search_strategy: draft.search_strategy,
        output_schema: draft.output_schema ?? undefined,
        doc_filter: selectedDocHash ?? undefined,
        model_id: draft.model_id,
        temperature: draft.temperature,
        force_retrieve: forceRetrieve,
      });
      useStudioStore.getState().setRunResult(result);
      // Track cache state for refinement loop
      if (draft.context_source === "documents" && !result.metadata.error) {
        useStudioStore.getState().setHasRetrievalCache(true);
      }
    } catch (err) {
      console.error("Failed to run prompt:", err);
      useStudioStore.getState().setRunResult({
        text: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
        structured: null,
        metadata: { error: true },
      });
    } finally {
      useStudioStore.getState().setRunLoading(false);
    }
  };

  if (selectedPromptLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 size={20} className="animate-spin text-[var(--text-secondary)]" />
      </div>
    );
  }

  if (!selectedPrompt) {
    return (
      <div className="flex items-center justify-center h-full text-[12px] text-[var(--text-secondary)]">
        Prompt not found
      </div>
    );
  }

  // Check if run result contains a chart spec
  const hasVizSpec = runResult?.structured?.spec != null;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-[var(--border-subtle)] shrink-0">
        <button
          onClick={handleBack}
          className="p-1 rounded hover:bg-[var(--bg-surface)] transition-colors duration-75"
          title="Back to library"
        >
          <ArrowLeft size={14} className="text-[var(--text-secondary)]" />
        </button>

        {nameEditing ? (
          <input
            autoFocus
            value={nameValue}
            onChange={(e) => setNameValue(e.target.value)}
            onBlur={handleNameSave}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleNameSave();
              if (e.key === "Escape") setNameEditing(false);
            }}
            className="flex-1 text-[12px] bg-transparent border-b border-[var(--accent)]
              text-[var(--text-primary)] outline-none px-1 py-0.5"
          />
        ) : (
          <button
            onClick={() => {
              setNameValue(selectedPrompt.name);
              setNameEditing(true);
            }}
            className="flex-1 text-[12px] text-[var(--text-primary)] text-left truncate
              hover:text-[var(--accent)] transition-colors duration-75"
            title="Click to rename"
          >
            {selectedPrompt.name}
          </button>
        )}

        {draftDirty && (
          <span className="text-[10px] text-[var(--accent)] shrink-0">unsaved</span>
        )}
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto">
        {/* Context Source selector */}
        <div className="px-3 py-2 border-b border-[var(--border-subtle)]">
          <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider mb-1.5 block">
            Context Source
          </label>
          <SegmentedControl
            options={CONTEXT_SOURCES}
            value={draft.context_source}
            onChange={(v) => useStudioStore.getState().setDraftContextSource(v as ContextSource)}
          />
        </div>

        {/* Template editor */}
        <div className="px-3 py-2">
          <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider mb-1 block">
            Template
          </label>
          <textarea
            ref={textareaRef}
            value={draft.template}
            onChange={(e) => useStudioStore.getState().setDraftTemplate(e.target.value)}
            placeholder="Enter your prompt template... Use {{ variable_name }} for Jinja2 variables."
            spellCheck={false}
            className="w-full min-h-[200px] bg-[var(--bg-desk)] border border-[var(--border)]
              rounded-md px-3 py-2 text-[12px] text-[var(--text-primary)] font-mono
              leading-relaxed resize-none outline-none
              focus:border-[var(--accent)] transition-colors duration-100
              placeholder:text-[var(--text-secondary)] placeholder:opacity-40"
          />

          {/* Detected variables */}
          {detectedVars.length > 0 && (
            <div className="mt-2 flex items-center gap-1.5 flex-wrap">
              <span className="text-[10px] text-[var(--text-secondary)]">Variables:</span>
              {detectedVars.map((v) => (
                <span
                  key={v}
                  className="text-[10px] px-1.5 py-0.5 rounded
                    bg-[var(--highlight)] text-[var(--accent)] border border-[var(--accent)]
                    border-opacity-30"
                >
                  {v}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Variable inputs (for test run) */}
        {detectedVars.length > 0 && (
          <div className="px-3 py-2 border-t border-[var(--border-subtle)]">
            <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider mb-1.5 block">
              Test Variables
            </label>
            <div className="space-y-1.5">
              {detectedVars.map((v) => (
                <div key={v} className="flex items-center gap-2">
                  <span className="text-[11px] text-[var(--text-secondary)] font-mono w-24 shrink-0 truncate">
                    {v}
                  </span>
                  <input
                    type="text"
                    value={runVariables[v] ?? ""}
                    onChange={(e) =>
                      useStudioStore.getState().setRunVariable(v, e.target.value)
                    }
                    placeholder={`Value for ${v}`}
                    className="flex-1 text-[11px] bg-[var(--bg-desk)] border border-[var(--border)]
                      rounded px-2 py-1 text-[var(--text-primary)] outline-none
                      focus:border-[var(--accent)] transition-colors duration-100
                      placeholder:text-[var(--text-secondary)] placeholder:opacity-40"
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Output Format selector */}
        <div className="px-3 py-2 border-t border-[var(--border-subtle)]">
          <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider mb-1.5 block">
            Output Format
          </label>
          <SegmentedControl
            options={outputFormatOptions}
            value={draft.output_format}
            onChange={(v) => useStudioStore.getState().setDraftOutputFormat(v as OutputFormat)}
          />
        </div>

        {/* JSON Schema editor (conditional) */}
        {draft.output_format === "json" && (
          <div className="px-3 py-2 border-t border-[var(--border-subtle)]">
            <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider mb-1.5 block">
              JSON Output Schema
            </label>
            <textarea
              value={draft.output_schema ? JSON.stringify(draft.output_schema, null, 2) : ""}
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value);
                  useStudioStore.getState().setDraftOutputSchema(parsed);
                } catch {
                  // Allow typing — only update store when valid JSON
                }
              }}
              placeholder='{"key": "string", "values": ["number"]}'
              spellCheck={false}
              className="w-full min-h-[80px] bg-[var(--bg-desk)] border border-[var(--border)]
                rounded-md px-3 py-2 text-[11px] text-[var(--text-primary)] font-mono
                resize-none outline-none
                focus:border-[var(--accent)] transition-colors duration-100
                placeholder:text-[var(--text-secondary)] placeholder:opacity-40"
            />
          </div>
        )}

        {/* Advanced Parameters (collapsible) */}
        <details className="border-t border-[var(--border-subtle)]">
          <summary className="px-3 py-2 text-[10px] text-[var(--text-secondary)] uppercase tracking-wider cursor-pointer hover:text-[var(--text-primary)] select-none">
            Advanced Parameters
          </summary>
          <div className="px-3 pb-3 space-y-3">
            {/* Model override */}
            <div>
              <label className="text-[10px] text-[var(--text-secondary)] mb-1 block">
                Model Override
              </label>
              <input
                type="text"
                value={draft.model_id ?? ""}
                onChange={(e) =>
                  useStudioStore.getState().setDraftModelId(e.target.value || null)
                }
                placeholder="Default (from config)"
                className="w-full text-[11px] bg-[var(--bg-desk)] border border-[var(--border)]
                  rounded px-2 py-1 text-[var(--text-primary)] outline-none
                  focus:border-[var(--accent)] transition-colors duration-100
                  placeholder:text-[var(--text-secondary)] placeholder:opacity-40"
              />
            </div>

            {/* Temperature */}
            <div>
              <label className="text-[10px] text-[var(--text-secondary)] mb-1 block">
                Temperature: {draft.temperature.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={draft.temperature}
                onChange={(e) =>
                  useStudioStore.getState().setDraftTemperature(parseFloat(e.target.value))
                }
                className="w-full h-1 accent-[var(--accent)]"
              />
            </div>

            {/* Search Strategy (only when source=documents) */}
            {draft.context_source === "documents" && (
              <div>
                <label className="text-[10px] text-[var(--text-secondary)] mb-1.5 block">
                  Search Strategy
                </label>
                <SegmentedControl
                  options={SEARCH_STRATEGIES}
                  value={draft.search_strategy}
                  onChange={(v) =>
                    useStudioStore.getState().toggleSearchStrategy(v as "semantic" | "numeric")
                  }
                  multi
                />
              </div>
            )}
          </div>
        </details>

        {/* Action bar */}
        <div className="px-3 py-2 border-t border-[var(--border-subtle)] flex items-center gap-2 flex-wrap">
          <button
            onClick={() => handleRun(false)}
            disabled={runLoading || !draft.template.trim()}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded text-[11px]
              bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)]
              active:bg-[var(--accent-pressed)] transition-colors duration-75
              disabled:opacity-40 disabled:cursor-not-allowed"
            title={hasRetrievalCache ? "Re-generate with cached chunks (fast)" : "Retrieve + Generate"}
          >
            {runLoading ? (
              <Loader2 size={12} className="animate-spin" />
            ) : hasRetrievalCache ? (
              <Zap size={12} />
            ) : (
              <Play size={12} />
            )}
            {hasRetrievalCache ? "Re-generate" : "Run"}
          </button>

          {hasRetrievalCache && (
            <button
              onClick={() => handleRun(true)}
              disabled={runLoading || !draft.template.trim()}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded text-[11px]
                border border-[var(--border-sharp)] text-[var(--text-secondary)]
                hover:bg-[var(--bg-surface)] hover:text-[var(--text-primary)]
                transition-colors duration-75
                disabled:opacity-40 disabled:cursor-not-allowed"
              title="Bust cache and re-retrieve from vector store"
            >
              <RefreshCw size={11} />
              Re-retrieve
            </button>
          )}

          <button
            onClick={handlePublish}
            disabled={publishing || !draft.template.trim()}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded text-[11px]
              border border-[var(--border-sharp)] text-[var(--text-primary)]
              hover:bg-[var(--bg-surface)] active:bg-[var(--bg-tertiary)]
              transition-colors duration-75
              disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {publishing ? (
              <Loader2 size={12} className="animate-spin" />
            ) : (
              <Save size={12} />
            )}
            Publish v{(selectedPrompt.versions?.length ?? 0) + 1}
          </button>

          {selectedDocHash && (
            <span className="text-[10px] text-[var(--text-secondary)] ml-auto truncate max-w-[120px]"
              title="Run will be scoped to this document"
            >
              Scoped: {useDocStore.getState().selectedFilename}
            </span>
          )}
        </div>

        {/* Output viewer */}
        {(runResult || runLoading) && (
          <div className="px-3 py-2 border-t border-[var(--border-subtle)]">
            <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider mb-1.5 block">
              Output
              {runResult?.metadata && !runResult.metadata.error && (
                <span className="ml-2 normal-case opacity-60">
                  {String(runResult.metadata.duration_ms ?? "")}ms
                  {runResult.metadata.chunks_retrieved != null &&
                    ` · ${String(runResult.metadata.chunks_retrieved)} chunks`
                  }
                  {runResult.metadata.cache_hit === true && " · cached"}
                </span>
              )}
            </label>

            {runLoading ? (
              <div className="flex items-center gap-2 py-4 text-[11px] text-[var(--text-secondary)]">
                <Loader2 size={14} className="animate-spin" />
                Executing...
              </div>
            ) : runResult ? (
              <>
                {/* Chart rendering for viz specs */}
                {hasVizSpec && (
                  <ChatVizBlock
                    spec={runResult.structured!.spec as Record<string, unknown>}
                    data={(runResult.structured!.data ?? runResult.structured!.rows ?? []) as Record<string, unknown>[]}
                    sql={(runResult.structured!.sql as string) ?? ""}
                    title={(runResult.structured!.title as string) ?? ""}
                  />
                )}

                {/* Text/JSON output */}
                {!hasVizSpec && (
                  <div className="bg-[var(--bg-desk)] border border-[var(--border)] rounded-md p-3">
                    <pre className="text-[11px] text-[var(--text-primary)] font-mono whitespace-pre-wrap leading-relaxed max-h-[300px] overflow-y-auto">
                      {runResult.structured
                        ? JSON.stringify(runResult.structured, null, 2)
                        : runResult.text
                      }
                    </pre>
                  </div>
                )}
              </>
            ) : null}
          </div>
        )}

        {/* Version history */}
        {selectedPrompt.versions && selectedPrompt.versions.length > 0 && (
          <div className="px-3 py-2 border-t border-[var(--border-subtle)]">
            <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider mb-1.5 block">
              Versions ({selectedPrompt.versions.length})
            </label>
            <div className="space-y-1">
              {[...selectedPrompt.versions].reverse().map((v) => (
                <button
                  key={v.id}
                  onClick={() => useStudioStore.getState().loadDraftFromVersion(v)}
                  className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-left
                    hover:bg-[var(--bg-surface)] transition-colors duration-75 group"
                >
                  <span className="text-[11px] text-[var(--text-primary)] font-mono">
                    v{v.version}
                  </span>
                  <span className="text-[10px] text-[var(--text-secondary)]">
                    {v.context_source} / {v.output_format}
                  </span>
                  <span className="text-[10px] text-[var(--text-secondary)] ml-auto opacity-60">
                    {v.created_at
                      ? new Date(v.created_at).toLocaleDateString()
                      : ""
                    }
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Tab Bar ─────────────────────────────────────────────────────────

const STUDIO_TABS = [
  { id: "prompts" as const, label: "PROMPTS" },
  { id: "workflows" as const, label: "WORKFLOWS" },
  { id: "tools" as const, label: "TOOLS" },
];

function StudioTabBar() {
  const studioTab = useStudioStore((s) => s.studioTab);

  return (
    <div className="flex items-center gap-1 px-3 py-1.5 border-b border-[var(--border-subtle)] shrink-0">
      {STUDIO_TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => {
            useStudioStore.getState().setStudioTab(tab.id);
            // Reset prompt view when switching tabs
            if (tab.id === "prompts") {
              useStudioStore.getState().setView("library");
            }
          }}
          className={`px-2.5 py-1 rounded text-[10px] uppercase tracking-wider
            transition-colors duration-75 border
            ${studioTab === tab.id
              ? "bg-[var(--bg-panel)] border-[var(--border-sharp)] text-[var(--text-primary)]"
              : "border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-surface)]"
            }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────

export default function StudioPanel() {
  const studioTab = useStudioStore((s) => s.studioTab);
  const view = useStudioStore((s) => s.view);

  return (
    <div className="flex flex-col h-full">
      <StudioTabBar />
      <div className="flex-1 min-h-0">
        {studioTab === "prompts" ? (
          view === "library" ? <PromptLibrary /> : <PromptEditor />
        ) : studioTab === "workflows" ? (
          <WorkflowBuilderPanel />
        ) : (
          <ToolsPanel />
        )}
      </div>
    </div>
  );
}
