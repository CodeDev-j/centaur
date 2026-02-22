import { create } from "zustand";
import {
  PromptSummary,
  PromptDetail,
  PromptRunResult,
  PromptVariable,
  ContextSource,
  OutputFormat,
  SearchStrategy,
} from "@/lib/api";

// ─── Draft state (Zustand only — never persisted until Publish) ──────

export interface DraftState {
  template: string;
  variables: PromptVariable[];
  context_source: ContextSource;
  output_format: OutputFormat;
  search_strategy: SearchStrategy[];
  output_schema: Record<string, unknown> | null;
  model_id: string | null;
  temperature: number;
}

/** Validation: chart/table require metrics_db as context source */
export function isValidCombination(source: ContextSource, format: OutputFormat): boolean {
  if (format === "chart" && source !== "metrics_db") return false;
  if (format === "table" && source !== "metrics_db") return false;
  return true;
}

// ─── Store interface ─────────────────────────────────────────────────

interface StudioState {
  // ── Prompt library ────────────────────────────────
  prompts: PromptSummary[];
  promptsLoading: boolean;
  selectedPromptId: string | null;
  selectedPrompt: PromptDetail | null;
  selectedPromptLoading: boolean;

  // ── Draft editor state (ephemeral) ────────────────
  draft: DraftState;
  draftDirty: boolean;

  // ── Test run ──────────────────────────────────────
  runResult: PromptRunResult | null;
  runLoading: boolean;
  runVariables: Record<string, string>;
  hasRetrievalCache: boolean;

  // ── UI ────────────────────────────────────────────
  studioTab: "prompts" | "workflows" | "tools";
  view: "library" | "editor";

  // ── Actions ───────────────────────────────────────
  setPrompts: (prompts: PromptSummary[]) => void;
  setPromptsLoading: (v: boolean) => void;
  selectPrompt: (id: string | null) => void;
  setSelectedPrompt: (detail: PromptDetail | null) => void;
  setSelectedPromptLoading: (v: boolean) => void;

  // Draft editing
  setDraftTemplate: (template: string) => void;
  setDraftContextSource: (source: ContextSource) => void;
  setDraftOutputFormat: (format: OutputFormat) => void;
  toggleSearchStrategy: (strategy: SearchStrategy) => void;
  setDraftTemperature: (temp: number) => void;
  setDraftModelId: (id: string | null) => void;
  setDraftOutputSchema: (schema: Record<string, unknown> | null) => void;
  setDraftVariables: (vars: PromptVariable[]) => void;
  loadDraftFromVersion: (version: {
    template: string;
    variables: PromptVariable[];
    context_source: ContextSource;
    output_format: OutputFormat;
    search_strategy: SearchStrategy[];
    output_schema: Record<string, unknown> | null;
    model_id: string | null;
    temperature: number;
  }) => void;
  markDraftClean: () => void;

  // Run
  setRunResult: (result: PromptRunResult | null) => void;
  setRunLoading: (v: boolean) => void;
  setRunVariable: (key: string, value: string) => void;
  clearRunVariables: () => void;
  setHasRetrievalCache: (v: boolean) => void;

  // Navigation
  setStudioTab: (tab: "prompts" | "workflows" | "tools") => void;
  setView: (view: "library" | "editor") => void;
  goBackToLibrary: () => void;
}

// ─── Default draft ───────────────────────────────────────────────────

const DEFAULT_DRAFT: DraftState = {
  template: "",
  variables: [],
  context_source: "documents",
  output_format: "text",
  search_strategy: ["semantic"],
  output_schema: null,
  model_id: null,
  temperature: 0.1,
};

// ─── Store ───────────────────────────────────────────────────────────

export const useStudioStore = create<StudioState>((set) => ({
  prompts: [],
  promptsLoading: false,
  selectedPromptId: null,
  selectedPrompt: null,
  selectedPromptLoading: false,

  draft: { ...DEFAULT_DRAFT },
  draftDirty: false,

  runResult: null,
  runLoading: false,
  runVariables: {},
  hasRetrievalCache: false,

  studioTab: "prompts",
  view: "library",

  // Library actions
  setPrompts: (prompts) => set({ prompts }),
  setPromptsLoading: (v) => set({ promptsLoading: v }),
  selectPrompt: (id) =>
    set({
      selectedPromptId: id,
      selectedPrompt: null,
      selectedPromptLoading: true,
      runResult: null,
      runVariables: {},
      draftDirty: false,
      hasRetrievalCache: false,
    }),
  setSelectedPrompt: (detail) => set({ selectedPrompt: detail }),
  setSelectedPromptLoading: (v) => set({ selectedPromptLoading: v }),

  // Draft actions
  setDraftTemplate: (template) =>
    set((s) => ({ draft: { ...s.draft, template }, draftDirty: true })),
  setDraftContextSource: (source) =>
    set((s) => {
      const draft = { ...s.draft, context_source: source };
      // Auto-fix invalid combos: if chart/table and source isn't metrics_db, reset to text
      if (!isValidCombination(source, draft.output_format)) {
        draft.output_format = "text";
      }
      return { draft, draftDirty: true };
    }),
  setDraftOutputFormat: (format) =>
    set((s) => ({ draft: { ...s.draft, output_format: format }, draftDirty: true })),
  toggleSearchStrategy: (strategy) =>
    set((s) => {
      const current = s.draft.search_strategy;
      const has = current.includes(strategy);
      // Don't allow empty — at least one must remain
      if (has && current.length === 1) return {};
      const next = has ? current.filter((s) => s !== strategy) : [...current, strategy];
      return { draft: { ...s.draft, search_strategy: next }, draftDirty: true };
    }),
  setDraftTemperature: (temp) =>
    set((s) => ({ draft: { ...s.draft, temperature: temp }, draftDirty: true })),
  setDraftModelId: (id) =>
    set((s) => ({ draft: { ...s.draft, model_id: id }, draftDirty: true })),
  setDraftOutputSchema: (schema) =>
    set((s) => ({ draft: { ...s.draft, output_schema: schema }, draftDirty: true })),
  setDraftVariables: (vars) =>
    set((s) => ({ draft: { ...s.draft, variables: vars }, draftDirty: true })),
  loadDraftFromVersion: (version) =>
    set({
      draft: {
        template: version.template,
        variables: version.variables,
        context_source: version.context_source,
        output_format: version.output_format,
        search_strategy: version.search_strategy,
        output_schema: version.output_schema,
        model_id: version.model_id,
        temperature: version.temperature,
      },
      draftDirty: false,
      runResult: null,
      runVariables: {},
    }),
  markDraftClean: () => set({ draftDirty: false }),

  // Run actions
  setRunResult: (result) => set({ runResult: result }),
  setRunLoading: (v) => set({ runLoading: v }),
  setRunVariable: (key, value) =>
    set((s) => ({ runVariables: { ...s.runVariables, [key]: value } })),
  clearRunVariables: () => set({ runVariables: {} }),
  setHasRetrievalCache: (v) => set({ hasRetrievalCache: v }),

  // Navigation
  setStudioTab: (tab) => set({ studioTab: tab }),
  setView: (view) => set({ view }),
  goBackToLibrary: () =>
    set({
      view: "library",
      selectedPromptId: null,
      selectedPrompt: null,
      draft: { ...DEFAULT_DRAFT },
      draftDirty: false,
      runResult: null,
      runVariables: {},
      hasRetrievalCache: false,
    }),
}));
