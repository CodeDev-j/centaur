import { create } from "zustand";
import {
  PromptSummary,
  PromptDetail,
  PromptRunResult,
  PromptVariable,
} from "@/lib/api";

// ─── Draft state (Zustand only — never persisted until Publish) ──────

export interface DraftState {
  template: string;
  variables: PromptVariable[];
  exec_mode: string;
  output_schema: Record<string, unknown> | null;
  model_id: string | null;
  retrieval_mode: string | null;
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
  studioTab: "prompts" | "workflows";
  view: "library" | "editor";

  // ── Actions ───────────────────────────────────────
  setPrompts: (prompts: PromptSummary[]) => void;
  setPromptsLoading: (v: boolean) => void;
  selectPrompt: (id: string | null) => void;
  setSelectedPrompt: (detail: PromptDetail | null) => void;
  setSelectedPromptLoading: (v: boolean) => void;

  // Draft editing
  setDraftTemplate: (template: string) => void;
  setDraftExecMode: (mode: string) => void;
  setDraftRetrievalMode: (mode: string | null) => void;
  setDraftModelId: (id: string | null) => void;
  setDraftOutputSchema: (schema: Record<string, unknown> | null) => void;
  setDraftVariables: (vars: PromptVariable[]) => void;
  loadDraftFromVersion: (version: {
    template: string;
    variables: PromptVariable[];
    exec_mode: string;
    output_schema: Record<string, unknown> | null;
    model_id: string | null;
    retrieval_mode: string | null;
  }) => void;
  markDraftClean: () => void;

  // Run
  setRunResult: (result: PromptRunResult | null) => void;
  setRunLoading: (v: boolean) => void;
  setRunVariable: (key: string, value: string) => void;
  clearRunVariables: () => void;
  setHasRetrievalCache: (v: boolean) => void;

  // Navigation
  setStudioTab: (tab: "prompts" | "workflows") => void;
  setView: (view: "library" | "editor") => void;
  goBackToLibrary: () => void;
}

// ─── Default draft ───────────────────────────────────────────────────

const DEFAULT_DRAFT: DraftState = {
  template: "",
  variables: [],
  exec_mode: "rag",
  output_schema: null,
  model_id: null,
  retrieval_mode: "qualitative",
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
  setDraftExecMode: (mode) =>
    set((s) => ({ draft: { ...s.draft, exec_mode: mode }, draftDirty: true })),
  setDraftRetrievalMode: (mode) =>
    set((s) => ({ draft: { ...s.draft, retrieval_mode: mode }, draftDirty: true })),
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
        exec_mode: version.exec_mode,
        output_schema: version.output_schema,
        model_id: version.model_id,
        retrieval_mode: version.retrieval_mode,
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
