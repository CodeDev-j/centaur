import { create } from "zustand";
import { ChunkDetail, DocStats } from "@/lib/api";

// ─── Panel types ─────────────────────────────────────────────────────

export type PanelId = "chat" | "inspect" | "explore" | "studio";

/** Canonical left-to-right display order */
export const PANEL_ORDER: PanelId[] = ["chat", "inspect", "explore", "studio"];

const DEFAULT_PANEL_WIDTH = 380;
const MIN_PANEL_WIDTH = 240;
const MAX_PANEL_WIDTH = 960;

export type ExplorerViewMode = "list" | "table";
export type ExplorerSortMode = "document" | "statement";

// ─── Interfaces ──────────────────────────────────────────────────────

interface ExplorerFilters {
  periodicity?: string;
  accounting_basis?: string;
  series_nature?: string;
}

interface InspectState {
  // ── Multi-column panel state ─────────────────────────
  openPanels: PanelId[];
  panelLRU: PanelId[];
  lastAutoClosedPanel: PanelId | null;
  panelWidths: Record<PanelId, number>;

  // ── Panel actions ────────────────────────────────────
  togglePanel: (id: PanelId) => void;
  ensurePanelOpen: (id: PanelId) => void;
  setPanelWidth: (id: PanelId, width: number) => void;

  // ── Chunk inspector state ────────────────────────────
  inspectChunks: ChunkDetail[] | null;
  inspectAllChunks: ChunkDetail[] | null;
  inspectTypeFilter: string[];
  activeChunkId: string | null;
  docStats: DocStats | null;
  inspectLoading: boolean;

  // ── Metric explorer state ────────────────────────────
  explorerChunks: ChunkDetail[] | null;
  explorerSummaryChunks: ChunkDetail[] | null;
  explorerLoading: boolean;
  explorerFilters: ExplorerFilters;
  explorerSearchQuery: string;
  explorerActiveRowIdx: number | null;
  explorerFlatRowCount: number;
  explorerViewMode: ExplorerViewMode;
  explorerSortMode: ExplorerSortMode;

  // ── Actions ──────────────────────────────────────────
  setInspectChunks: (chunks: ChunkDetail[] | null) => void;
  setInspectAllChunks: (chunks: ChunkDetail[] | null) => void;
  toggleInspectType: (type: string) => void;
  setActiveChunkId: (id: string | null) => void;
  setDocStats: (stats: DocStats | null) => void;
  setInspectLoading: (v: boolean) => void;
  setExplorerChunks: (chunks: ChunkDetail[] | null) => void;
  setExplorerSummaryChunks: (chunks: ChunkDetail[] | null) => void;
  setExplorerLoading: (v: boolean) => void;
  setExplorerFilter: (key: string, value: string | undefined) => void;
  setExplorerSearchQuery: (query: string) => void;
  setExplorerActiveRowIdx: (idx: number | null) => void;
  setExplorerFlatRowCount: (count: number) => void;
  setExplorerViewMode: (mode: ExplorerViewMode) => void;
  setExplorerSortMode: (mode: ExplorerSortMode) => void;
  resetInspect: () => void;
  resetExplorer: () => void;
}

// ─── Helpers ─────────────────────────────────────────────────────────

function updateLRU(lru: PanelId[], id: PanelId): PanelId[] {
  return [id, ...lru.filter((p) => p !== id)];
}

/** Core logic for opening a panel (used by togglePanel and ensurePanelOpen) */
function openPanel(s: InspectState, id: PanelId) {
  return {
    openPanels: [...s.openPanels, id],
    panelLRU: updateLRU(s.panelLRU, id),
  };
}

// ─── Store ───────────────────────────────────────────────────────────

export { MIN_PANEL_WIDTH, MAX_PANEL_WIDTH };

export const useInspectStore = create<InspectState>((set) => ({
  // Panel state — default: only Chat open
  openPanels: ["chat"],
  panelLRU: ["chat"],
  lastAutoClosedPanel: null,
  panelWidths: {
    chat: DEFAULT_PANEL_WIDTH,
    inspect: DEFAULT_PANEL_WIDTH,
    explore: DEFAULT_PANEL_WIDTH,
    studio: 520,
  },

  togglePanel: (id) =>
    set((s) => {
      const isOpen = s.openPanels.includes(id);

      if (isOpen) {
        // Don't allow closing the last panel
        if (s.openPanels.length <= 1) return s;
        return {
          openPanels: s.openPanels.filter((p) => p !== id),
          panelLRU: s.panelLRU.filter((p) => p !== id),
        };
      }

      return openPanel(s, id);
    }),

  ensurePanelOpen: (id) =>
    set((s) => {
      if (s.openPanels.includes(id)) {
        // Already open — just bump LRU
        return { panelLRU: updateLRU(s.panelLRU, id) };
      }
      return openPanel(s, id);
    }),

  setPanelWidth: (id, width) =>
    set((s) => ({
      panelWidths: {
        ...s.panelWidths,
        [id]: Math.max(MIN_PANEL_WIDTH, Math.min(MAX_PANEL_WIDTH, width)),
      },
    })),

  // Chunk inspector state
  inspectChunks: null,
  inspectAllChunks: null,
  inspectTypeFilter: [],
  activeChunkId: null,
  docStats: null,
  inspectLoading: false,

  // Metric explorer state
  explorerChunks: null,
  explorerSummaryChunks: null,
  explorerLoading: false,
  explorerFilters: {},
  explorerSearchQuery: "",
  explorerActiveRowIdx: null,
  explorerFlatRowCount: 0,
  explorerViewMode: "list",
  explorerSortMode: "document",

  // Actions
  setInspectChunks: (chunks) => set({ inspectChunks: chunks }),
  setInspectAllChunks: (chunks) => set({ inspectAllChunks: chunks }),
  toggleInspectType: (type) =>
    set((s) => {
      const cur = s.inspectTypeFilter;
      if (cur.includes(type)) {
        return { inspectTypeFilter: cur.filter((t) => t !== type) };
      }
      return { inspectTypeFilter: [...cur, type] };
    }),
  setActiveChunkId: (id) => set({ activeChunkId: id }),
  setDocStats: (stats) => set({ docStats: stats }),
  setInspectLoading: (v) => set({ inspectLoading: v }),
  setExplorerChunks: (chunks) => set({ explorerChunks: chunks }),
  setExplorerSummaryChunks: (chunks) => set({ explorerSummaryChunks: chunks }),
  setExplorerLoading: (v) => set({ explorerLoading: v }),
  setExplorerFilter: (key, value) =>
    set((s) => ({ explorerFilters: { ...s.explorerFilters, [key]: value } })),
  setExplorerSearchQuery: (query) =>
    set({ explorerSearchQuery: query, explorerActiveRowIdx: null }),
  setExplorerActiveRowIdx: (idx) => set({ explorerActiveRowIdx: idx }),
  setExplorerFlatRowCount: (count) => set({ explorerFlatRowCount: count }),
  setExplorerViewMode: (mode) => set({ explorerViewMode: mode }),
  setExplorerSortMode: (mode) => set({ explorerSortMode: mode }),

  resetInspect: () =>
    set({ inspectChunks: null, inspectAllChunks: null, inspectTypeFilter: [], activeChunkId: null }),
  resetExplorer: () =>
    set({
      explorerChunks: null,
      explorerSummaryChunks: null,
      explorerFilters: {},
      explorerSearchQuery: "",
      explorerActiveRowIdx: null,
      explorerFlatRowCount: 0,
      explorerViewMode: "list",
      explorerSortMode: "document",
    }),
}));
