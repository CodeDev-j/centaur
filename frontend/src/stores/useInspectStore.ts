import { create } from "zustand";
import { ChunkDetail, DocStats } from "@/lib/api";

export type RightPanelTab = "chat" | "inspect" | "explore";

interface ExplorerFilters {
  periodicity?: string;
  accounting_basis?: string;
  series_nature?: string;
}

interface InspectState {
  // Right panel tab (replaces inspectMode boolean)
  rightPanelTab: RightPanelTab;
  inspectMode: boolean; // derived getter for backward compat

  // Chunk inspector state
  inspectChunks: ChunkDetail[] | null;
  activeChunkId: string | null;
  docStats: DocStats | null;
  inspectLoading: boolean;

  // Metric explorer state
  explorerChunks: ChunkDetail[] | null;
  explorerSummaryChunks: ChunkDetail[] | null;
  explorerLoading: boolean;
  explorerFilters: ExplorerFilters;
  explorerSearchQuery: string;
  explorerActiveRowIdx: number | null;
  explorerFlatRowCount: number;

  // Actions
  setRightPanelTab: (tab: RightPanelTab) => void;
  setInspectMode: (v: boolean) => void; // backward-compat shim
  setInspectChunks: (chunks: ChunkDetail[] | null) => void;
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
  resetInspect: () => void;
  resetExplorer: () => void;
}

export const useInspectStore = create<InspectState>((set) => ({
  rightPanelTab: "chat",
  inspectMode: false,

  inspectChunks: null,
  activeChunkId: null,
  docStats: null,
  inspectLoading: false,

  explorerChunks: null,
  explorerSummaryChunks: null,
  explorerLoading: false,
  explorerFilters: {},
  explorerSearchQuery: "",
  explorerActiveRowIdx: null,
  explorerFlatRowCount: 0,

  setRightPanelTab: (tab) =>
    set({ rightPanelTab: tab, inspectMode: tab === "inspect", activeChunkId: null }),
  setInspectMode: (v) =>
    set({ rightPanelTab: v ? "inspect" : "chat", inspectMode: v, activeChunkId: null }),

  setInspectChunks: (chunks) => set({ inspectChunks: chunks }),
  setActiveChunkId: (id) => set({ activeChunkId: id }),
  setDocStats: (stats) => set({ docStats: stats }),
  setInspectLoading: (v) => set({ inspectLoading: v }),

  setExplorerChunks: (chunks) => set({ explorerChunks: chunks }),
  setExplorerSummaryChunks: (chunks) => set({ explorerSummaryChunks: chunks }),
  setExplorerLoading: (v) => set({ explorerLoading: v }),
  setExplorerFilter: (key, value) =>
    set((s) => ({ explorerFilters: { ...s.explorerFilters, [key]: value } })),
  setExplorerSearchQuery: (query) => set({ explorerSearchQuery: query, explorerActiveRowIdx: null }),
  setExplorerActiveRowIdx: (idx) => set({ explorerActiveRowIdx: idx }),
  setExplorerFlatRowCount: (count) => set({ explorerFlatRowCount: count }),

  resetInspect: () => set({ inspectChunks: null, activeChunkId: null }),
  resetExplorer: () => set({
    explorerChunks: null,
    explorerSummaryChunks: null,
    explorerFilters: {},
    explorerSearchQuery: "",
    explorerActiveRowIdx: null,
    explorerFlatRowCount: 0,
  }),
}));
