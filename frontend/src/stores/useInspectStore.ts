import { create } from "zustand";
import { ChunkDetail, DocStats } from "@/lib/api";

interface InspectState {
  inspectMode: boolean;
  inspectChunks: ChunkDetail[] | null;
  activeChunkId: string | null;
  docStats: DocStats | null;
  inspectLoading: boolean;

  setInspectMode: (v: boolean) => void;
  setInspectChunks: (chunks: ChunkDetail[] | null) => void;
  setActiveChunkId: (id: string | null) => void;
  setDocStats: (stats: DocStats | null) => void;
  setInspectLoading: (v: boolean) => void;
  resetInspect: () => void;
}

export const useInspectStore = create<InspectState>((set) => ({
  inspectMode: false,
  inspectChunks: null,
  activeChunkId: null,
  docStats: null,
  inspectLoading: false,

  setInspectMode: (v) => set({ inspectMode: v, activeChunkId: null }),
  setInspectChunks: (chunks) => set({ inspectChunks: chunks }),
  setActiveChunkId: (id) => set({ activeChunkId: id }),
  setDocStats: (stats) => set({ docStats: stats }),
  setInspectLoading: (v) => set({ inspectLoading: v }),
  resetInspect: () =>
    set({ inspectChunks: null, activeChunkId: null }),
}));
