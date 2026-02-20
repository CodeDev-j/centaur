import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ViewerState {
  currentPage: number;
  numPages: number;
  zoomScale: number | null; // null = fit-width
  renderedSize: { width: number; height: number };
  sidebarCollapsed: boolean;
  sidebarManualOverride: boolean; // true after user manually toggles (resets each session)

  setCurrentPage: (page: number) => void;
  setNumPages: (n: number) => void;
  setZoomScale: (scale: number | null) => void;
  setRenderedSize: (size: { width: number; height: number }) => void;
  setSidebarCollapsed: (v: boolean) => void;
  toggleSidebar: () => void;
  resetForNewDoc: () => void;
}

export const useViewerStore = create<ViewerState>()(
  persist(
    (set) => ({
      currentPage: 1,
      numPages: 0,
      zoomScale: null,
      renderedSize: { width: 0, height: 0 },
      sidebarCollapsed: false,
      sidebarManualOverride: false,

      setCurrentPage: (page) => set({ currentPage: page }),
      setNumPages: (n) => set({ numPages: n }),
      setZoomScale: (scale) => set({ zoomScale: scale }),
      setRenderedSize: (size) => set({ renderedSize: size }),
      setSidebarCollapsed: (v) => set({ sidebarCollapsed: v }),
      toggleSidebar: () =>
        set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed, sidebarManualOverride: true })),

      resetForNewDoc: () =>
        set({ currentPage: 1, zoomScale: null, numPages: 0 }),
    }),
    {
      name: "centaur-viewer",
      partialize: (state) => ({ sidebarCollapsed: state.sidebarCollapsed }),
    }
  )
);
