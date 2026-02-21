"use client";

import { useEffect, useState, useMemo } from "react";
import { PanelLeftOpen } from "lucide-react";
import DocumentList from "@/components/DocumentList";
import DocumentViewer from "@/components/DocumentViewer";
import ErrorBoundary from "@/components/ErrorBoundary";
import RightPanel from "@/components/RightPanel";
import { useDocStore } from "@/stores/useDocStore";
import { useViewerStore } from "@/stores/useViewerStore";
import { useChatStore } from "@/stores/useChatStore";
import { useInspectStore } from "@/stores/useInspectStore";
import { fetchDocChunks, fetchDocStats } from "@/lib/api";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";

export default function Home() {
  useKeyboardShortcuts();
  const selectedDocHash = useDocStore((s) => s.selectedDocHash);
  const currentPage = useViewerStore((s) => s.currentPage);
  const sidebarCollapsed = useViewerStore((s) => s.sidebarCollapsed);
  const openPanels = useInspectStore((s) => s.openPanels);

  // H3 fix: Zustand persist rehydrates sidebarCollapsed from localStorage on client
  // only. Render server-default (false) until mounted to avoid hydration mismatch.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const effectiveSidebarCollapsed = mounted ? sidebarCollapsed : false;

  const inspectOpen = openPanels.includes("inspect");
  const exploreOpen = openPanels.includes("explore");
  const chatOpen = openPanels.includes("chat");

  // ---------------------------------------------------------------------------
  // Auto-collapse sidebar on narrow viewports (< 1280px)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const mql = window.matchMedia("(max-width: 1279px)");

    const handleChange = (e: MediaQueryListEvent | MediaQueryList) => {
      const isNarrow = e.matches;
      if (isNarrow && !useViewerStore.getState().sidebarManualOverride) {
        useViewerStore.getState().setSidebarCollapsed(true);
      }
    };

    handleChange(mql);
    mql.addEventListener("change", handleChange);
    return () => mql.removeEventListener("change", handleChange);
  }, []);

  // ---------------------------------------------------------------------------
  // Smart default: selecting a document opens Explorer alongside Chat
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (selectedDocHash) {
      useInspectStore.getState().ensurePanelOpen("explore");
    }
  }, [selectedDocHash]);

  // ---------------------------------------------------------------------------
  // Fetch all document chunks when inspect panel is open and document changes
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!inspectOpen || !selectedDocHash) return;

    let cancelled = false;
    useInspectStore.getState().setInspectLoading(true);
    useInspectStore.getState().setActiveChunkId(null);

    fetchDocChunks(selectedDocHash)
      .then((data) => {
        if (!cancelled) useInspectStore.getState().setInspectAllChunks(data.chunks);
      })
      .catch((err) => {
        if (!cancelled) {
          console.error("Failed to fetch chunks:", err);
          useInspectStore.getState().setInspectAllChunks(null);
        }
      })
      .finally(() => {
        if (!cancelled) useInspectStore.getState().setInspectLoading(false);
      });

    return () => { cancelled = true; };
  }, [inspectOpen, selectedDocHash]);

  // ---------------------------------------------------------------------------
  // Fetch explorer data when explore panel is open and document changes
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!exploreOpen || !selectedDocHash) return;

    let cancelled = false;
    useInspectStore.getState().setExplorerLoading(true);

    Promise.all([
      fetchDocChunks(selectedDocHash, { item_type: "visual", chunk_role: "series" }),
      fetchDocChunks(selectedDocHash, { item_type: "visual", chunk_role: "summary" }),
    ])
      .then(([seriesData, summaryData]) => {
        if (!cancelled) {
          useInspectStore.getState().setExplorerChunks(seriesData.chunks);
          useInspectStore.getState().setExplorerSummaryChunks(summaryData.chunks);
        }
      })
      .catch(() => {
        if (!cancelled) {
          useInspectStore.getState().setExplorerChunks(null);
          useInspectStore.getState().setExplorerSummaryChunks(null);
        }
      })
      .finally(() => {
        if (!cancelled) useInspectStore.getState().setExplorerLoading(false);
      });

    return () => { cancelled = true; };
  }, [exploreOpen, selectedDocHash]);

  // ---------------------------------------------------------------------------
  // Fetch document stats when document changes
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!selectedDocHash) {
      useInspectStore.getState().setDocStats(null);
      return;
    }
    fetchDocStats(selectedDocHash)
      .then((stats) => useInspectStore.getState().setDocStats(stats))
      .catch((err) => {
        console.error("fetchDocStats failed:", err);
        useInspectStore.getState().setDocStats(null);
      });
  }, [selectedDocHash]);

  // ---------------------------------------------------------------------------
  // Compute chunk highlight bboxes for DocumentViewer (P2: memoized)
  // ---------------------------------------------------------------------------
  const activeChunkId = useInspectStore((s) => s.activeChunkId);
  const inspectAllChunks = useInspectStore((s) => s.inspectAllChunks);
  const activeChunk =
    inspectOpen && activeChunkId
      ? inspectAllChunks?.find((c) => c.chunk_id === activeChunkId) ?? null
      : null;

  const highlightBboxes = useMemo(() => {
    if (!inspectOpen || !activeChunk) return null;
    const vb = activeChunk.value_bboxes;
    if (vb && Object.keys(vb).length > 0) {
      // For series chunks, filter to only this series' values + label
      if (activeChunk.chunk_role === "series") {
        const seriesLabel = activeChunk.metadata?.series_label
          ? String(activeChunk.metadata.series_label)
          : null;
        // Extract numeric values from chunk text: "Data: 2018=USD 0.8B, ..."
        const seriesNums: number[] = [];
        const dataMatch = activeChunk.chunk_text?.match(/Data:\s*(.+)/);
        if (dataMatch) {
          const re = /=(?:[A-Z]{3}\s)?(-?[\d.]+)/g;
          let m;
          while ((m = re.exec(dataMatch[1])) !== null) {
            seriesNums.push(parseFloat(m[1]));
          }
        }
        if (seriesLabel || seriesNums.length > 0) {
          // Compare numerically (avoids "17.0" vs "17" mismatch)
          const filtered = Object.entries(vb).filter(([key]) => {
            if (seriesLabel && key === seriesLabel) return true;
            const keyNum = parseFloat(key);
            if (isNaN(keyNum)) return false;
            return seriesNums.some((v) => Math.abs(v - keyNum) < 0.01);
          });
          if (filtered.length > 0) {
            // If chunk has a coarse bbox (chart region), exclude bboxes outside it
            const region = activeChunk.bbox;
            return filtered.flatMap(([, rects]) =>
              rects
                .filter(([bx, by]) =>
                  !region ||
                  (bx >= region.x - 0.02 &&
                    by >= region.y - 0.02 &&
                    bx <= region.x + region.width + 0.02 &&
                    by <= region.y + region.height + 0.02)
                )
                .map(([x, y, width, height]) => ({ x, y, width, height }))
            );
          }
        }
      }
      // Summary/narrative/table chunks: show all bboxes
      return Object.values(vb).flatMap((rects) =>
        rects.map(([x, y, width, height]) => ({ x, y, width, height }))
      );
    }
    if (activeChunk.bbox) return [activeChunk.bbox];
    return null;
  }, [inspectOpen, activeChunk]);

  // Citation overlay (when chat panel is open)
  const activeCitationIndex = useChatStore((s) => s.activeCitationIndex);
  const citations = useChatStore((s) => s.citations);

  return (
    <div className="app-shell h-screen flex">
      {/* Left sidebar: Document list */}
      {effectiveSidebarCollapsed ? (
        <button
          onClick={() => useViewerStore.getState().toggleSidebar()}
          className="sidebar-rail shrink-0"
          title="Show sidebar (Ctrl+B)"
        >
          <span
            className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]"
            style={{ boxShadow: "0 0 6px var(--accent)" }}
          />
          <PanelLeftOpen size={14} className="text-[var(--text-secondary)]" />
        </button>
      ) : (
        <div className="panel-left border-r border-[var(--border)] bg-[var(--bg-secondary)] shrink-0 w-64">
          <DocumentList />
        </div>
      )}

      {/* Center: Document viewer (desk — deepest Z-layer) */}
      <div className="panel-center flex-1 min-w-0 bg-[var(--bg-desk)]">
        <ErrorBoundary>
          <DocumentViewer
            highlightBboxes={highlightBboxes}
            activeCitation={
              chatOpen &&
              activeCitationIndex !== null &&
              activeCitationIndex < citations.length &&
              citations[activeCitationIndex]?.bbox?.page_number === currentPage
                ? citations[activeCitationIndex]
                : null
            }
          />
        </ErrorBoundary>
      </div>

      {/* Right: Panel bay (toggle pills + multi-column panels) */}
      <div className="panel-right border-l border-[var(--border)] shrink-0">
        <RightPanel />
      </div>
    </div>
  );
}
