"use client";

import { useEffect } from "react";
import DocumentList from "@/components/DocumentList";
import DocumentViewer from "@/components/DocumentViewer";
import RightPanel from "@/components/RightPanel";
import { useDocStore } from "@/stores/useDocStore";
import { useViewerStore } from "@/stores/useViewerStore";
import { useChatStore } from "@/stores/useChatStore";
import { useInspectStore } from "@/stores/useInspectStore";
import { fetchPageChunks, fetchDocChunks, fetchDocStats } from "@/lib/api";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";

export default function Home() {
  useKeyboardShortcuts();
  const selectedDocHash = useDocStore((s) => s.selectedDocHash);
  const currentPage = useViewerStore((s) => s.currentPage);
  const sidebarCollapsed = useViewerStore((s) => s.sidebarCollapsed);
  const rightPanelTab = useInspectStore((s) => s.rightPanelTab);
  const inspectMode = rightPanelTab === "inspect";

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
  // Fetch chunks when inspect tab is active and page/document changes
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!inspectMode || !selectedDocHash) return;

    let cancelled = false;
    useInspectStore.getState().setInspectLoading(true);
    useInspectStore.getState().setActiveChunkId(null);

    fetchPageChunks(selectedDocHash, currentPage)
      .then((data) => {
        if (!cancelled) useInspectStore.getState().setInspectChunks(data.chunks);
      })
      .catch((err) => {
        if (!cancelled) {
          console.error("Failed to fetch chunks:", err);
          useInspectStore.getState().setInspectChunks(null);
        }
      })
      .finally(() => {
        if (!cancelled) useInspectStore.getState().setInspectLoading(false);
      });

    return () => { cancelled = true; };
  }, [inspectMode, selectedDocHash, currentPage]);

  // ---------------------------------------------------------------------------
  // Fetch explorer data when explore tab is active and document changes
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (rightPanelTab !== "explore" || !selectedDocHash) return;

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
  }, [rightPanelTab, selectedDocHash]);

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
      .catch(() => useInspectStore.getState().setDocStats(null));
  }, [selectedDocHash]);

  // ---------------------------------------------------------------------------
  // Compute chunk highlight bboxes for DocumentViewer
  // Fine-grained value_bboxes when available, coarse bbox as fallback
  // ---------------------------------------------------------------------------
  const activeChunkId = useInspectStore((s) => s.activeChunkId);
  const inspectChunks = useInspectStore((s) => s.inspectChunks);
  const activeChunk =
    inspectMode && activeChunkId
      ? inspectChunks?.find((c) => c.chunk_id === activeChunkId) ?? null
      : null;

  let highlightBboxes: { x: number; y: number; width: number; height: number }[] | null = null;
  if (inspectMode && activeChunk) {
    const vb = activeChunk.value_bboxes;
    if (vb && Object.keys(vb).length > 0) {
      highlightBboxes = Object.values(vb).flatMap((rects) =>
        rects.map(([x, y, width, height]) => ({ x, y, width, height }))
      );
    } else if (activeChunk.bbox) {
      highlightBboxes = [activeChunk.bbox];
    }
  }

  // Citation overlay (only when on chat tab)
  const activeCitationIndex = useChatStore((s) => s.activeCitationIndex);
  const citations = useChatStore((s) => s.citations);

  return (
    <div className="app-shell h-screen flex">
      {/* Left sidebar: Document list */}
      <div
        className={`panel-left border-r border-[var(--border)] bg-[var(--bg-secondary)] shrink-0 transition-all duration-200 ${
          sidebarCollapsed ? "w-0 overflow-hidden" : "w-64"
        }`}
      >
        <DocumentList />
      </div>

      {/* Center: Document viewer (desk — deepest Z-layer) */}
      <div className="panel-center flex-1 min-w-0 bg-[var(--bg-desk)]">
        <DocumentViewer
          highlightBboxes={highlightBboxes}
          activeCitation={
            rightPanelTab === "chat" &&
            activeCitationIndex !== null &&
            activeCitationIndex < citations.length &&
            citations[activeCitationIndex]?.bbox?.page_number === currentPage
              ? citations[activeCitationIndex]
              : null
          }
        />
      </div>

      {/* Right panel: Chat + Inspect + Explorer tabs */}
      <div className="panel-right w-96 border-l border-[var(--border)] bg-[var(--bg-secondary)] shrink-0">
        <RightPanel />
      </div>
    </div>
  );
}
