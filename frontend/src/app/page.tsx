"use client";

import { useEffect } from "react";
import DocumentList from "@/components/DocumentList";
import DocumentViewer from "@/components/DocumentViewer";
import RightPanel from "@/components/RightPanel";
import { useDocStore } from "@/stores/useDocStore";
import { useViewerStore } from "@/stores/useViewerStore";
import { useChatStore } from "@/stores/useChatStore";
import { useInspectStore } from "@/stores/useInspectStore";
import { fetchPageChunks, fetchDocStats } from "@/lib/api";

export default function Home() {
  const selectedDocHash = useDocStore((s) => s.selectedDocHash);
  const currentPage = useViewerStore((s) => s.currentPage);
  const sidebarCollapsed = useViewerStore((s) => s.sidebarCollapsed);
  const inspectMode = useInspectStore((s) => s.inspectMode);

  // ---------------------------------------------------------------------------
  // Fetch chunks when inspect mode is active and page/document changes
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
  // Compute chunk highlight bbox for DocumentViewer
  // ---------------------------------------------------------------------------
  const activeChunkId = useInspectStore((s) => s.activeChunkId);
  const inspectChunks = useInspectStore((s) => s.inspectChunks);
  const activeChunk =
    inspectMode && activeChunkId
      ? inspectChunks?.find((c) => c.chunk_id === activeChunkId) ?? null
      : null;
  const highlightBbox =
    inspectMode && activeChunk?.bbox ? activeChunk.bbox : null;

  // Citation overlay (only when not in inspect mode)
  const activeCitationIndex = useChatStore((s) => s.activeCitationIndex);
  const citations = useChatStore((s) => s.citations);

  return (
    <div className="app-shell h-screen flex">
      {/* Left sidebar: Document list */}
      <div
        className={`border-r border-[var(--border)] bg-[var(--bg-secondary)] shrink-0 transition-all duration-200 ${
          sidebarCollapsed ? "w-0 overflow-hidden" : "w-64"
        }`}
      >
        <DocumentList />
      </div>

      {/* Center: Document viewer */}
      <div className="flex-1 min-w-0 bg-[var(--bg-secondary)]">
        <DocumentViewer
          highlightBbox={highlightBbox}
          activeCitation={
            !inspectMode &&
            activeCitationIndex !== null &&
            activeCitationIndex < citations.length &&
            citations[activeCitationIndex]?.bbox?.page_number === currentPage
              ? citations[activeCitationIndex]
              : null
          }
        />
      </div>

      {/* Right panel: Chat + Inspect tabs */}
      <div className="w-96 border-l border-[var(--border)] bg-[var(--bg-secondary)] shrink-0">
        <RightPanel />
      </div>
    </div>
  );
}
