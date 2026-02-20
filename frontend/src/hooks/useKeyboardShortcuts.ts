"use client";

import { useEffect } from "react";
import { useViewerStore } from "@/stores/useViewerStore";
import { useInspectStore } from "@/stores/useInspectStore";
import { useChatStore } from "@/stores/useChatStore";

/**
 * Global keyboard shortcuts. Call once in the root component.
 *
 * ArrowLeft/Right  — previous / next page
 * Escape           — deselect citation/chunk, blur input
 * / or Ctrl+K      — focus chat input
 * Ctrl+B / Cmd+B   — toggle sidebar
 * 1 / 2 / 3        — switch right panel tab
 * Ctrl+Shift+C     — copy current page chunks as TSV
 * j / k             — move up/down in Explorer rows
 * Enter             — navigate PDF to active Explorer row's page
 */
export function useKeyboardShortcuts() {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const tag = target.tagName.toLowerCase();
      const isEditable = tag === "input" || tag === "textarea" || target.isContentEditable;
      const ctrl = e.ctrlKey || e.metaKey;

      // Allow Escape from inputs; block everything else when typing
      if (isEditable && e.key !== "Escape") return;

      switch (true) {
        // Page navigation
        case e.key === "ArrowLeft" && !ctrl: {
          e.preventDefault();
          const { currentPage, setCurrentPage } = useViewerStore.getState();
          if (currentPage > 1) setCurrentPage(currentPage - 1);
          break;
        }
        case e.key === "ArrowRight" && !ctrl: {
          e.preventDefault();
          const { currentPage, numPages, setCurrentPage } = useViewerStore.getState();
          if (currentPage < numPages) setCurrentPage(currentPage + 1);
          break;
        }

        // Deselect
        case e.key === "Escape": {
          useChatStore.getState().setActiveCitationIndex(null);
          useInspectStore.getState().setActiveChunkId(null);
          if (isEditable) (target as HTMLElement).blur();
          break;
        }

        // Focus chat input
        case (e.key === "/" && !ctrl) || (e.key === "k" && ctrl): {
          e.preventDefault();
          const chatInput = document.querySelector<HTMLInputElement>(
            'input[placeholder*="Ask about"]'
          );
          chatInput?.focus();
          break;
        }

        // Toggle sidebar
        case e.key === "b" && ctrl: {
          e.preventDefault();
          useViewerStore.getState().toggleSidebar();
          break;
        }

        // Right panel tabs
        case e.key === "1" && !ctrl: {
          useInspectStore.getState().setRightPanelTab("chat");
          break;
        }
        case e.key === "2" && !ctrl: {
          useInspectStore.getState().setRightPanelTab("inspect");
          break;
        }
        case e.key === "3" && !ctrl: {
          useInspectStore.getState().setRightPanelTab("explore");
          break;
        }

        // Explorer: move down
        case e.key === "j" && !ctrl: {
          const tab = useInspectStore.getState().rightPanelTab;
          if (tab !== "explore") break;
          e.preventDefault();
          const { explorerActiveRowIdx, explorerFlatRowCount } = useInspectStore.getState();
          const maxIdx = explorerFlatRowCount - 1;
          if (maxIdx < 0) break;
          const next = explorerActiveRowIdx === null ? 0 : Math.min(explorerActiveRowIdx + 1, maxIdx);
          useInspectStore.getState().setExplorerActiveRowIdx(next);
          break;
        }

        // Explorer: move up
        case e.key === "k" && !ctrl: {
          const tab = useInspectStore.getState().rightPanelTab;
          if (tab !== "explore") break;
          e.preventDefault();
          const { explorerActiveRowIdx: idx } = useInspectStore.getState();
          const prev = idx === null ? 0 : Math.max(idx - 1, 0);
          useInspectStore.getState().setExplorerActiveRowIdx(prev);
          break;
        }

        // Explorer: navigate to active row's page
        case e.key === "Enter" && !ctrl: {
          const tab = useInspectStore.getState().rightPanelTab;
          if (tab !== "explore") break;
          // Enter is handled by MetricExplorerPanel's active row logic
          // (the row component auto-navigates on selection via the store)
          break;
        }

        // Copy page chunks as TSV
        case e.key === "C" && ctrl && e.shiftKey: {
          e.preventDefault();
          const chunks = useInspectStore.getState().inspectChunks;
          if (chunks && chunks.length > 0) {
            import("@/lib/tsvExport").then(({ copyTsvToClipboard }) => {
              copyTsvToClipboard(chunks);
            });
          }
          break;
        }
      }
    };

    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, []);
}
