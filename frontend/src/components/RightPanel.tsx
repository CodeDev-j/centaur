"use client";

import { useDocStore } from "@/stores/useDocStore";
import { useInspectStore, RightPanelTab } from "@/stores/useInspectStore";
import ChatPanel from "./ChatPanel";
import ChunkInspectorPanel from "./ChunkInspectorPanel";
import MetricExplorerPanel from "./MetricExplorerPanel";

const TABS: { key: RightPanelTab; label: string; requiresDoc: boolean }[] = [
  { key: "chat", label: "CHAT", requiresDoc: false },
  { key: "inspect", label: "INSPECT", requiresDoc: true },
  { key: "explore", label: "EXPLORER", requiresDoc: true },
];

export default function RightPanel() {
  const hasDocument = useDocStore((s) => !!s.selectedDocHash);
  const rightPanelTab = useInspectStore((s) => s.rightPanelTab);
  const setRightPanelTab = useInspectStore((s) => s.setRightPanelTab);

  const activeIndex = TABS.findIndex((t) => t.key === rightPanelTab);

  return (
    <div className="flex flex-col h-full">
      {/* Segmented control */}
      <div className="px-4 pt-4 pb-3 shrink-0">
        <div className="relative flex rounded-lg border border-[var(--border)] bg-[var(--bg-desk)] p-1">
          {/* Sliding indicator */}
          <div
            className="absolute top-1 bottom-1 rounded-md bg-[var(--bg-surface)] border border-[var(--border-subtle)] transition-transform duration-300"
            style={{
              width: `calc((100% - 8px) / ${TABS.length})`,
              transform: `translateX(calc(${activeIndex} * 100%))`,
              transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)",
            }}
          />

          {/* Tab buttons */}
          {TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setRightPanelTab(tab.key)}
              disabled={tab.requiresDoc && !hasDocument}
              className={`
                relative z-10 flex-1 py-1.5 text-h2 text-center
                transition-colors duration-150
                ${rightPanelTab === tab.key
                  ? "text-[var(--text-primary)]"
                  : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }
                disabled:opacity-30 disabled:cursor-not-allowed
              `}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Panels — all mounted, only one visible (preserves scroll/state) */}
      <div className="flex-1 min-h-0 relative">
        <div className="absolute inset-0" style={{ display: rightPanelTab === "chat" ? "block" : "none" }}>
          <ChatPanel />
        </div>
        <div className="absolute inset-0" style={{ display: rightPanelTab === "inspect" ? "block" : "none" }}>
          <ChunkInspectorPanel />
        </div>
        <div className="absolute inset-0" style={{ display: rightPanelTab === "explore" ? "block" : "none" }}>
          <MetricExplorerPanel />
        </div>
      </div>
    </div>
  );
}
