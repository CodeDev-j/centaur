"use client";

import { useDocStore } from "@/stores/useDocStore";
import { useInspectStore } from "@/stores/useInspectStore";
import ChatPanel from "./ChatPanel";
import ChunkInspectorPanel from "./ChunkInspectorPanel";

export default function RightPanel() {
  const hasDocument = useDocStore((s) => !!s.selectedDocHash);
  const inspectMode = useInspectStore((s) => s.inspectMode);
  const setInspectMode = useInspectStore((s) => s.setInspectMode);

  return (
    <div className="flex flex-col h-full">
      {/* Tab bar */}
      <div className="flex border-b border-[var(--border)] shrink-0">
        <button
          onClick={() => setInspectMode(false)}
          className={`flex-1 px-4 py-2.5 text-sm font-medium transition-colors ${
            !inspectMode
              ? "text-[var(--text-primary)] border-b-2 border-[var(--accent)]"
              : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
          }`}
        >
          Chat
        </button>
        <button
          onClick={() => setInspectMode(true)}
          disabled={!hasDocument}
          className={`flex-1 px-4 py-2.5 text-sm font-medium transition-colors ${
            inspectMode
              ? "text-[var(--text-primary)] border-b-2 border-[var(--accent)]"
              : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
          } disabled:opacity-30 disabled:cursor-not-allowed`}
        >
          Inspect
        </button>
      </div>

      {/* Panels — both mounted, only one visible (preserves Chat scroll/state) */}
      <div className="flex-1 min-h-0 relative">
        <div
          className="absolute inset-0"
          style={{ display: inspectMode ? "none" : "block" }}
        >
          <ChatPanel />
        </div>
        <div
          className="absolute inset-0"
          style={{ display: inspectMode ? "block" : "none" }}
        >
          <ChunkInspectorPanel />
        </div>
      </div>
    </div>
  );
}
