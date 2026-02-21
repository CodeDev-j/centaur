"use client";

import { useCallback, useRef, useEffect } from "react";
import { useDocStore } from "@/stores/useDocStore";
import { useInspectStore, PANEL_ORDER, PanelId } from "@/stores/useInspectStore";
import ChatPanel from "./ChatPanel";
import ChunkInspectorPanel from "./ChunkInspectorPanel";
import MetricExplorerPanel from "./MetricExplorerPanel";
import StudioPanel from "./StudioPanel";

// ─── Panel metadata ──────────────────────────────────────────────────

const PANEL_META: Record<PanelId, { label: string; requiresDoc: boolean }> = {
  chat: { label: "CHAT", requiresDoc: false },
  inspect: { label: "INSPECT", requiresDoc: true },
  explore: { label: "EXPLORER", requiresDoc: true },
  studio: { label: "STUDIO", requiresDoc: false },
};

// ─── Toggle Pill ─────────────────────────────────────────────────────

function TogglePill({
  id,
  isOpen,
  disabled,
  onToggle,
}: {
  id: PanelId;
  isOpen: boolean;
  disabled: boolean;
  onToggle: () => void;
}) {
  const meta = PANEL_META[id];

  return (
    <button
      onClick={onToggle}
      disabled={disabled}
      className={`
        toggle-pill relative flex items-center gap-1.5 px-3 py-1.5 rounded-md
        text-h2 select-none transition-all duration-75
        border
        ${isOpen
          ? "toggle-pill-active bg-[var(--bg-panel)] border-[var(--border-sharp)] text-[var(--text-primary)]"
          : "bg-transparent border-[var(--border-sharp)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface)] hover:text-[var(--text-primary)]"
        }
        disabled:opacity-30 disabled:cursor-not-allowed
      `}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full shrink-0 transition-colors duration-75 ${
          isOpen ? "bg-[var(--accent)]" : "border border-[var(--text-secondary)]"
        }`}
      />
      {meta.label}
    </button>
  );
}

// ─── Resize Handle ───────────────────────────────────────────────────

function ResizeHandle({ panelId }: { panelId: PanelId }) {
  // R9 fix: track drag cleanup so we can call it on unmount (mid-drag panel close)
  const cleanupRef = useRef<(() => void) | null>(null);
  useEffect(() => () => { cleanupRef.current?.(); }, []);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const startX = e.clientX;
      const startWidth = useInspectStore.getState().panelWidths[panelId];

      const onMouseMove = (ev: MouseEvent) => {
        const delta = ev.clientX - startX;
        useInspectStore.getState().setPanelWidth(panelId, startWidth - delta);
      };

      const cleanup = () => {
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", cleanup);
        document.body.style.removeProperty("cursor");
        document.body.style.removeProperty("user-select");
        cleanupRef.current = null;
      };

      cleanupRef.current = cleanup;
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", cleanup);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    },
    [panelId],
  );

  return <div className="resize-handle" onMouseDown={onMouseDown} />;
}

// ─── Panel Column ────────────────────────────────────────────────────

function PanelColumn({
  id,
  isOpen,
  isFirst,
  width,
}: {
  id: PanelId;
  isOpen: boolean;
  isFirst: boolean;
  width: number;
}) {
  return (
    <>
      {/* Resize handle (rendered before panel, on its left edge) */}
      {isOpen && <ResizeHandle panelId={id} />}

      <div
        className={`
          h-full overflow-hidden shrink-0
          ${isOpen ? "bg-[var(--bg-panel)]" : ""}
          ${isOpen && isFirst ? "panel-bay-first" : ""}
        `}
        style={{
          width: isOpen ? `${width}px` : "0px",
        }}
      >
        <div className="h-full flex flex-col" style={{ width: `${width}px` }}>
          {id === "chat" && <ChatPanel />}
          {id === "inspect" && <ChunkInspectorPanel />}
          {id === "explore" && <MetricExplorerPanel />}
          {id === "studio" && <StudioPanel />}
        </div>
      </div>
    </>
  );
}

// ─── Main Component ──────────────────────────────────────────────────

export default function RightPanel() {
  const hasDocument = useDocStore((s) => !!s.selectedDocHash);
  const openPanels = useInspectStore((s) => s.openPanels);
  const panelWidths = useInspectStore((s) => s.panelWidths);
  const togglePanel = useInspectStore((s) => s.togglePanel);

  // Track which panels are open in canonical order (for isFirst detection)
  const openInOrder = PANEL_ORDER.filter((id) => openPanels.includes(id));

  return (
    <div className="flex flex-col h-full bg-[var(--bg-desk)]">
      {/* Toggle pill bar */}
      <div className="flex items-center gap-2 px-3 py-2.5 shrink-0 border-b border-[var(--border-subtle)]">
        {PANEL_ORDER.map((id) => (
          <TogglePill
            key={id}
            id={id}
            isOpen={openPanels.includes(id)}
            disabled={PANEL_META[id].requiresDoc && !hasDocument}
            onToggle={() => togglePanel(id)}
          />
        ))}
      </div>

      {/* Panel columns */}
      <div className="flex flex-1 min-h-0 bg-[var(--bg-desk)]">
        {PANEL_ORDER.map((id) => (
          <PanelColumn
            key={id}
            id={id}
            isOpen={openPanels.includes(id)}
            isFirst={openInOrder[0] === id}
            width={panelWidths[id]}
          />
        ))}
      </div>
    </div>
  );
}
