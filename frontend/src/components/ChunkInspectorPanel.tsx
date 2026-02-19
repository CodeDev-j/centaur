"use client";

import { useInspectStore } from "@/stores/useInspectStore";
import { useViewerStore } from "@/stores/useViewerStore";
import { ChunkDetail } from "@/lib/api";

// Type badge colors — high contrast on dark (#141414) background
const TYPE_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  financial_table: { bg: "#22c55e20", text: "#22c55e", label: "Table" },
  chart_table:     { bg: "#14b8a620", text: "#14b8a6", label: "Chart Table" },
  visual:          { bg: "#3b82f620", text: "#3b82f6", label: "Visual" },
  narrative:       { bg: "#f59e0b20", text: "#f59e0b", label: "Narrative" },
  header:          { bg: "#a855f720", text: "#a855f7", label: "Header" },
};

function getTypeStyle(itemType: string, chunkRole?: string | null) {
  const base = TYPE_COLORS[itemType] || { bg: "#66666620", text: "#666", label: itemType };
  if (itemType === "visual" && chunkRole === "series") {
    return { bg: "#06b6d420", text: "#06b6d4", label: "Series" };
  }
  if (itemType === "visual" && chunkRole === "summary") {
    return { ...base, label: "Summary" };
  }
  return base;
}

function MetadataSummary({ chunk }: { chunk: ChunkDetail }) {
  const m = chunk.metadata;
  const parts: string[] = [];

  switch (chunk.item_type) {
    case "financial_table":
      if (m.accounting_basis) parts.push(String(m.accounting_basis));
      if (m.periodicity) parts.push(String(m.periodicity));
      if (m.currency) parts.push(String(m.currency));
      break;
    case "visual":
      if (chunk.chunk_role === "series") {
        if (m.series_label) parts.push(String(m.series_label));
        if (m.series_nature) parts.push(String(m.series_nature));
        if (m.periodicity) parts.push(String(m.periodicity));
      } else {
        if (m.archetype) parts.push(String(m.archetype));
      }
      break;
    case "chart_table":
      if (m.archetype) parts.push(String(m.archetype));
      break;
    case "narrative":
      if (m.category) parts.push(String(m.category));
      if (m.sentiment) parts.push(String(m.sentiment));
      break;
    case "header":
      if (m.level) parts.push(`Level ${m.level}`);
      break;
  }

  if (parts.length === 0) return null;
  return (
    <span className="text-xs text-[var(--text-secondary)] truncate">
      {parts.join(" · ")}
    </span>
  );
}

function ChunkCard({
  chunk,
  isActive,
  onSelect,
}: {
  chunk: ChunkDetail;
  isActive: boolean;
  onSelect: () => void;
}) {
  const style = getTypeStyle(chunk.item_type, chunk.chunk_role);

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left p-3 rounded-lg border transition-colors ${
        isActive
          ? "border-[var(--accent)] bg-[var(--highlight)]"
          : "border-[var(--border)] bg-[var(--bg-tertiary)] hover:border-[var(--text-secondary)]"
      }`}
    >
      <div className="flex items-center gap-2 mb-1">
        <span
          className="text-xs font-medium px-2 py-0.5 rounded-full"
          style={{ backgroundColor: style.bg, color: style.text }}
        >
          {style.label}
        </span>
        {chunk.bbox && (
          <span className="text-[10px] text-[var(--text-secondary)]">has bbox</span>
        )}
        {chunk.value_bboxes_count > 0 && (
          <span className="text-[10px] text-[var(--text-secondary)]">
            {chunk.value_bboxes_count} values
          </span>
        )}
      </div>

      {!isActive && (
        <>
          <p className="text-xs text-[var(--text-primary)] line-clamp-2 mb-1">
            {chunk.chunk_text}
          </p>
          <MetadataSummary chunk={chunk} />
        </>
      )}

      {isActive && (
        <div className="mt-2 space-y-2">
          <div className="max-h-40 overflow-auto text-xs text-[var(--text-primary)] whitespace-pre-wrap bg-[var(--bg-primary)] rounded p-2">
            {chunk.chunk_text}
          </div>
          <div className="text-[10px] text-[var(--text-secondary)] font-mono space-y-0.5">
            <div>chunk_id: {chunk.chunk_id}</div>
            <div>item_id: {chunk.item_id}</div>
          </div>
          <div className="space-y-1">
            {Object.entries(chunk.metadata).map(([key, value]) => {
              if (value === null || value === undefined || value === "") return null;
              return (
                <div key={key} className="flex gap-2 text-[10px]">
                  <span className="text-[var(--text-secondary)] shrink-0 font-mono">
                    {key}:
                  </span>
                  <span className="text-[var(--text-primary)] truncate">
                    {typeof value === "object" ? JSON.stringify(value) : String(value)}
                  </span>
                </div>
              );
            })}
          </div>
          {chunk.bbox && (
            <div className="text-[10px] text-[var(--text-secondary)] font-mono">
              bbox: ({chunk.bbox.x.toFixed(3)}, {chunk.bbox.y.toFixed(3)}) {chunk.bbox.width.toFixed(3)}x{chunk.bbox.height.toFixed(3)}
            </div>
          )}
        </div>
      )}
    </button>
  );
}

export default function ChunkInspectorPanel() {
  const chunks = useInspectStore((s) => s.inspectChunks);
  const docStats = useInspectStore((s) => s.docStats);
  const activeChunkId = useInspectStore((s) => s.activeChunkId);
  const setActiveChunkId = useInspectStore((s) => s.setActiveChunkId);
  const isLoading = useInspectStore((s) => s.inspectLoading);
  const currentPage = useViewerStore((s) => s.currentPage);

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-[var(--border)] space-y-1">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">
            Page {currentPage}
            {chunks && (
              <span className="font-normal text-[var(--text-secondary)]">
                {" "}&mdash; {chunks.length} chunk{chunks.length !== 1 ? "s" : ""}
              </span>
            )}
          </h3>
        </div>

        {docStats && (
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(docStats.by_type).map(([type, count]) => {
              const style = TYPE_COLORS[type] || { bg: "#66666620", text: "#666", label: type };
              return (
                <span
                  key={type}
                  className="text-[10px] px-1.5 py-0.5 rounded"
                  style={{ backgroundColor: style.bg, color: style.text }}
                >
                  {count} {style.label}
                </span>
              );
            })}
          </div>
        )}
      </div>

      <div className="flex-1 overflow-auto p-3 space-y-2">
        {isLoading && (
          <div className="text-sm text-[var(--text-secondary)] text-center mt-8 animate-pulse">
            Loading chunks...
          </div>
        )}

        {!isLoading && chunks && chunks.length === 0 && (
          <div className="text-sm text-[var(--text-secondary)] text-center mt-8">
            No chunks indexed for this page
          </div>
        )}

        {!isLoading && !chunks && (
          <div className="text-sm text-[var(--text-secondary)] text-center mt-8">
            Select a document to inspect chunks
          </div>
        )}

        {!isLoading &&
          chunks?.map((chunk) => (
            <ChunkCard
              key={chunk.chunk_id}
              chunk={chunk}
              isActive={activeChunkId === chunk.chunk_id}
              onSelect={() =>
                setActiveChunkId(
                  activeChunkId === chunk.chunk_id ? null : chunk.chunk_id
                )
              }
            />
          ))}
      </div>
    </div>
  );
}
