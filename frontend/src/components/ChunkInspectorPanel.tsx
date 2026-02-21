"use client";

import { useState, useCallback, useMemo } from "react";
import { ClipboardCopy } from "lucide-react";
import { useInspectStore } from "@/stores/useInspectStore";
import { useViewerStore } from "@/stores/useViewerStore";
import { ChunkDetail } from "@/lib/api";
import { copyTsvToClipboard } from "@/lib/tsvExport";

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

// ─── Page Group Header ────────────────────────────────────────────────

function PageGroupHeader({
  pageNumber,
  chunkCount,
  isCurrent,
}: {
  pageNumber: number;
  chunkCount: number;
  isCurrent: boolean;
}) {
  return (
    <div
      className={`sticky top-0 z-10 flex items-center gap-2 px-1 py-1.5 text-xs font-medium ${
        isCurrent
          ? "text-[var(--accent)]"
          : "text-[var(--text-secondary)]"
      }`}
      style={{ background: "var(--bg-panel)" }}
    >
      <div className="h-px flex-1 bg-[var(--border-sharp)]" />
      <span className="shrink-0">
        Page {pageNumber}
        <span className="font-normal opacity-70"> &middot; {chunkCount}</span>
      </span>
      <div className="h-px flex-1 bg-[var(--border-sharp)]" />
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────

export default function ChunkInspectorPanel() {
  const allChunks = useInspectStore((s) => s.inspectAllChunks);
  const docStats = useInspectStore((s) => s.docStats);
  const activeChunkId = useInspectStore((s) => s.activeChunkId);
  const setActiveChunkId = useInspectStore((s) => s.setActiveChunkId);
  const typeFilter = useInspectStore((s) => s.inspectTypeFilter);
  const toggleType = useInspectStore((s) => s.toggleInspectType);
  const isLoading = useInspectStore((s) => s.inspectLoading);
  const currentPage = useViewerStore((s) => s.currentPage);

  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);

  // Filter chunks by active type filter
  const filtered = useMemo(() => {
    if (!allChunks) return [];
    if (typeFilter.length === 0) return allChunks;
    return allChunks.filter((c) => typeFilter.includes(c.item_type));
  }, [allChunks, typeFilter]);

  // Group filtered chunks by page number
  const pageGroups = useMemo(() => {
    const groups = new Map<number, ChunkDetail[]>();
    for (const chunk of filtered) {
      const existing = groups.get(chunk.page_number) ?? [];
      existing.push(chunk);
      groups.set(chunk.page_number, existing);
    }
    return [...groups.entries()]
      .sort(([a], [b]) => a - b)
      .map(([pageNumber, chunks]) => ({ pageNumber, chunks }));
  }, [filtered]);

  const handleSelectChunk = useCallback((chunk: ChunkDetail) => {
    const { activeChunkId: currentId, setActiveChunkId: setId } = useInspectStore.getState();
    if (currentId === chunk.chunk_id) {
      setId(null);
      return;
    }
    setId(chunk.chunk_id);
    // Navigate to the chunk's page if different from current
    const { currentPage: viewerPage } = useViewerStore.getState();
    if (chunk.page_number !== viewerPage) {
      useViewerStore.getState().setCurrentPage(chunk.page_number);
    }
  }, []);

  const handleCopyTsv = useCallback(async () => {
    if (filtered.length === 0) return;
    const ok = await copyTsvToClipboard(filtered);
    setCopyFeedback(ok ? "Copied!" : "Failed");
    setTimeout(() => setCopyFeedback(null), 2000);
  }, [filtered]);

  const isFiltered = typeFilter.length > 0;

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-[var(--border)] space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">
            {isFiltered ? (
              <>
                {filtered.length} chunk{filtered.length !== 1 ? "s" : ""}
                <span className="font-normal text-[var(--text-secondary)]">
                  {" "}&mdash; filtered
                </span>
              </>
            ) : (
              <>
                {allChunks ? allChunks.length : 0} chunks
              </>
            )}
          </h3>
          {filtered.length > 0 && (
            <button
              onClick={handleCopyTsv}
              className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] relative"
              title="Copy visible chunks as TSV"
            >
              <ClipboardCopy size={14} />
              {copyFeedback && (
                <span className="absolute -top-6 right-0 text-[10px] bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-1.5 py-0.5 whitespace-nowrap">
                  {copyFeedback}
                </span>
              )}
            </button>
          )}
        </div>

        {/* Clickable type filter badges */}
        {docStats && (
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(docStats.by_type).map(([type, count]) => {
              const style = TYPE_COLORS[type] || { bg: "#66666620", text: "#666", label: type };
              const isActive = typeFilter.includes(type);
              const isDimmed = isFiltered && !isActive;

              return (
                <button
                  key={type}
                  onClick={() => toggleType(type)}
                  className="text-[10px] px-1.5 py-0.5 rounded transition-all duration-100 cursor-pointer"
                  style={{
                    backgroundColor: isDimmed ? "transparent" : style.bg,
                    color: isDimmed ? "var(--text-muted)" : style.text,
                    border: isDimmed
                      ? "1px solid var(--border-subtle)"
                      : isActive
                        ? `1px solid ${style.text}40`
                        : "1px solid transparent",
                  }}
                >
                  {count} {style.label}
                </button>
              );
            })}
          </div>
        )}
      </div>

      <div className="flex-1 overflow-auto p-3 space-y-1">
        {isLoading && (
          <div className="text-sm text-[var(--text-secondary)] text-center mt-8 animate-pulse">
            Loading chunks...
          </div>
        )}

        {!isLoading && allChunks && filtered.length === 0 && (
          <div className="text-sm text-[var(--text-secondary)] text-center mt-8">
            {isFiltered ? "No chunks match the selected filters" : "No chunks indexed for this document"}
          </div>
        )}

        {!isLoading && !allChunks && (
          <div className="text-sm text-[var(--text-secondary)] text-center mt-8">
            Select a document to inspect chunks
          </div>
        )}

        {!isLoading &&
          pageGroups.map(({ pageNumber, chunks }) => (
            <div key={pageNumber}>
              <PageGroupHeader
                pageNumber={pageNumber}
                chunkCount={chunks.length}
                isCurrent={pageNumber === currentPage}
              />
              <div className="space-y-2 mb-3">
                {chunks.map((chunk) => (
                  <ChunkCard
                    key={chunk.chunk_id}
                    chunk={chunk}
                    isActive={activeChunkId === chunk.chunk_id}
                    onSelect={() => handleSelectChunk(chunk)}
                  />
                ))}
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}
