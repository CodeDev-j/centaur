"use client";

import { useMemo, useCallback, useEffect, useRef, useState } from "react";
import { Download, Search, ChevronDown, ChevronRight } from "lucide-react";
import { useInspectStore } from "@/stores/useInspectStore";
import { useViewerStore } from "@/stores/useViewerStore";
import { useDocStore } from "@/stores/useDocStore";
import { downloadTsv } from "@/lib/tsvExport";
import {
  dedup,
  groupByPage,
  formatValue,
  formatDelta,
  LedgerRow,
  LedgerPageGroup,
} from "@/lib/seriesParser";
import MiniSparkline from "./MiniSparkline";

// ─── Filter Dropdown ─────────────────────────────────────────────────

function FilterPill({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value?: string;
  options: string[];
  onChange: (v: string | undefined) => void;
}) {
  // Auto-hide when 0 or 1 options (no filtering value)
  if (options.length <= 1) return null;
  return (
    <select
      value={value || ""}
      onChange={(e) => onChange(e.target.value || undefined)}
      className="text-[11px] bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-1.5 py-0.5 text-[var(--text-secondary)]"
      title={label}
    >
      <option value="">{label}: All</option>
      {options.map((opt) => (
        <option key={opt} value={opt}>
          {opt}
        </option>
      ))}
    </select>
  );
}

// ─── Page Group Section ──────────────────────────────────────────────

function PageGroupHeader({
  group,
  isCollapsed,
  onToggle,
}: {
  group: LedgerPageGroup;
  isCollapsed: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      className="w-full flex items-center gap-1.5 px-3 py-1.5 text-left border-b border-[var(--border)] bg-[var(--bg-primary)] hover:bg-[var(--bg-tertiary)] transition-colors sticky top-0 z-[1]"
    >
      {isCollapsed ? (
        <ChevronRight size={12} className="text-[var(--text-secondary)] shrink-0" />
      ) : (
        <ChevronDown size={12} className="text-[var(--text-secondary)] shrink-0" />
      )}
      <span className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider shrink-0">
        P.{group.pageNumber}
      </span>
      <span className="text-[11px] font-medium text-[var(--text-primary)] truncate min-w-0">
        {group.chartTitle}
      </span>
      <span className="text-[10px] text-[var(--text-secondary)] shrink-0 ml-auto">
        {group.rows.length}
      </span>
    </button>
  );
}

// ─── Ledger Row ──────────────────────────────────────────────────────

function LedgerRowComponent({
  row,
  isActive,
  isExpanded,
  onSelect,
  rowRef,
}: {
  row: LedgerRow;
  isActive: boolean;
  isExpanded: boolean;
  onSelect: () => void;
  rowRef: (el: HTMLDivElement | null) => void;
}) {
  const sparkValues = row.dataPoints.map((dp) => dp.value);

  return (
    <div ref={rowRef}>
      <button
        onClick={onSelect}
        className={`w-full flex items-center gap-1 px-3 py-1.5 text-left transition-colors border-l-2 ${
          isActive
            ? "ledger-row-active border-[var(--accent)]"
            : "border-transparent hover:bg-[var(--bg-tertiary)]"
        }`}
      >
        {/* Label */}
        <span className="text-[12px] text-[var(--text-primary)] truncate min-w-0 flex-1">
          {row.label}
          {row.basis && (
            <span className="ml-1 text-[10px] text-[var(--text-secondary)] opacity-70">
              [{row.basis}]
            </span>
          )}
        </span>

        {/* Sparkline */}
        <MiniSparkline values={sparkValues} width={48} height={18} />

        {/* Latest value */}
        <span className="text-[12px] font-mono tabular-nums text-[var(--text-primary)] w-[76px] text-right shrink-0">
          {row.formattedLatest}
        </span>

        {/* Delta pill */}
        {row.delta ? (
          <span
            className={`text-[11px] font-mono tabular-nums w-[62px] text-right shrink-0 px-1 py-0.5 rounded ${
              row.delta.deltaPct > 0
                ? "delta-positive"
                : row.delta.deltaPct < 0
                  ? "delta-negative"
                  : "text-[var(--text-secondary)]"
            }`}
          >
            {formatDelta(row.delta)}
          </span>
        ) : (
          <span className="w-[62px] shrink-0" />
        )}
      </button>

      {/* Expanded: Ticker Tape */}
      {isExpanded && (
        <div className="px-3 pb-2 ml-2 border-l-2 border-[var(--accent)]">
          <div className="bg-[var(--bg-primary)] rounded p-2 space-y-0.5">
            {row.dataPoints.map((dp, i) => (
              <div key={i} className="flex justify-between text-[11px] font-mono tabular-nums">
                <span className="text-[var(--text-secondary)]">{dp.period}</span>
                <span className="text-[var(--text-primary)]">
                  {formatValue(dp.value, dp.magnitude, dp.currency)}
                </span>
              </div>
            ))}
            {/* Metadata footer */}
            <div className="flex flex-wrap gap-x-3 gap-y-0.5 pt-1 mt-1 border-t border-[var(--border)]">
              {row.basis && (
                <span className="text-[10px] text-[var(--text-secondary)]">
                  Basis: {row.basis}
                </span>
              )}
              {row.nature && (
                <span className="text-[10px] text-[var(--text-secondary)]">
                  Nature: {row.nature}
                </span>
              )}
              {row.periodicity && (
                <span className="text-[10px] text-[var(--text-secondary)]">
                  {row.periodicity}
                </span>
              )}
              <span className="text-[10px] text-[var(--text-secondary)]">
                Pg {row.pageNumber}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────

export default function MetricExplorerPanel() {
  const chunks = useInspectStore((s) => s.explorerChunks);
  const summaryChunks = useInspectStore((s) => s.explorerSummaryChunks);
  const isLoading = useInspectStore((s) => s.explorerLoading);
  const filters = useInspectStore((s) => s.explorerFilters);
  const setFilter = useInspectStore((s) => s.setExplorerFilter);
  const searchQuery = useInspectStore((s) => s.explorerSearchQuery);
  const setSearchQuery = useInspectStore((s) => s.setExplorerSearchQuery);
  const activeRowIdx = useInspectStore((s) => s.explorerActiveRowIdx);
  const selectedDocHash = useDocStore((s) => s.selectedDocHash);
  const selectedFilename = useDocStore((s) => s.selectedFilename);

  const [expandedChunkId, setExpandedChunkId] = useState<string | null>(null);
  const [collapsedPages, setCollapsedPages] = useState<Set<number>>(new Set());
  const rowRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  // ── Data pipeline ────────────────────────────────────────────────
  const dedupedChunks = useMemo(() => {
    if (!chunks) return [];
    return dedup(chunks);
  }, [chunks]);

  const pageGroups = useMemo(() => {
    return groupByPage(dedupedChunks, summaryChunks ?? []);
  }, [dedupedChunks, summaryChunks]);

  // Apply search + filters
  const filteredGroups = useMemo(() => {
    const query = searchQuery.toLowerCase();
    return pageGroups
      .map((group) => ({
        ...group,
        rows: group.rows.filter((row) => {
          if (query && !row.label.toLowerCase().includes(query)) return false;
          if (filters.periodicity && row.periodicity !== filters.periodicity) return false;
          if (filters.accounting_basis && row.basis !== filters.accounting_basis) return false;
          if (filters.series_nature && row.nature !== filters.series_nature) return false;
          return true;
        }),
      }))
      .filter((group) => group.rows.length > 0);
  }, [pageGroups, searchQuery, filters]);

  // Flat row list for keyboard navigation
  const flatRows = useMemo(() => {
    const rows: LedgerRow[] = [];
    for (const group of filteredGroups) {
      if (!collapsedPages.has(group.pageNumber)) {
        rows.push(...group.rows);
      }
    }
    return rows;
  }, [filteredGroups, collapsedPages]);

  // Total deduped count for header
  const totalDeduped = useMemo(() => {
    return pageGroups.reduce((sum, g) => sum + g.rows.length, 0);
  }, [pageGroups]);

  // Filter options (extracted from all data, auto-hide when uniform)
  const filterOptions = useMemo(() => {
    const p = new Set<string>();
    const a = new Set<string>();
    const n = new Set<string>();
    for (const group of pageGroups) {
      for (const row of group.rows) {
        if (row.periodicity) p.add(row.periodicity);
        if (row.basis) a.add(row.basis);
        if (row.nature) n.add(row.nature);
      }
    }
    return {
      periodicity: [...p].sort(),
      accounting_basis: [...a].sort(),
      series_nature: [...n].sort(),
    };
  }, [pageGroups]);

  // Sync flat row count to store for keyboard navigation bounds
  useEffect(() => {
    useInspectStore.getState().setExplorerFlatRowCount(flatRows.length);
  }, [flatRows.length]);

  // Scroll active row into view
  useEffect(() => {
    if (activeRowIdx === null) return;
    const el = rowRefs.current.get(activeRowIdx);
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeRowIdx]);

  // ── Handlers ─────────────────────────────────────────────────────
  const handleRowSelect = useCallback(
    (row: LedgerRow, flatIdx: number) => {
      // Navigate PDF to page
      useViewerStore.getState().setCurrentPage(row.pageNumber);
      // Toggle expansion
      setExpandedChunkId((prev) => (prev === row.chunkId ? null : row.chunkId));
      // Set active for keyboard
      useInspectStore.getState().setExplorerActiveRowIdx(flatIdx);
    },
    [],
  );

  const handleDownload = useCallback(() => {
    if (!chunks || chunks.length === 0) return;
    const name = (selectedFilename || "export").replace(/\.pdf$/i, "");
    downloadTsv(chunks, `${name}_metrics.tsv`);
  }, [chunks, selectedFilename]);

  const togglePageCollapse = useCallback((pageNum: number) => {
    setCollapsedPages((prev) => {
      const next = new Set(prev);
      if (next.has(pageNum)) next.delete(pageNum);
      else next.add(pageNum);
      return next;
    });
  }, []);

  // ── Render ───────────────────────────────────────────────────────
  if (!selectedDocHash) {
    return (
      <div className="flex flex-col h-full items-center justify-center">
        <div className="text-sm text-[var(--text-secondary)]">
          Select a document to explore metrics
        </div>
      </div>
    );
  }

  // Track flat index across groups for keyboard nav
  let flatIdx = 0;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-3 py-2 border-b border-[var(--border)] space-y-1.5">
        <div className="flex items-center justify-between">
          <h3 className="text-[12px] font-semibold">
            Metric Explorer
            {totalDeduped > 0 && (
              <span className="font-normal text-[var(--text-secondary)]">
                {" "}&mdash; {totalDeduped} series
              </span>
            )}
          </h3>
          <button
            onClick={handleDownload}
            disabled={!chunks || chunks.length === 0}
            className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-30"
            title="Download all metrics as TSV"
          >
            <Download size={13} />
          </button>
        </div>

        {/* Search */}
        <div className="relative">
          <Search
            size={12}
            className="absolute left-2 top-1/2 -translate-y-1/2 text-[var(--text-secondary)]"
          />
          <input
            type="text"
            placeholder="Search metrics..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded pl-7 pr-2 py-1 text-[11px] outline-none focus:border-[var(--accent)] text-[var(--text-primary)] placeholder:text-[var(--text-secondary)]"
          />
        </div>

        {/* Filters (auto-hide when not useful) */}
        {(filterOptions.periodicity.length > 1 ||
          filterOptions.accounting_basis.length > 1 ||
          filterOptions.series_nature.length > 1) && (
          <div className="flex flex-wrap gap-1">
            <FilterPill
              label="Period"
              value={filters.periodicity}
              options={filterOptions.periodicity}
              onChange={(v) => setFilter("periodicity", v)}
            />
            <FilterPill
              label="Basis"
              value={filters.accounting_basis}
              options={filterOptions.accounting_basis}
              onChange={(v) => setFilter("accounting_basis", v)}
            />
            <FilterPill
              label="Nature"
              value={filters.series_nature}
              options={filterOptions.series_nature}
              onChange={(v) => setFilter("series_nature", v)}
            />
          </div>
        )}
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-auto explorer-scroll">
        {isLoading && (
          <div className="text-[12px] text-[var(--text-secondary)] text-center mt-8 animate-pulse">
            Loading metrics...
          </div>
        )}

        {!isLoading && chunks !== null && filteredGroups.length === 0 && (
          <div className="text-[12px] text-[var(--text-secondary)] text-center mt-8">
            {searchQuery ? "No metrics match your search" : "No metric series found"}
          </div>
        )}

        {!isLoading && chunks === null && (
          <div className="text-[12px] text-[var(--text-secondary)] text-center mt-8">
            Switch to this tab to load metrics
          </div>
        )}

        {!isLoading &&
          filteredGroups.map((group) => {
            const isCollapsed = collapsedPages.has(group.pageNumber);
            return (
              <div key={group.pageNumber}>
                <PageGroupHeader
                  group={group}
                  isCollapsed={isCollapsed}
                  onToggle={() => togglePageCollapse(group.pageNumber)}
                />
                {!isCollapsed &&
                  group.rows.map((row) => {
                    const currentFlatIdx = flatIdx++;
                    return (
                      <LedgerRowComponent
                        key={row.chunkId}
                        row={row}
                        isActive={activeRowIdx === currentFlatIdx}
                        isExpanded={expandedChunkId === row.chunkId}
                        onSelect={() => handleRowSelect(row, currentFlatIdx)}
                        rowRef={(el) => {
                          if (el) rowRefs.current.set(currentFlatIdx, el);
                          else rowRefs.current.delete(currentFlatIdx);
                        }}
                      />
                    );
                  })}
              </div>
            );
          })}
      </div>
    </div>
  );
}
