"use client";

import { useMemo, useCallback, useEffect, useRef, useState } from "react";
import { Download, Search, ChevronDown, ChevronRight, List, Grid3x3, Copy, ArrowDownNarrowWide, FileText } from "lucide-react";
import { useInspectStore } from "@/stores/useInspectStore";
import { useViewerStore } from "@/stores/useViewerStore";
import { useDocStore } from "@/stores/useDocStore";
import { downloadTsv } from "@/lib/tsvExport";
import {
  dedup,
  groupByPage,
  formatValue,
  formatDelta,
  collectPeriods,
  rawNumericString,
  rowToTsvLine,
  LedgerRow,
  LedgerPageGroup,
  ParsedDataPoint,
} from "@/lib/seriesParser";
import MiniSparkline from "./MiniSparkline";

// ─── Clipboard helper ───────────────────────────────────────────────

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

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

// ─── Ledger Row (List View) ─────────────────────────────────────────

function LedgerRowComponent({
  row,
  isActive,
  isExpanded,
  onSelect,
  onCopyRow,
  rowRef,
}: {
  row: LedgerRow;
  isActive: boolean;
  isExpanded: boolean;
  onSelect: () => void;
  onCopyRow: () => void;
  rowRef: (el: HTMLDivElement | null) => void;
}) {
  const sparkValues = row.dataPoints.map((dp) => dp.value);
  const [flashValue, setFlashValue] = useState(false);

  const handleCopyLatest = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      const last = row.dataPoints[row.dataPoints.length - 1];
      if (!last) return;
      copyText(rawNumericString(last.value));
      setFlashValue(true);
      setTimeout(() => setFlashValue(false), 150);
    },
    [row.dataPoints],
  );

  return (
    <div ref={rowRef} className="group">
      <button
        onClick={onSelect}
        className={`w-full flex items-center gap-1 px-3 py-1.5 text-left transition-colors border-l-2 ${
          isActive
            ? "ledger-row-active border-[var(--accent)]"
            : "border-transparent hover:bg-[var(--bg-tertiary)]"
        }`}
        style={{ minWidth: 340 }}
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

        {/* Latest value (clickable to copy) */}
        <span
          onClick={handleCopyLatest}
          className={`text-[12px] font-mono tabular-nums text-[var(--text-primary)] w-[76px] text-right shrink-0 cursor-copy rounded px-0.5 ${
            flashValue ? "copy-flash" : ""
          }`}
          title="Click to copy raw value"
        >
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

        {/* Per-row copy button (visible on hover) */}
        <span
          onClick={(e) => {
            e.stopPropagation();
            onCopyRow();
          }}
          className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)] shrink-0"
          title="Copy row as TSV"
        >
          <Copy size={11} />
        </span>
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

// ─── Table View ─────────────────────────────────────────────────────

function TableView({
  groups,
  onNavigatePage,
}: {
  groups: LedgerPageGroup[];
  onNavigatePage: (page: number) => void;
}) {
  const [flashCell, setFlashCell] = useState<string | null>(null);

  const allRows = useMemo(() => groups.flatMap((g) => g.rows), [groups]);
  const periods = useMemo(() => collectPeriods(allRows), [allRows]);

  // Build lookup: chunkId → period → dataPoint
  const matrix = useMemo(() => {
    const m = new Map<string, Map<string, ParsedDataPoint>>();
    for (const row of allRows) {
      const rowMap = new Map<string, ParsedDataPoint>();
      for (const dp of row.dataPoints) {
        rowMap.set(dp.period, dp);
      }
      m.set(row.chunkId, rowMap);
    }
    return m;
  }, [allRows]);

  const handleCopyCell = useCallback((value: number, cellKey: string) => {
    copyText(rawNumericString(value));
    setFlashCell(cellKey);
    setTimeout(() => setFlashCell(null), 150);
  }, []);

  const handleCopyTable = useCallback(() => {
    const header = ["Metric", "Pg", ...periods, "Δ"].join("\t");
    const rows = allRows.map((row) => {
      const vals = periods.map((p) => {
        const dp = matrix.get(row.chunkId)?.get(p);
        return dp ? rawNumericString(dp.value) : "";
      });
      const delta = row.delta ? formatDelta(row.delta) : "";
      return [row.label, row.pageNumber, ...vals, delta].join("\t");
    });
    copyText([header, ...rows].join("\n"));
  }, [allRows, periods, matrix]);

  // Track page group boundaries for visual separators
  const rowPageMap = useMemo(() => {
    const map = new Map<string, number>();
    let groupIdx = 0;
    for (const group of groups) {
      for (const row of group.rows) {
        map.set(row.chunkId, groupIdx);
      }
      groupIdx++;
    }
    return map;
  }, [groups]);

  return (
    <div className="flex-1 overflow-auto explorer-scroll relative">
      <table className="explorer-table">
        <thead>
          <tr>
            <th className="col-metric text-left">Metric</th>
            <th className="text-right" style={{ width: 28 }}>Pg</th>
            {periods.map((p) => (
              <th key={p} className="text-right">{p}</th>
            ))}
            <th className="text-center" style={{ width: 52 }}>Trend</th>
            <th className="text-right" style={{ width: 64 }}>Δ</th>
            <th style={{ width: 28 }}>
              <button
                onClick={handleCopyTable}
                className="p-0.5 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]"
                title="Copy table to clipboard"
              >
                <Copy size={11} />
              </button>
            </th>
          </tr>
        </thead>
        <tbody>
          {allRows.map((row, ri) => {
            const rowData = matrix.get(row.chunkId);
            const prevGroupIdx = ri > 0 ? rowPageMap.get(allRows[ri - 1].chunkId) : undefined;
            const curGroupIdx = rowPageMap.get(row.chunkId);
            const isGroupBreak = ri > 0 && curGroupIdx !== prevGroupIdx;

            return (
              <tr
                key={row.chunkId}
                className={isGroupBreak ? "group-break" : ""}
              >
                <td
                  className="col-metric cursor-pointer hover:text-[var(--accent)]"
                  onClick={() => onNavigatePage(row.pageNumber)}
                  title={`${row.label}${row.basis ? ` [${row.basis}]` : ""} — click to view page`}
                >
                  {row.label}
                  {row.basis && (
                    <span className="text-[10px] text-[var(--text-secondary)] ml-1">
                      [{row.basis}]
                    </span>
                  )}
                </td>
                <td className="text-right text-[10px] text-[var(--text-secondary)]">
                  {row.pageNumber}
                </td>
                {periods.map((p) => {
                  const dp = rowData?.get(p);
                  const cellKey = `${row.chunkId}:${p}`;
                  return (
                    <td
                      key={p}
                      className={`cell-value ${flashCell === cellKey ? "copy-flash" : ""}`}
                      onClick={() => dp && handleCopyCell(dp.value, cellKey)}
                    >
                      {dp ? formatValue(dp.value, dp.magnitude, dp.currency) : ""}
                    </td>
                  );
                })}
                <td className="text-center">
                  <MiniSparkline
                    values={row.dataPoints.map((d) => d.value)}
                    width={48}
                    height={18}
                  />
                </td>
                <td
                  className={`text-right text-[11px] px-2 ${
                    row.delta && row.delta.deltaPct > 0
                      ? "delta-positive"
                      : row.delta && row.delta.deltaPct < 0
                        ? "delta-negative"
                        : "text-[var(--text-secondary)]"
                  }`}
                >
                  {row.delta ? formatDelta(row.delta) : ""}
                </td>
                <td />
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ─── Math Footer ────────────────────────────────────────────────────

function MathFooter({ rows }: { rows: LedgerRow[] }) {
  const stats = useMemo(() => {
    const withDelta = rows.filter((r) => r.delta !== null);
    const deltas = withDelta.map((r) => r.delta!.deltaPct);
    const positive = deltas.filter((d) => d > 0).length;
    const negative = deltas.filter((d) => d < 0).length;

    // Median
    const sorted = [...deltas].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median =
      sorted.length === 0
        ? null
        : sorted.length % 2 === 0
          ? (sorted[mid - 1] + sorted[mid]) / 2
          : sorted[mid];

    return { total: rows.length, positive, negative, median };
  }, [rows]);

  if (stats.total === 0) return null;

  return (
    <div className="math-footer">
      <span>Σ {stats.total}</span>
      {stats.positive > 0 && <span className="text-[#10b981]">↑ {stats.positive}</span>}
      {stats.negative > 0 && <span className="text-[#f43f5e]">↓ {stats.negative}</span>}
      {stats.median !== null && (
        <span>
          Median Δ: {stats.median > 0 ? "+" : ""}
          {stats.median.toFixed(1)}%
        </span>
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
  const viewMode = useInspectStore((s) => s.explorerViewMode);
  const setViewMode = useInspectStore((s) => s.setExplorerViewMode);
  const sortMode = useInspectStore((s) => s.explorerSortMode);
  const setSortMode = useInspectStore((s) => s.setExplorerSortMode);
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
    return groupByPage(dedupedChunks, summaryChunks ?? [], sortMode);
  }, [dedupedChunks, summaryChunks, sortMode]);

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

  // Flat row list for keyboard navigation + math footer
  // P7 fix: precompute per-row flat indices here instead of mutable counter in render
  const { flatRows, flatIndexMap } = useMemo(() => {
    const rows: LedgerRow[] = [];
    const indexMap = new Map<string, number>();
    let idx = 0;
    for (const group of filteredGroups) {
      if (viewMode === "table" || !collapsedPages.has(group.pageNumber)) {
        for (const row of group.rows) {
          indexMap.set(row.chunkId, idx++);
          rows.push(row);
        }
      }
    }
    return { flatRows: rows, flatIndexMap: indexMap };
  }, [filteredGroups, collapsedPages, viewMode]);

  // Total deduped count for header
  const totalDeduped = useMemo(() => {
    return pageGroups.reduce((sum, g) => sum + g.rows.length, 0);
  }, [pageGroups]);

  // Filter options
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

  // Sync flat row count to store
  useEffect(() => {
    useInspectStore.getState().setExplorerFlatRowCount(flatRows.length);
  }, [flatRows.length]);

  // Scroll active row into view (list view only)
  useEffect(() => {
    if (activeRowIdx === null || viewMode !== "list") return;
    const el = rowRefs.current.get(activeRowIdx);
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeRowIdx, viewMode]);

  // ── Handlers ─────────────────────────────────────────────────────
  const handleRowSelect = useCallback(
    (row: LedgerRow, flatIdx: number) => {
      useViewerStore.getState().setCurrentPage(row.pageNumber);
      setExpandedChunkId((prev) => (prev === row.chunkId ? null : row.chunkId));
      useInspectStore.getState().setExplorerActiveRowIdx(flatIdx);
    },
    [],
  );

  const handleCopyRow = useCallback((row: LedgerRow) => {
    copyText(rowToTsvLine(row));
  }, []);

  const handleNavigatePage = useCallback((page: number) => {
    useViewerStore.getState().setCurrentPage(page);
  }, []);

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

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-3 py-2 border-b border-[var(--border)] space-y-1.5 shrink-0">
        <div className="flex items-center justify-between">
          <h3 className="text-[12px] font-semibold">
            Metric Explorer
            {totalDeduped > 0 && (
              <span className="font-normal text-[var(--text-secondary)]">
                {" "}&mdash; {totalDeduped} series
              </span>
            )}
          </h3>
          <div className="flex items-center gap-1">
            {/* Sort mode toggle */}
            <div className="flex border border-[var(--border)] rounded overflow-hidden">
              <button
                onClick={() => setSortMode("document")}
                className={`p-1 transition-colors ${
                  sortMode === "document"
                    ? "bg-[var(--bg-surface)] text-[var(--text-primary)]"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }`}
                title="Document order (alphabetical within page groups)"
              >
                <FileText size={16} />
              </button>
              <button
                onClick={() => setSortMode("statement")}
                className={`p-1 transition-colors ${
                  sortMode === "statement"
                    ? "bg-[var(--bg-surface)] text-[var(--text-primary)]"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }`}
                title="Statement order (income statement waterfall)"
              >
                <ArrowDownNarrowWide size={16} />
              </button>
            </div>
            {/* View mode toggle */}
            <div className="flex border border-[var(--border)] rounded overflow-hidden">
              <button
                onClick={() => setViewMode("list")}
                className={`p-1 transition-colors ${
                  viewMode === "list"
                    ? "bg-[var(--bg-surface)] text-[var(--text-primary)]"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }`}
                title="List view"
              >
                <List size={16} />
              </button>
              <button
                onClick={() => setViewMode("table")}
                className={`p-1 transition-colors ${
                  viewMode === "table"
                    ? "bg-[var(--bg-surface)] text-[var(--text-primary)]"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }`}
                title="Table view"
              >
                <Grid3x3 size={16} />
              </button>
            </div>
            <button
              onClick={handleDownload}
              disabled={!chunks || chunks.length === 0}
              className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-30"
              title="Download all metrics as TSV"
            >
              <Download size={16} />
            </button>
          </div>
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

        {/* Filters */}
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

      {/* Content area */}
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

      {/* Table view */}
      {!isLoading && viewMode === "table" && filteredGroups.length > 0 && (
        <TableView groups={filteredGroups} onNavigatePage={handleNavigatePage} />
      )}

      {/* List view */}
      {!isLoading && viewMode === "list" && filteredGroups.length > 0 && (
        <div className="flex-1 overflow-auto explorer-scroll" style={{ overflowX: "auto" }}>
          {filteredGroups.map((group) => {
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
                    const currentFlatIdx = flatIndexMap.get(row.chunkId) ?? -1;
                    return (
                      <LedgerRowComponent
                        key={row.chunkId}
                        row={row}
                        isActive={activeRowIdx === currentFlatIdx}
                        isExpanded={expandedChunkId === row.chunkId}
                        onSelect={() => handleRowSelect(row, currentFlatIdx)}
                        onCopyRow={() => handleCopyRow(row)}
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
      )}

      {/* Sticky math footer */}
      {!isLoading && flatRows.length > 0 && <MathFooter rows={flatRows} />}
    </div>
  );
}
