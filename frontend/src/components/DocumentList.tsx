"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  Search,
  Upload,
  RefreshCw,
  Loader2,
  Download,
  X,
  ChevronDown,
  Tag,
  Trash2,
  Square,
  CheckSquare,
} from "lucide-react";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  listDocuments,
  uploadDocument,
  getIngestStatusUrl,
  fetchAuditSummary,
  fetchFacets,
  downloadExcel,
  batchDocuments,
  DocumentSummary,
  AuditSummary,
  Facets,
} from "@/lib/api";
import { useDocStore, DocSortKey } from "@/stores/useDocStore";
import { useViewerStore } from "@/stores/useViewerStore";
import { useInspectStore } from "@/stores/useInspectStore";
import { useChatStore } from "@/stores/useChatStore";

// ── Helpers ──────────────────────────────────────────────────────────

const DOC_TYPE_LABEL: Record<string, string> = {
  earnings_slides: "Earnings Slides",
  annual_report: "Annual Report",
  investor_presentation: "Investor Pres.",
  "10k": "10-K",
  "10q": "10-Q",
  "8k": "8-K",
  proxy_statement: "Proxy Statement",
  cim: "CIM",
  lender_presentation: "Lender Pres.",
  financial_model: "Financial Model",
  term_sheet: "Term Sheet",
  credit_agreement: "Credit Agreement",
  fairness_opinion: "Fairness Opinion",
  teaser: "Teaser",
  management_presentation: "Mgmt Pres.",
  quality_of_earnings: "QoE",
  industry_report: "Industry Report",
  sector_analysis: "Sector Analysis",
  comp_sheet: "Comp Sheet",
  market_update: "Market Update",
  expert_transcript: "Expert Transcript",
  economic_report: "Economic Report",
  central_bank: "Central Bank",
  market_outlook: "Market Outlook",
  policy_brief: "Policy Brief",
  legal_opinion: "Legal Opinion",
  regulatory_filing: "Regulatory Filing",
  internal_memo: "Internal Memo",
  other: "Other",
};

function formatDocType(raw: string | undefined | null): string {
  if (!raw) return "";
  return DOC_TYPE_LABEL[raw] || raw.replace(/_/g, " ");
}

const SORT_OPTIONS: { value: DocSortKey; label: string }[] = [
  { value: "recent", label: "Recent" },
  { value: "name", label: "Name" },
  { value: "company", label: "Company" },
  { value: "doc_date", label: "Date" },
];

// ── Main Component ───────────────────────────────────────────────────

export default function DocumentList() {
  const documents = useDocStore((s) => s.documents);
  const selectedDocHash = useDocStore((s) => s.selectedDocHash);
  const isUploading = useDocStore((s) => s.isUploading);
  const multiSelected = useDocStore((s) => s.multiSelected);
  const filters = useDocStore((s) => s.filters);
  const sortKey = useDocStore((s) => s.sortKey);
  const setDocuments = useDocStore((s) => s.setDocuments);
  const selectDocument = useDocStore((s) => s.selectDocument);
  const setIsUploading = useDocStore((s) => s.setIsUploading);
  const setFacets = useDocStore((s) => s.setFacets);
  const filteredDocuments = useDocStore((s) => s.filteredDocuments);

  const [isLoading, setIsLoading] = useState(false);
  const [auditSummaries, setAuditSummaries] = useState<Record<string, AuditSummary>>({});
  const [showSortMenu, setShowSortMenu] = useState(false);
  const [hoverHash, setHoverHash] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const eventSourcesRef = useRef<Map<string, EventSource>>(new Map());
  const parentRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => filteredDocuments(), [documents, filters, sortKey]);

  // Virtualizer
  const rowVirtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 52,
    overscan: 8,
  });

  // ── Data fetching ──────────────────────────────────────────────────

  const refresh = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      return docs;
    } catch {
      return [];
    }
  }, [setDocuments]);

  const subscribeToStatus = useCallback(
    (docHash: string) => {
      if (eventSourcesRef.current.has(docHash)) return;
      const url = getIngestStatusUrl(docHash);
      const es = new EventSource(url);
      eventSourcesRef.current.set(docHash, es);
      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.status === "completed" || data.status === "failed") {
            es.close();
            eventSourcesRef.current.delete(docHash);
            refresh();
          }
        } catch { /* ignore */ }
      };
      es.onerror = () => {
        es.close();
        eventSourcesRef.current.delete(docHash);
        setTimeout(refresh, 2000);
      };
    },
    [refresh]
  );

  useEffect(() => {
    setIsLoading(true);
    refresh()
      .then((docs) => {
        const completed = (docs || []).filter((d) => d.status === "completed");
        completed.forEach((doc) => {
          fetchAuditSummary(doc.doc_hash)
            .then((summary) =>
              setAuditSummaries((prev) => ({ ...prev, [doc.doc_hash]: summary }))
            )
            .catch(() => {});
        });
      })
      .finally(() => setIsLoading(false));
    // Fetch facets for filter dropdowns
    fetchFacets()
      .then((f) => setFacets(f))
      .catch(() => {});
  }, [refresh, setFacets]);

  useEffect(() => {
    documents
      .filter((d) => d.status === "processing")
      .forEach((d) => subscribeToStatus(d.doc_hash));
  }, [documents, subscribeToStatus]);

  useEffect(() => {
    return () => {
      eventSourcesRef.current.forEach((es) => es.close());
      eventSourcesRef.current.clear();
    };
  }, []);

  // ── Handlers ───────────────────────────────────────────────────────

  const handleSelectDocument = useCallback(
    (hash: string, e: React.MouseEvent) => {
      // Multi-select with Ctrl/Shift
      if (e.ctrlKey || e.metaKey) {
        useDocStore.getState().toggleMultiSelect(hash);
        return;
      }
      if (e.shiftKey) {
        useDocStore.getState().rangeSelect(hash);
        return;
      }

      const prevHash = useDocStore.getState().selectedDocHash;
      selectDocument(hash);
      useViewerStore.getState().resetForNewDoc();
      useInspectStore.getState().resetInspect();
      useInspectStore.getState().resetExplorer();
      useChatStore.getState().setDocScope("selected");
      if (prevHash && prevHash !== hash) {
        const filename = useDocStore.getState().selectedFilename;
        if (filename && useChatStore.getState().messages.length > 0) {
          useChatStore.getState().insertDivider(filename);
        }
      }
    },
    [selectDocument]
  );

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    try {
      const result = await uploadDocument(file);
      await refresh();
      subscribeToStatus(result.doc_hash);
    } catch (err) {
      alert(`Upload failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleBatchTag = async () => {
    const tag = prompt("Enter tag name:");
    if (!tag) return;
    const hashes = [...multiSelected];
    await batchDocuments(hashes, "tag", [tag]);
    useDocStore.getState().clearMultiSelect();
    refresh();
  };

  const handleBatchDelete = async () => {
    const hashes = [...multiSelected];
    if (!confirm(`Archive ${hashes.length} document(s)?`)) return;
    await batchDocuments(hashes, "delete");
    useDocStore.getState().clearMultiSelect();
    refresh();
  };

  // Active filter pills
  const activeFilters: { key: string; label: string }[] = [];
  if (filters.documentType) activeFilters.push({ key: "documentType", label: formatDocType(filters.documentType) });
  if (filters.company) activeFilters.push({ key: "company", label: filters.company });
  if (filters.sector) activeFilters.push({ key: "sector", label: filters.sector });
  if (filters.project) activeFilters.push({ key: "project", label: filters.project });
  if (filters.status !== "all") activeFilters.push({ key: "status", label: filters.status });

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 h-12 border-b border-[var(--border)] flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-h1 text-[var(--text-primary)]">CENTAUR</span>
          <span
            className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]"
            style={{ boxShadow: "0 0 6px var(--accent)" }}
          />
        </div>
        <div className="flex gap-1">
          <button
            onClick={() => { setIsLoading(true); refresh().finally(() => setIsLoading(false)); }}
            className="p-1 rounded hover:bg-[var(--bg-tertiary)]"
            title="Refresh"
          >
            <RefreshCw size={14} className={isLoading ? "animate-spin" : ""} />
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-1 rounded hover:bg-[var(--bg-tertiary)]"
            title="Upload"
            disabled={isUploading}
          >
            <Upload size={14} />
          </button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.xlsx,.xls,.csv"
          onChange={handleUpload}
          className="hidden"
        />
      </div>

      {/* Search bar */}
      <div className="px-3 py-2 border-b border-[var(--border-subtle)] shrink-0">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 flex-1 bg-[var(--bg-surface)] rounded px-2 py-1">
            <Search size={12} className="text-[var(--text-tertiary)] shrink-0" />
            <input
              id="doc-search-input"
              type="text"
              placeholder="Search documents..."
              value={filters.search}
              onChange={(e) => useDocStore.getState().setFilter("search", e.target.value)}
              className="bg-transparent text-xs text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] outline-none w-full"
            />
            {filters.search && (
              <button onClick={() => useDocStore.getState().setFilter("search", "")} className="shrink-0">
                <X size={10} className="text-[var(--text-tertiary)]" />
              </button>
            )}
          </div>
          {/* Sort dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowSortMenu(!showSortMenu)}
              className="flex items-center gap-0.5 text-[10px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] px-1.5 py-1 rounded hover:bg-[var(--bg-surface)]"
            >
              {SORT_OPTIONS.find((o) => o.value === sortKey)?.label}
              <ChevronDown size={10} />
            </button>
            {showSortMenu && (
              <div className="absolute right-0 top-full mt-1 bg-[var(--bg-surface)] border border-[var(--border-sharp)] rounded shadow-lg z-20 min-w-[80px]">
                {SORT_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => { useDocStore.getState().setSortKey(opt.value); setShowSortMenu(false); }}
                    className={`block w-full text-left px-3 py-1.5 text-[10px] hover:bg-[var(--bg-tertiary)] ${
                      sortKey === opt.value ? "text-[var(--accent)]" : "text-[var(--text-secondary)]"
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Active filter pills */}
      {activeFilters.length > 0 && (
        <div className="px-3 py-1.5 flex flex-wrap gap-1 border-b border-[var(--border-subtle)] shrink-0">
          {activeFilters.map((f) => (
            <span
              key={f.key}
              className="inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] bg-[var(--accent)]/20 text-[var(--accent)] rounded"
            >
              {f.label}
              <button onClick={() => {
                if (f.key === "status") useDocStore.getState().setFilter("status", "all");
                else useDocStore.getState().setFilter(f.key as keyof typeof filters, null as never);
              }}>
                <X size={8} />
              </button>
            </span>
          ))}
          <button
            onClick={() => useDocStore.getState().clearFilters()}
            className="text-[10px] text-[var(--text-tertiary)] hover:text-[var(--text-primary)] px-1"
          >
            Clear all
          </button>
        </div>
      )}

      {/* Batch action bar */}
      {multiSelected.size > 0 && (
        <div className="px-3 py-2 flex items-center gap-2 bg-[var(--accent)]/10 border-b border-[var(--accent)]/30 shrink-0">
          <span className="text-[10px] text-[var(--accent)] font-medium">
            {multiSelected.size} selected
          </span>
          <button
            onClick={handleBatchTag}
            className="flex items-center gap-1 px-2 py-0.5 text-[10px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] bg-[var(--bg-surface)] rounded"
          >
            <Tag size={10} /> Tag
          </button>
          <button
            onClick={handleBatchDelete}
            className="flex items-center gap-1 px-2 py-0.5 text-[10px] text-red-400 hover:text-red-300 bg-[var(--bg-surface)] rounded"
          >
            <Trash2 size={10} /> Archive
          </button>
          <button
            onClick={() => useDocStore.getState().clearMultiSelect()}
            className="text-[10px] text-[var(--text-tertiary)] hover:text-[var(--text-primary)] ml-auto"
          >
            Cancel
          </button>
        </div>
      )}

      {/* Virtualized document list */}
      <div ref={parentRef} className="flex-1 overflow-auto">
        {filtered.length === 0 && !isLoading && (
          <div className="text-caption text-[var(--text-secondary)] text-center mt-8 px-4">
            {documents.length === 0 ? "0 Documents" : "No matches"}
          </div>
        )}

        <div
          style={{ height: `${rowVirtualizer.getTotalSize()}px`, width: "100%", position: "relative" }}
        >
          {rowVirtualizer.getVirtualItems().map((virtualRow) => {
            const doc = filtered[virtualRow.index];
            const isSelected = selectedDocHash === doc.doc_hash;
            const isMultiSelected = multiSelected.has(doc.doc_hash);
            const isHovered = hoverHash === doc.doc_hash;
            const showCheckbox = multiSelected.size > 0 || isHovered;

            return (
              <div
                key={doc.doc_hash}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: `${virtualRow.size}px`,
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              >
                <button
                  onClick={(e) => {
                    if (doc.status === "processing") return;
                    handleSelectDocument(doc.doc_hash, e);
                  }}
                  onMouseEnter={() => setHoverHash(doc.doc_hash)}
                  onMouseLeave={() => setHoverHash(null)}
                  disabled={doc.status === "processing"}
                  className={`w-full text-left px-3 py-2 border-b border-[var(--border-subtle)] transition-colors duration-75 ${
                    doc.status === "processing"
                      ? "opacity-60 cursor-wait"
                      : "hover:bg-[var(--bg-tertiary)]"
                  } ${isSelected ? "bg-[var(--bg-tertiary)]" : ""} ${
                    isMultiSelected ? "bg-[var(--accent)]/10" : ""
                  }`}
                  title={[
                    doc.company_name || doc.filename,
                    [formatDocType(doc.document_type), doc.period_label].filter(Boolean).join(" · "),
                    doc.sector ? `Sector: ${doc.sector}` : null,
                    doc.currency ? `Currency: ${doc.currency}` : null,
                    doc.upload_date ? `Uploaded: ${doc.upload_date.slice(0, 10)}` : null,
                  ].filter(Boolean).join("\n")}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    {/* Status dot or checkbox */}
                    <div className="w-4 shrink-0 flex items-center justify-center">
                      {doc.status === "processing" ? (
                        <Loader2 size={14} className="text-[var(--accent)] animate-spin" />
                      ) : showCheckbox ? (
                        isMultiSelected ? (
                          <CheckSquare size={14} className="text-[var(--accent)]" />
                        ) : (
                          <Square size={14} className="text-[var(--text-tertiary)]" />
                        )
                      ) : (
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            doc.status === "completed" ? "bg-green-500" :
                            doc.status === "failed" ? "bg-red-500" :
                            "bg-[var(--text-tertiary)]"
                          }`}
                        />
                      )}
                    </div>
                    {/* Two-line card */}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <p className="text-xs truncate flex-1 text-[var(--text-primary)]">
                          {doc.company_name || doc.filename}
                        </p>
                        {auditSummaries[doc.doc_hash] && (
                          <AuditBadge summary={auditSummaries[doc.doc_hash]} />
                        )}
                        {doc.status === "completed" && isHovered && (
                          <span
                            role="button"
                            tabIndex={0}
                            onClick={(e) => { e.stopPropagation(); downloadExcel(doc.doc_hash, doc.filename); }}
                            className="p-0.5 rounded hover:bg-[var(--bg-surface)] text-[var(--text-tertiary)] hover:text-[var(--text-primary)] cursor-pointer"
                            title="Export to Excel"
                          >
                            <Download size={10} />
                          </span>
                        )}
                      </div>
                      <p className="text-[10px] text-[var(--text-tertiary)] truncate">
                        {doc.status === "processing"
                          ? "Processing..."
                          : [formatDocType(doc.document_type), doc.period_label]
                              .filter(Boolean)
                              .join(" · ") || (doc.upload_date ? doc.upload_date.slice(0, 10) : "Ready")}
                      </p>
                    </div>
                  </div>
                </button>
              </div>
            );
          })}
        </div>

        {isUploading && (
          <div className="px-4 py-3 text-xs text-[var(--text-secondary)] animate-pulse">
            Uploading file...
          </div>
        )}
      </div>
    </div>
  );
}

/** Small severity dot: red (errors) > amber (warnings) > green (clean) */
function AuditBadge({ summary }: { summary: AuditSummary }) {
  const total = summary.error + summary.warning + summary.info;
  if (total === 0) return null;
  if (summary.error > 0) {
    return (
      <span
        className="w-1.5 h-1.5 rounded-full bg-red-500 shrink-0"
        title={`${summary.error} errors, ${summary.warning} warnings`}
      />
    );
  }
  return (
    <span
      className="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0"
      title={`${summary.warning} warnings, ${summary.info} info`}
    />
  );
}
