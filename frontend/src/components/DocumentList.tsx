"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { FileText, Upload, RefreshCw, Loader2, Download, AlertTriangle, AlertCircle, Info } from "lucide-react";
import {
  listDocuments,
  uploadDocument,
  getIngestStatusUrl,
  fetchAuditSummary,
  downloadExcel,
  DocumentSummary,
  AuditSummary,
} from "@/lib/api";
import { useDocStore } from "@/stores/useDocStore";
import { useViewerStore } from "@/stores/useViewerStore";
import { useInspectStore } from "@/stores/useInspectStore";
import { useChatStore } from "@/stores/useChatStore";

export default function DocumentList() {
  const documents = useDocStore((s) => s.documents);
  const selectedDocHash = useDocStore((s) => s.selectedDocHash);
  const isUploading = useDocStore((s) => s.isUploading);
  const setDocuments = useDocStore((s) => s.setDocuments);
  const selectDocument = useDocStore((s) => s.selectDocument);
  const setIsUploading = useDocStore((s) => s.setIsUploading);

  const [isLoading, setIsLoading] = useState(false);
  const [auditSummaries, setAuditSummaries] = useState<Record<string, AuditSummary>>({});
  const fileInputRef = useRef<HTMLInputElement>(null);
  const eventSourcesRef = useRef<Map<string, EventSource>>(new Map());

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
        } catch {
          // Ignore parse errors
        }
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
        // Fetch audit summaries for all completed documents
        const completed = (docs || []).filter((d) => d.status === "completed");
        completed.forEach((doc) => {
          fetchAuditSummary(doc.doc_hash)
            .then((summary) => {
              setAuditSummaries((prev) => ({ ...prev, [doc.doc_hash]: summary }));
            })
            .catch(() => { /* non-critical */ });
        });
      })
      .finally(() => setIsLoading(false));
  }, [refresh]);

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

  const handleSelectDocument = useCallback(
    (hash: string) => {
      const prevHash = useDocStore.getState().selectedDocHash;
      selectDocument(hash);
      useViewerStore.getState().resetForNewDoc();
      useInspectStore.getState().resetInspect();
      useInspectStore.getState().resetExplorer();

      // Auto-activate doc scope when selecting a document
      useChatStore.getState().setDocScope("selected");

      // Insert divider if switching documents mid-conversation
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
      alert(
        `Upload failed: ${err instanceof Error ? err.message : "Unknown error"}`
      );
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const statusLabel = (status: string) => {
    switch (status) {
      case "processing": return "Processing...";
      case "completed": return "Ready";
      case "failed": return "Failed";
      default: return status;
    }
  };

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
            onClick={() => {
              setIsLoading(true);
              refresh().finally(() => setIsLoading(false));
            }}
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

      {/* Document list */}
      <div className="flex-1 overflow-auto">
        {documents.length === 0 && !isLoading && (
          <div className="text-caption text-[var(--text-secondary)] text-center mt-8 px-4">
            0 Documents
          </div>
        )}

        {documents.map((doc) => (
          <button
            key={doc.doc_hash}
            onClick={() => handleSelectDocument(doc.doc_hash)}
            disabled={doc.status === "processing"}
            className={`w-full text-left px-4 py-3 border-b border-[var(--border)] transition-colors duration-100 ${
              doc.status === "processing"
                ? "opacity-60 cursor-wait"
                : "hover:bg-[var(--bg-tertiary)]"
            } ${selectedDocHash === doc.doc_hash ? "bg-[var(--bg-tertiary)]" : ""}`}
          >
            <div className="flex items-start gap-2">
              {doc.status === "processing" ? (
                <Loader2
                  size={16}
                  className="mt-0.5 shrink-0 text-[var(--accent)] animate-spin"
                />
              ) : (
                <FileText
                  size={16}
                  className="mt-0.5 shrink-0 text-[var(--text-secondary)]"
                />
              )}
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5">
                  <p className="text-sm truncate flex-1">{doc.filename}</p>
                  {/* Audit severity badge */}
                  {auditSummaries[doc.doc_hash] && (
                    <AuditBadge summary={auditSummaries[doc.doc_hash]} />
                  )}
                  {/* Excel export button */}
                  {doc.status === "completed" && (
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={(e) => {
                        e.stopPropagation();
                        downloadExcel(doc.doc_hash, doc.filename);
                      }}
                      className="p-0.5 rounded hover:bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
                      title="Export to Excel"
                    >
                      <Download size={12} />
                    </span>
                  )}
                </div>
                <p className="text-xs text-[var(--text-secondary)] mt-0.5">
                  {statusLabel(doc.status)}
                  {doc.upload_date &&
                    doc.status !== "processing" &&
                    ` · ${new Date(doc.upload_date).toISOString().slice(0, 10)}`}
                </p>
              </div>
            </div>
          </button>
        ))}

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
  if (total === 0) {
    return (
      <span
        className="w-2 h-2 rounded-full bg-green-500 shrink-0"
        title="No audit findings"
      />
    );
  }
  if (summary.error > 0) {
    return (
      <span
        className="w-2 h-2 rounded-full bg-red-500 shrink-0"
        title={`${summary.error} errors, ${summary.warning} warnings`}
      />
    );
  }
  return (
    <span
      className="w-2 h-2 rounded-full bg-amber-500 shrink-0"
      title={`${summary.warning} warnings, ${summary.info} info`}
    />
  );
}
