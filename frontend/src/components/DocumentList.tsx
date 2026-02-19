"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { FileText, Upload, RefreshCw, Loader2 } from "lucide-react";
import {
  listDocuments,
  uploadDocument,
  getIngestStatusUrl,
  DocumentSummary,
} from "@/lib/api";

interface DocumentListProps {
  selectedDocHash: string | null;
  onSelectDocument: (hash: string) => void;
}

export default function DocumentList({
  selectedDocHash,
  onSelectDocument,
}: DocumentListProps) {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
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
  }, []);

  // Subscribe to SSE for a processing document
  const subscribeToStatus = useCallback(
    (docHash: string) => {
      // Don't double-subscribe
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
        // Fallback: single refresh after a short delay
        setTimeout(refresh, 2000);
      };
    },
    [refresh]
  );

  // Initial load
  useEffect(() => {
    setIsLoading(true);
    refresh().finally(() => setIsLoading(false));
  }, [refresh]);

  // Subscribe to SSE for any documents that are still processing
  // (handles page reloads while ingestion is running)
  useEffect(() => {
    documents
      .filter((d) => d.status === "processing")
      .forEach((d) => subscribeToStatus(d.doc_hash));
  }, [documents, subscribeToStatus]);

  // Cleanup all EventSource connections on unmount
  useEffect(() => {
    return () => {
      eventSourcesRef.current.forEach((es) => es.close());
      eventSourcesRef.current.clear();
    };
  }, []);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const result = await uploadDocument(file);
      // Refresh list so the new doc appears with status="processing"
      await refresh();
      // Subscribe to SSE for instant notification on completion
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
      case "processing":
        return "Processing...";
      case "completed":
        return "Ready";
      case "failed":
        return "Failed";
      default:
        return status;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[var(--border)] flex items-center justify-between">
        <h2 className="font-semibold text-sm">Documents</h2>
        <div className="flex gap-1">
          <button
            onClick={() => {
              setIsLoading(true);
              refresh().finally(() => setIsLoading(false));
            }}
            className="p-1 rounded hover:bg-[var(--bg-tertiary)]"
            title="Refresh"
          >
            <RefreshCw
              size={14}
              className={isLoading ? "animate-spin" : ""}
            />
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
          <div className="text-[var(--text-secondary)] text-xs text-center mt-8 px-4">
            No documents ingested yet. Upload a PDF to get started.
          </div>
        )}

        {documents.map((doc) => (
          <button
            key={doc.doc_hash}
            onClick={() => onSelectDocument(doc.doc_hash)}
            disabled={doc.status === "processing"}
            className={`w-full text-left px-4 py-3 border-b border-[var(--border)] transition-colors ${
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
              <div className="min-w-0">
                <p className="text-sm truncate">{doc.filename}</p>
                <p className="text-xs text-[var(--text-secondary)] mt-0.5">
                  {statusLabel(doc.status)}
                  {doc.upload_date &&
                    doc.status !== "processing" &&
                    ` · ${new Date(doc.upload_date).toLocaleDateString()}`}
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
