"use client";

import { useState, useRef, useCallback } from "react";
import { FileSearch, MessageSquare, BarChart3 } from "lucide-react";
import { uploadDocument, listDocuments } from "@/lib/api";
import { useDocStore } from "@/stores/useDocStore";

const FEATURES = [
  { icon: FileSearch, title: "EXTRACT", desc: "Tables, charts & metrics with spatial precision" },
  { icon: MessageSquare, title: "QUERY", desc: "Ask questions with cited, page-level answers" },
  { icon: BarChart3, title: "EXPLORE", desc: "Browse every metric in a structured ledger" },
] as const;

export default function WelcomeDropzone() {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);

  const handleFiles = useCallback(async (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadError(null);
    try {
      await uploadDocument(file);
      const docs = await listDocuments();
      useDocStore.getState().setDocuments(docs);
    } catch (err) {
      console.error("Upload failed:", err);
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current++;
    if (dragCounter.current === 1) setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current === 0) setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter.current = 0;
      setIsDragOver(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles],
  );

  return (
    <div
      className="flex items-center justify-center h-full p-8"
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        disabled={isUploading}
        className={`
          w-full max-w-lg rounded-md p-10
          flex flex-col items-center gap-6
          transition-all duration-100 cursor-pointer
          ${isDragOver
            ? "bg-[var(--bg-surface)] border border-solid border-[var(--accent)]"
            : "bg-[var(--bg-panel)] border border-dashed border-[#333333] hover:border-[#444444]"
          }
          ${isUploading ? "opacity-50 pointer-events-none" : ""}
        `}
      >
        {/* Wordmark — recessed watermark (sidebar owns the bright brand mark) */}
        <span
          className="text-[var(--text-watermark)]"
          style={{
            fontFamily: "var(--font-geist-mono), ui-monospace, monospace",
            fontSize: "12px",
            fontWeight: 500,
            letterSpacing: "0.25em",
            textTransform: "uppercase" as const,
          }}
        >
          CENTAUR
        </span>

        {/* Primary instruction — the focal point */}
        <div className="text-center">
          <p className="text-[16px] text-[var(--text-primary)]" style={{ fontFamily: "var(--font-geist-sans), sans-serif" }}>
            {isUploading ? "Uploading..." : "Drag & drop financial documents here"}
          </p>
          {!isUploading && !uploadError && (
            <p className="text-[13px] text-[#888888] mt-1" style={{ fontFamily: "var(--font-geist-sans), sans-serif" }}>
              or click to browse
            </p>
          )}
          {uploadError && (
            <p className="text-[13px] text-red-400 mt-1">
              {uploadError}
            </p>
          )}
        </div>

        {/* Divider — solid hex, no opacity */}
        <div className="w-16 h-px bg-[#222222]" />

        {/* Feature grid — solid hex colors throughout */}
        <div className="grid grid-cols-3 gap-6 w-full">
          {FEATURES.map(({ icon: Icon, title, desc }) => (
            <div key={title} className="flex flex-col items-center text-center gap-2">
              <Icon size={16} strokeWidth={1.5} className="text-[var(--text-muted)]" />
              <span
                className="text-[var(--text-secondary)]"
                style={{
                  fontFamily: "var(--font-geist-mono), ui-monospace, monospace",
                  fontSize: "12px",
                  fontWeight: 500,
                  letterSpacing: "0.05em",
                  textTransform: "uppercase" as const,
                }}
              >
                {title}
              </span>
              <span className="text-[13px] text-[var(--text-tertiary)] leading-relaxed" style={{ fontFamily: "var(--font-geist-sans), sans-serif" }}>
                {desc}
              </span>
            </div>
          ))}
        </div>
      </button>

      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.xlsx,.xls,.csv"
        onChange={(e) => handleFiles(e.target.files)}
        className="hidden"
      />
    </div>
  );
}
