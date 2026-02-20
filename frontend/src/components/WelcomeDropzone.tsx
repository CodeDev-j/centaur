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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);

  const handleFiles = useCallback(async (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      await uploadDocument(file);
      const docs = await listDocuments();
      useDocStore.getState().setDocuments(docs);
    } catch (err) {
      console.error("Upload failed:", err);
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
          w-full max-w-lg rounded-xl border border-dashed p-10
          flex flex-col items-center gap-6
          transition-all duration-200 cursor-pointer
          ${isDragOver
            ? "border-[var(--accent)] bg-[rgba(79,70,229,0.05)] shadow-[inset_0_0_0_2px_var(--accent)]"
            : "border-[var(--border-subtle)] hover:border-[var(--border-sharp)]"
          }
          ${isUploading ? "opacity-50 pointer-events-none" : ""}
        `}
      >
        {/* Wordmark */}
        <h1 className="text-h1 text-[var(--text-primary)]">CENTAUR</h1>

        {/* Instruction */}
        <div className="text-center">
          <p className="text-caption text-[var(--text-secondary)]">
            {isUploading ? "Uploading..." : "Drag & drop financial documents here"}
          </p>
          {!isUploading && (
            <p className="text-caption text-[var(--text-secondary)] mt-1 opacity-60">
              or click to browse
            </p>
          )}
        </div>

        {/* Divider */}
        <div className="w-16 h-px bg-[var(--border-subtle)]" />

        {/* Feature grid */}
        <div className="grid grid-cols-3 gap-6 w-full">
          {FEATURES.map(({ icon: Icon, title, desc }) => (
            <div key={title} className="flex flex-col items-center text-center gap-2">
              <Icon size={18} className="text-[var(--text-secondary)] opacity-60" />
              <span className="text-h2 text-[var(--text-secondary)]">{title}</span>
              <span className="text-caption text-[var(--text-secondary)] opacity-50 leading-tight">
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
