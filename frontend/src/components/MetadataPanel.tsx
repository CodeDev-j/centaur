"use client";

import { useEffect, useState, useCallback } from "react";
import { useDocStore } from "@/stores/useDocStore";
import {
  DocumentMeta,
  fetchDocumentMeta,
  updateDocumentMeta,
} from "@/lib/api";
import { Save, RotateCcw, X, Plus } from "lucide-react";

// ── Document type labels ─────────────────────────────────────────────

const DOC_TYPE_OPTIONS = [
  "earnings_release",
  "earnings_presentation",
  "annual_report",
  "10-K",
  "10-Q",
  "investor_presentation",
  "credit_agreement",
  "cim",
  "lender_presentation",
  "board_materials",
  "due_diligence",
  "valuation_report",
  "pitch_book",
  "research_report",
  "press_release",
  "regulatory_filing",
  "other",
] as const;

const DOC_TYPE_LABEL: Record<string, string> = {
  earnings_release: "Earnings Release",
  earnings_presentation: "Earnings Presentation",
  annual_report: "Annual Report",
  "10-K": "10-K",
  "10-Q": "10-Q",
  investor_presentation: "Investor Presentation",
  credit_agreement: "Credit Agreement",
  cim: "CIM",
  lender_presentation: "Lender Presentation",
  board_materials: "Board Materials",
  due_diligence: "Due Diligence",
  valuation_report: "Valuation Report",
  pitch_book: "Pitch Book",
  research_report: "Research Report",
  press_release: "Press Release",
  regulatory_filing: "Regulatory Filing",
  other: "Other",
};

// ── Field Badge ──────────────────────────────────────────────────────

function FieldBadge({ isOverridden }: { isOverridden: boolean }) {
  return (
    <span
      className={`text-[9px] font-mono uppercase px-1.5 py-0.5 rounded ${
        isOverridden
          ? "bg-amber-500/20 text-amber-400"
          : "bg-[var(--accent)]/15 text-[var(--accent)]"
      }`}
    >
      {isOverridden ? "EDITED" : "AUTO"}
    </span>
  );
}

// ── Tag Input ────────────────────────────────────────────────────────

function TagInput({
  tags,
  onChange,
}: {
  tags: string[];
  onChange: (tags: string[]) => void;
}) {
  const [input, setInput] = useState("");

  const addTag = () => {
    const tag = input.trim().toLowerCase();
    if (tag && !tags.includes(tag)) {
      onChange([...tags, tag]);
    }
    setInput("");
  };

  return (
    <div className="flex flex-wrap gap-1.5 items-center">
      {tags.map((tag) => (
        <span
          key={tag}
          className="flex items-center gap-1 px-2 py-0.5 rounded bg-[var(--bg-desk)] text-[var(--text-secondary)] text-xs"
        >
          {tag}
          <button
            onClick={() => onChange(tags.filter((t) => t !== tag))}
            className="hover:text-red-400 transition-colors"
          >
            <X size={10} />
          </button>
        </span>
      ))}
      <div className="flex items-center gap-1">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              addTag();
            }
          }}
          placeholder="Add tag..."
          className="w-20 bg-transparent text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none"
        />
        <button
          onClick={addTag}
          className="text-[var(--text-muted)] hover:text-[var(--text-primary)]"
        >
          <Plus size={12} />
        </button>
      </div>
    </div>
  );
}

// ── Form Field ───────────────────────────────────────────────────────

function FormField({
  label,
  fieldName,
  overrides,
  children,
}: {
  label: string;
  fieldName: string;
  overrides: Record<string, boolean>;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wider">
          {label}
        </label>
        <FieldBadge isOverridden={!!overrides[fieldName]} />
      </div>
      {children}
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────

export default function MetadataPanel() {
  const selectedDocHash = useDocStore((s) => s.selectedDocHash);
  const selectedFilename = useDocStore((s) => s.selectedFilename);

  const [meta, setMeta] = useState<DocumentMeta | null>(null);
  const [draft, setDraft] = useState<Partial<DocumentMeta>>({});
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load metadata when document changes
  useEffect(() => {
    if (!selectedDocHash) {
      setMeta(null);
      setDraft({});
      return;
    }
    setLoading(true);
    setError(null);
    fetchDocumentMeta(selectedDocHash)
      .then((m) => {
        setMeta(m);
        setDraft({});
      })
      .catch((err) => {
        setError(err.message);
        setMeta(null);
      })
      .finally(() => setLoading(false));
  }, [selectedDocHash]);

  const isDirty = Object.keys(draft).length > 0;

  const updateDraft = useCallback(
    <K extends keyof DocumentMeta>(field: K, value: DocumentMeta[K]) => {
      setDraft((prev) => {
        // If reverting to original value, remove from draft
        if (meta && meta[field] === value) {
          const next = { ...prev };
          delete next[field];
          return next;
        }
        return { ...prev, [field]: value };
      });
    },
    [meta]
  );

  const handleSave = async () => {
    if (!selectedDocHash || !isDirty) return;
    setSaving(true);
    try {
      const updated = await updateDocumentMeta(selectedDocHash, draft);
      setMeta(updated);
      setDraft({});
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => setDraft({});

  // Resolved values: draft overrides meta
  const val = <K extends keyof DocumentMeta>(field: K): DocumentMeta[K] | undefined =>
    field in draft ? (draft[field] as DocumentMeta[K]) : meta?.[field];

  const overrides = meta?.user_overrides ?? {};

  const inputClass =
    "w-full bg-[var(--bg-desk)] border border-[var(--border-subtle)] rounded px-2.5 py-1.5 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-[var(--accent)] focus:outline-none transition-colors";

  if (!selectedDocHash) {
    return (
      <div className="h-full flex items-center justify-center px-6">
        <p className="text-sm text-[var(--text-muted)] text-center">
          Select a document to view its metadata.
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="w-5 h-5 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (error && !meta) {
    return (
      <div className="h-full flex items-center justify-center px-6">
        <p className="text-sm text-red-400 text-center">{error}</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[var(--border-subtle)] shrink-0">
        <h2 className="text-h2 text-[var(--text-primary)]">METADATA</h2>
        <p className="text-xs text-[var(--text-muted)] mt-0.5 truncate">
          {selectedFilename}
        </p>
      </div>

      {/* Form */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4 scrollbar-thin">
        {/* Company Name */}
        <FormField label="Company" fieldName="company_name" overrides={overrides}>
          <input
            className={inputClass}
            value={(val("company_name") as string) ?? ""}
            onChange={(e) => updateDraft("company_name", e.target.value || null)}
            placeholder="e.g. Alphabet Inc."
          />
        </FormField>

        {/* Document Type */}
        <FormField label="Document Type" fieldName="document_type" overrides={overrides}>
          <select
            className={inputClass}
            value={(val("document_type") as string) ?? ""}
            onChange={(e) => updateDraft("document_type", e.target.value || null)}
          >
            <option value="">— Select —</option>
            {DOC_TYPE_OPTIONS.map((t) => (
              <option key={t} value={t}>
                {DOC_TYPE_LABEL[t] ?? t}
              </option>
            ))}
          </select>
        </FormField>

        {/* Project / Deal Code */}
        <FormField label="Project / Deal" fieldName="project_code" overrides={overrides}>
          <input
            className={inputClass}
            value={(val("project_code") as string) ?? ""}
            onChange={(e) => updateDraft("project_code", e.target.value || null)}
            placeholder="e.g. Project Phoenix"
          />
        </FormField>

        {/* Period Label */}
        <FormField label="Period" fieldName="period_label" overrides={overrides}>
          <input
            className={inputClass}
            value={(val("period_label") as string) ?? ""}
            onChange={(e) => updateDraft("period_label", e.target.value || null)}
            placeholder="e.g. Q1 2025"
          />
        </FormField>

        {/* As-of Date */}
        <FormField label="As-of Date" fieldName="as_of_date" overrides={overrides}>
          <input
            type="date"
            className={inputClass}
            value={(val("as_of_date") as string) ?? ""}
            onChange={(e) => updateDraft("as_of_date", e.target.value || null)}
          />
        </FormField>

        {/* Sector */}
        <FormField label="Sector" fieldName="sector" overrides={overrides}>
          <input
            className={inputClass}
            value={(val("sector") as string) ?? ""}
            onChange={(e) => updateDraft("sector", e.target.value || null)}
            placeholder="e.g. Technology"
          />
        </FormField>

        {/* Geography */}
        <FormField label="Geography" fieldName="geography" overrides={overrides}>
          <input
            className={inputClass}
            value={(val("geography") as string) ?? ""}
            onChange={(e) => updateDraft("geography", e.target.value || null)}
            placeholder="e.g. United States"
          />
        </FormField>

        {/* Currency */}
        <FormField label="Currency" fieldName="currency" overrides={overrides}>
          <input
            className={inputClass}
            value={(val("currency") as string) ?? ""}
            onChange={(e) => updateDraft("currency", e.target.value || null)}
            placeholder="e.g. USD"
            maxLength={3}
          />
        </FormField>

        {/* Tags */}
        <FormField label="Tags" fieldName="tags" overrides={overrides}>
          <TagInput
            tags={(val("tags") as string[]) ?? []}
            onChange={(tags) => updateDraft("tags", tags)}
          />
        </FormField>

        {/* Notes */}
        <FormField label="Notes" fieldName="notes" overrides={overrides}>
          <textarea
            className={`${inputClass} min-h-[60px] resize-y`}
            value={(val("notes") as string) ?? ""}
            onChange={(e) => updateDraft("notes", e.target.value || null)}
            placeholder="Internal notes..."
            rows={3}
          />
        </FormField>
      </div>

      {/* Footer: read-only stats + save bar */}
      <div className="shrink-0 border-t border-[var(--border-subtle)]">
        {/* Read-only info */}
        <div className="px-4 py-2 flex items-center gap-4 text-[10px] text-[var(--text-muted)] font-mono">
          <span>{meta?.page_count ?? 0} pages</span>
          <span>{meta?.language?.toUpperCase() ?? "EN"}</span>
          <span>
            Confidence: {((meta?.extraction_confidence ?? 0) * 100).toFixed(0)}%
          </span>
          {meta?.last_edited_at && (
            <span>Edited: {new Date(meta.last_edited_at).toLocaleDateString()}</span>
          )}
        </div>

        {/* Save bar (only shown when dirty) */}
        {isDirty && (
          <div className="px-4 py-2 flex items-center gap-2 border-t border-[var(--border-subtle)] bg-[var(--bg-surface)]">
            <button
              onClick={handleSave}
              disabled={saving}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-[var(--accent)] text-white hover:brightness-110 disabled:opacity-50 transition-all"
            >
              <Save size={12} />
              {saving ? "Saving..." : "Save"}
            </button>
            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-desk)] transition-all"
            >
              <RotateCcw size={12} />
              Reset
            </button>
            {error && <span className="text-xs text-red-400 ml-auto">{error}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
