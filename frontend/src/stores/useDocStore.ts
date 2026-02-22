import { create } from "zustand";
import Fuse, { IFuseOptions } from "fuse.js";
import { DocumentSummary, Facets } from "@/lib/api";

// ── Filter state ─────────────────────────────────────────────────────

export interface DocFilters {
  search: string;
  documentType: string | null;
  company: string | null;
  sector: string | null;
  project: string | null;
  status: "all" | "completed" | "processing" | "failed";
}

export type DocSortKey = "recent" | "name" | "company" | "doc_date";

const DEFAULT_FILTERS: DocFilters = {
  search: "",
  documentType: null,
  company: null,
  sector: null,
  project: null,
  status: "all",
};

// ── Fuse.js config ───────────────────────────────────────────────────

const FUSE_OPTIONS: IFuseOptions<DocumentSummary> = {
  keys: [
    { name: "filename", weight: 0.3 },
    { name: "company_name", weight: 0.3 },
    { name: "document_type", weight: 0.15 },
    { name: "sector", weight: 0.1 },
    { name: "project_code", weight: 0.1 },
    { name: "tags", weight: 0.05 },
  ],
  threshold: 0.4,
  ignoreLocation: true,
};

// ── Store ────────────────────────────────────────────────────────────

interface DocState {
  // Data
  documents: DocumentSummary[];
  facets: Facets | null;

  // Selection
  selectedDocHash: string | null;
  selectedFilename: string | null;
  multiSelected: Set<string>;

  // Search & Filter
  filters: DocFilters;
  sortKey: DocSortKey;

  // UI
  isUploading: boolean;
  sidebarWidth: number;

  // Computed
  filteredDocuments: () => DocumentSummary[];

  // Actions
  setDocuments: (docs: DocumentSummary[]) => void;
  setFacets: (facets: Facets) => void;
  selectDocument: (hash: string | null) => void;
  toggleMultiSelect: (hash: string) => void;
  rangeSelect: (hash: string) => void;
  clearMultiSelect: () => void;
  setFilter: <K extends keyof DocFilters>(key: K, value: DocFilters[K]) => void;
  clearFilters: () => void;
  setSortKey: (key: DocSortKey) => void;
  setIsUploading: (v: boolean) => void;
  setSidebarWidth: (w: number) => void;
}

export const useDocStore = create<DocState>((set, get) => ({
  documents: [],
  facets: null,
  selectedDocHash: null,
  selectedFilename: null,
  multiSelected: new Set(),
  filters: { ...DEFAULT_FILTERS },
  sortKey: "recent",
  isUploading: false,
  sidebarWidth: 256,

  filteredDocuments: () => {
    const { documents, filters, sortKey } = get();
    let result = [...documents];

    // Status filter
    if (filters.status !== "all") {
      result = result.filter((d) => d.status === filters.status);
    }

    // Facet filters
    if (filters.documentType) {
      result = result.filter((d) => d.document_type === filters.documentType);
    }
    if (filters.company) {
      result = result.filter((d) => d.company_name === filters.company);
    }
    if (filters.sector) {
      result = result.filter((d) => d.sector === filters.sector);
    }
    if (filters.project) {
      result = result.filter((d) => d.project_code === filters.project);
    }

    // Fuzzy search (applied last)
    if (filters.search.trim()) {
      const fuse = new Fuse(result, FUSE_OPTIONS);
      result = fuse.search(filters.search).map((r) => r.item);
    }

    // Sort
    result.sort((a, b) => {
      switch (sortKey) {
        case "name":
          return (a.filename || "").localeCompare(b.filename || "");
        case "company":
          return (a.company_name || "zzz").localeCompare(b.company_name || "zzz");
        case "doc_date":
          return (b.as_of_date || "").localeCompare(a.as_of_date || "");
        default: // "recent"
          return (b.upload_date || "").localeCompare(a.upload_date || "");
      }
    });

    return result;
  },

  setDocuments: (docs) => set({ documents: docs }),
  setFacets: (facets) => set({ facets }),

  selectDocument: (hash) => {
    const docs = get().documents;
    const doc = hash ? docs.find((d) => d.doc_hash === hash) : null;
    set({
      selectedDocHash: hash,
      selectedFilename: doc?.filename ?? null,
      multiSelected: new Set(),
    });
  },

  toggleMultiSelect: (hash) => {
    const next = new Set(get().multiSelected);
    if (next.has(hash)) next.delete(hash);
    else next.add(hash);
    set({ multiSelected: next });
  },

  rangeSelect: (hash) => {
    const { multiSelected } = get();
    const filtered = get().filteredDocuments();
    const lastSelected = [...multiSelected].pop();
    if (!lastSelected) {
      set({ multiSelected: new Set([hash]) });
      return;
    }
    const startIdx = filtered.findIndex((d) => d.doc_hash === lastSelected);
    const endIdx = filtered.findIndex((d) => d.doc_hash === hash);
    if (startIdx === -1 || endIdx === -1) return;
    const [lo, hi] = startIdx < endIdx ? [startIdx, endIdx] : [endIdx, startIdx];
    const next = new Set(multiSelected);
    for (let i = lo; i <= hi; i++) next.add(filtered[i].doc_hash);
    set({ multiSelected: next });
  },

  clearMultiSelect: () => set({ multiSelected: new Set() }),

  setFilter: (key, value) =>
    set((s) => ({ filters: { ...s.filters, [key]: value } })),

  clearFilters: () => set({ filters: { ...DEFAULT_FILTERS } }),

  setSortKey: (key) => set({ sortKey: key }),
  setIsUploading: (v) => set({ isUploading: v }),
  setSidebarWidth: (w) => set({ sidebarWidth: Math.max(256, Math.min(400, w)) }),
}));
