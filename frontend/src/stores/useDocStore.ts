import { create } from "zustand";
import { DocumentSummary } from "@/lib/api";

interface DocState {
  documents: DocumentSummary[];
  selectedDocHash: string | null;
  selectedFilename: string | null;
  isUploading: boolean;

  setDocuments: (docs: DocumentSummary[]) => void;
  selectDocument: (hash: string | null) => void;
  setIsUploading: (v: boolean) => void;
}

export const useDocStore = create<DocState>((set, get) => ({
  documents: [],
  selectedDocHash: null,
  selectedFilename: null,
  isUploading: false,

  setDocuments: (docs) => set({ documents: docs }),

  selectDocument: (hash) => {
    const docs = get().documents;
    const doc = hash ? docs.find((d) => d.doc_hash === hash) : null;
    set({
      selectedDocHash: hash,
      selectedFilename: doc?.filename ?? null,
    });
  },

  setIsUploading: (v) => set({ isUploading: v }),
}));
