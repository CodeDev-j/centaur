"use client";

import { useState, useCallback } from "react";
import DocumentList from "@/components/DocumentList";
import DocumentViewer from "@/components/DocumentViewer";
import ChatPanel from "@/components/ChatPanel";
import { Citation } from "@/lib/api";

export default function Home() {
  const [selectedDocHash, setSelectedDocHash] = useState<string | null>(null);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [activeCitationIndex, setActiveCitationIndex] = useState<number | null>(
    null
  );

  const handleCitationsUpdate = useCallback((newCitations: Citation[]) => {
    setCitations(newCitations);
    setActiveCitationIndex(null);
  }, []);

  const handleCitationClick = useCallback(
    (index: number) => {
      const citation = citations[index];
      if (!citation) return;

      // Auto-select the document if not already open
      if (citation.doc_hash && citation.doc_hash !== selectedDocHash) {
        setSelectedDocHash(citation.doc_hash);
      }

      // Set active citation — DocumentViewer's useEffect navigates to the page
      setActiveCitationIndex(index);
    },
    [citations, selectedDocHash]
  );

  return (
    <div className="h-screen flex">
      {/* Left sidebar: Document list */}
      <div className="w-64 border-r border-[var(--border)] bg-[var(--bg-secondary)] shrink-0">
        <DocumentList
          selectedDocHash={selectedDocHash}
          onSelectDocument={setSelectedDocHash}
        />
      </div>

      {/* Center: Document viewer */}
      <div className="flex-1 min-w-0 bg-[var(--bg-primary)]">
        <DocumentViewer
          docHash={selectedDocHash}
          citations={citations}
          activeCitationIndex={activeCitationIndex}
        />
      </div>

      {/* Right panel: Chat */}
      <div className="w-96 border-l border-[var(--border)] bg-[var(--bg-secondary)] shrink-0">
        <ChatPanel
          onCitationsUpdate={handleCitationsUpdate}
          onCitationClick={handleCitationClick}
        />
      </div>
    </div>
  );
}
