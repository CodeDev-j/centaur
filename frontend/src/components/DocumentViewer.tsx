"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Citation, getPdfUrl } from "@/lib/api";
import BboxOverlay from "./BboxOverlay";

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface DocumentViewerProps {
  docHash: string | null;
  citations: Citation[];
  activeCitationIndex: number | null;
}

export default function DocumentViewer({
  docHash,
  citations,
  activeCitationIndex,
}: DocumentViewerProps) {
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [renderedSize, setRenderedSize] = useState({ width: 0, height: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const pageWrapperRef = useRef<HTMLDivElement>(null);

  const onDocumentLoadSuccess = useCallback(
    ({ numPages: n }: { numPages: number }) => {
      setNumPages(n);
      setCurrentPage(1);
    },
    []
  );

  // After the page renders, measure the actual canvas dimensions
  const onPageRenderSuccess = useCallback(() => {
    const wrapper = pageWrapperRef.current;
    if (wrapper) {
      const canvas = wrapper.querySelector("canvas");
      if (canvas) {
        setRenderedSize({
          width: canvas.clientWidth,
          height: canvas.clientHeight,
        });
      }
    }
  }, []);

  const goToPage = useCallback(
    (page: number) => {
      if (page >= 1 && page <= numPages) {
        setCurrentPage(page);
      }
    },
    [numPages]
  );

  // Navigate to cited page when a citation badge is clicked
  useEffect(() => {
    if (activeCitationIndex !== null && activeCitationIndex < citations.length) {
      const citation = citations[activeCitationIndex];
      const targetPage = citation.bbox?.page_number || citation.page_number;
      if (targetPage && targetPage >= 1 && targetPage <= numPages) {
        setCurrentPage(targetPage);
      }
    }
  }, [activeCitationIndex, citations, numPages]);

  // Only show the actively-clicked citation on the current page
  const activeCitation =
    activeCitationIndex !== null &&
    activeCitationIndex < citations.length &&
    citations[activeCitationIndex]?.bbox?.page_number === currentPage
      ? citations[activeCitationIndex]
      : null;

  if (!docHash) {
    return (
      <div className="flex items-center justify-center h-full text-[var(--text-secondary)]">
        <p>Select a document to view</p>
      </div>
    );
  }

  const pdfUrl = getPdfUrl(docHash);

  return (
    <div className="flex flex-col h-full">
      {/* Page controls */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--border)]">
        <button
          onClick={() => goToPage(currentPage - 1)}
          disabled={currentPage <= 1}
          className="p-1 rounded hover:bg-[var(--bg-tertiary)] disabled:opacity-30"
        >
          <ChevronLeft size={20} />
        </button>
        <span className="text-sm text-[var(--text-secondary)]">
          Page {currentPage} of {numPages}
        </span>
        <button
          onClick={() => goToPage(currentPage + 1)}
          disabled={currentPage >= numPages}
          className="p-1 rounded hover:bg-[var(--bg-tertiary)] disabled:opacity-30"
        >
          <ChevronRight size={20} />
        </button>
      </div>

      {/* PDF display */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto flex justify-center p-4"
      >
        <div className="pdf-page-container" ref={pageWrapperRef}>
          <Document
            file={pdfUrl}
            onLoadSuccess={onDocumentLoadSuccess}
            loading={
              <div className="text-[var(--text-secondary)] p-8">Loading PDF...</div>
            }
            error={
              <div className="text-red-400 p-8">Failed to load PDF</div>
            }
          >
            <Page
              pageNumber={currentPage}
              onRenderSuccess={onPageRenderSuccess}
              width={
                containerRef.current?.clientWidth
                  ? containerRef.current.clientWidth - 32
                  : 600
              }
              renderTextLayer={false}
              renderAnnotationLayer={false}
            />
          </Document>

          {/* Citation overlay — only the active citation */}
          {activeCitation && (
            <BboxOverlay
              citation={activeCitation}
              containerWidth={renderedSize.width}
              containerHeight={renderedSize.height}
              isActive
            />
          )}
        </div>
      </div>
    </div>
  );
}
