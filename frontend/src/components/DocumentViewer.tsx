"use client";

import { useRef, useCallback, useEffect } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import { ChevronLeft, ChevronRight, Minus, Plus, Maximize2, PanelLeftClose } from "lucide-react";
import { Citation, getPdfUrl } from "@/lib/api";
import { useDocStore } from "@/stores/useDocStore";
import { useViewerStore } from "@/stores/useViewerStore";
import BboxOverlay from "./BboxOverlay";

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface DocumentViewerProps {
  highlightBbox?: { x: number; y: number; width: number; height: number } | null;
  activeCitation?: Citation | null;
}

export default function DocumentViewer({
  highlightBbox,
  activeCitation,
}: DocumentViewerProps) {
  const docHash = useDocStore((s) => s.selectedDocHash);
  const currentPage = useViewerStore((s) => s.currentPage);
  const numPages = useViewerStore((s) => s.numPages);
  const zoomScale = useViewerStore((s) => s.zoomScale);
  const renderedSize = useViewerStore((s) => s.renderedSize);
  const sidebarCollapsed = useViewerStore((s) => s.sidebarCollapsed);

  const setCurrentPage = useViewerStore((s) => s.setCurrentPage);
  const setNumPages = useViewerStore((s) => s.setNumPages);
  const setZoomScale = useViewerStore((s) => s.setZoomScale);
  const setRenderedSize = useViewerStore((s) => s.setRenderedSize);
  const toggleSidebar = useViewerStore((s) => s.toggleSidebar);

  const containerRef = useRef<HTMLDivElement>(null);
  const pageWrapperRef = useRef<HTMLDivElement>(null);

  const onDocumentLoadSuccess = useCallback(
    ({ numPages: n }: { numPages: number }) => setNumPages(n),
    [setNumPages]
  );

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
  }, [setRenderedSize]);

  const goToPage = useCallback(
    (page: number) => {
      if (page >= 1 && page <= numPages) setCurrentPage(page);
    },
    [numPages, setCurrentPage]
  );

  // Compute container width for fit-width mode
  const getContainerWidth = () =>
    containerRef.current?.clientWidth
      ? containerRef.current.clientWidth - 32
      : 600;

  // Zoom helpers
  const FIT_WIDTH_SENTINEL = null;
  const effectivePercent = zoomScale ? Math.round(zoomScale * 100) : null;

  const zoomIn = useCallback(() => {
    const cw = getContainerWidth();
    const fitScale = cw / 612;
    const current = useViewerStore.getState().zoomScale ?? fitScale;
    setZoomScale(Math.min(current + 0.15, 3.0));
  }, [setZoomScale]);

  const zoomOut = useCallback(() => {
    const cw = getContainerWidth();
    const fitScale = cw / 612;
    const current = useViewerStore.getState().zoomScale ?? fitScale;
    setZoomScale(Math.max(current - 0.15, 0.3));
  }, [setZoomScale]);

  const fitWidth = useCallback(() => setZoomScale(FIT_WIDTH_SENTINEL), [setZoomScale]);

  // Ctrl+Scroll zoom
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      const cw = container.clientWidth ? container.clientWidth - 32 : 600;
      const fitScale = cw / 612;
      const current = useViewerStore.getState().zoomScale ?? fitScale;
      const delta = e.deltaY > 0 ? -0.08 : 0.08;
      const next = Math.max(0.3, Math.min(3.0, current + delta));
      useViewerStore.getState().setZoomScale(next);
    };

    container.addEventListener("wheel", handleWheel, { passive: false });
    return () => container.removeEventListener("wheel", handleWheel);
  }, []);

  // Build overlay
  const inspectOverlay = highlightBbox
    ? {
        source_file: "",
        page_number: currentPage,
        blurb: "Inspected chunk",
        bbox: { page_number: currentPage, ...highlightBbox },
      }
    : null;

  const overlaySource = inspectOverlay || activeCitation || null;

  if (!docHash) {
    return (
      <div className="flex items-center justify-center h-full text-[var(--text-secondary)]">
        <p>Select a document to view</p>
      </div>
    );
  }

  const pdfUrl = getPdfUrl(docHash);
  const containerWidth = getContainerWidth();

  return (
    <div className="flex flex-col h-full relative">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--border)]">
        <div className="flex items-center gap-2">
          <button
            onClick={toggleSidebar}
            className="p-1 rounded hover:bg-[var(--bg-tertiary)]"
            title={sidebarCollapsed ? "Show sidebar" : "Hide sidebar"}
          >
            <PanelLeftClose
              size={18}
              className={`transition-transform ${sidebarCollapsed ? "rotate-180" : ""}`}
            />
          </button>
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
              scale={zoomScale ?? undefined}
              width={zoomScale ? undefined : containerWidth}
              renderTextLayer={false}
              renderAnnotationLayer={false}
            />
          </Document>

          {overlaySource && (
            <BboxOverlay
              citation={overlaySource}
              containerWidth={renderedSize.width}
              containerHeight={renderedSize.height}
              isActive
            />
          )}
        </div>
      </div>

      {/* Floating zoom HUD */}
      <div className="zoom-hud">
        <button onClick={zoomOut} className="zoom-btn" title="Zoom out (Ctrl+Scroll)">
          <Minus size={14} />
        </button>
        <span className="text-xs text-[var(--text-secondary)] w-10 text-center font-mono">
          {effectivePercent ? `${effectivePercent}%` : "Fit"}
        </span>
        <button onClick={zoomIn} className="zoom-btn" title="Zoom in (Ctrl+Scroll)">
          <Plus size={14} />
        </button>
        <div className="w-px h-4 bg-[var(--border)] mx-1" />
        <button
          onClick={fitWidth}
          className={`zoom-btn ${zoomScale === null ? "text-[var(--accent)]" : ""}`}
          title="Fit width"
        >
          <Maximize2 size={14} />
        </button>
      </div>
    </div>
  );
}
