"use client";

import { Citation } from "@/lib/api";

interface BboxOverlayProps {
  citation: Citation;
  containerWidth: number;
  containerHeight: number;
  isActive?: boolean;
}

/**
 * Absolutely-positioned highlight div overlaid on a PDF page.
 * Scales normalized 0-1 coordinates to actual container dimensions.
 * Uses mix-blend-mode: multiply so text underneath stays legible.
 */
export default function BboxOverlay({
  citation,
  containerWidth,
  containerHeight,
  isActive = false,
}: BboxOverlayProps) {
  if (!citation.bbox) return null;

  const { x, y, width, height } = citation.bbox;
  const pad = 0.003;

  const style: React.CSSProperties = {
    left: Math.max(0, x - pad) * containerWidth,
    top: Math.max(0, y - pad) * containerHeight,
    width: (width + pad * 2) * containerWidth,
    height: (height + pad * 2) * containerHeight,
  };

  return (
    <div
      className={`bbox-overlay ${isActive ? "active" : ""}`}
      style={style}
      title={citation.blurb}
    />
  );
}
