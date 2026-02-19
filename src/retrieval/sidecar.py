"""
Citation Sidecar: The Trust Engine.

Manages the mapping between retrieved chunks and citation markers in generated text.
Ensures every factual claim can be traced back to a source document location.

Flow:
1. build_context_with_citations() → assigns [1], [2] markers to chunks,
   builds XML context block for generation LLM
2. assemble_cited_answer() → takes structured LLM output (segments + source_ids),
   deduplicates within each segment (same claim → one badge), assigns badges
   by first-appearance order, resolves bboxes from each badge's own citing text
3. _resolve_fine_bbox() → matches answer values against value_bboxes for
   cell-level / data-point-level highlighting instead of whole-region highlighting
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from langsmith import traceable

from src.schemas.citation import Citation, BoundingBox
from src.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def build_context_with_citations(
    chunks: List[RetrievedChunk],
) -> Tuple[str, Dict[int, RetrievedChunk]]:
    """
    Builds an XML context block with ephemeral citation IDs.

    Returns:
        context_str: XML-formatted context for the generation LLM
        sidecar_map: {1: chunk, 2: chunk, ...} for citation resolution
    """
    sidecar_map: Dict[int, RetrievedChunk] = {}
    blocks = []

    for i, chunk in enumerate(chunks, start=1):
        sidecar_map[i] = chunk
        source_tag = f"{chunk.source_file} p.{chunk.page_number}"
        blocks.append(
            f'<source id="{i}" file="{source_tag}" type="{chunk.item_type}">\n'
            f'{chunk.text}\n'
            f'</source>'
        )

    context_str = "\n\n".join(blocks)
    return context_str, sidecar_map


# ==============================================================================
# Fine-Grained BBox Resolution (Phase 2)
# ==============================================================================

def _extract_bbox(chunk: RetrievedChunk) -> Optional[BoundingBox]:
    """
    Coarse bbox from chunk metadata (entire table/chart region).
    Used as fallback when fine-grained value_bboxes don't match.
    """
    meta = chunk.metadata
    if all(k in meta for k in ("bbox_x", "bbox_y", "bbox_width", "bbox_height")):
        return BoundingBox(
            page_number=chunk.page_number,
            x=float(meta["bbox_x"]),
            y=float(meta["bbox_y"]),
            width=float(meta["bbox_width"]),
            height=float(meta["bbox_height"]),
        )
    return None


def _bbox_from_coords(page_number: int, coords_list: list) -> Optional[BoundingBox]:
    """Creates a BoundingBox from the first value_bboxes entry."""
    if not coords_list or not coords_list[0]:
        return None
    c = coords_list[0]
    if len(c) < 4:
        return None
    return BoundingBox(
        page_number=page_number,
        x=c[0], y=c[1], width=c[2], height=c[3],
    )


def _resolve_fine_bbox(
    chunk: RetrievedChunk,
    citing_texts: List[str],
) -> Optional[BoundingBox]:
    """
    Resolves the most specific bounding box for a citation by matching
    values from the answer text against the chunk's value_bboxes dict.

    Matching tiers (first match wins):
    1. Dollar amounts: "$72,764" → try raw key, then normalized float key
    2. Percentages: "25%" → try raw key
    3. Plain numbers: "72,764" → try raw, stripped, and float forms
    4. Series label: chunk metadata series_label → try as key
    5. Fallback: coarse region bbox

    Returns BoundingBox or None.
    """
    vb = chunk.metadata.get("value_bboxes")
    if not vb or not isinstance(vb, dict):
        return _extract_bbox(chunk)

    combined = " ".join(citing_texts)
    page = chunk.page_number

    # Tier 1: Dollar amounts ($72,764 or $16.8B)
    for m in re.finditer(r'\$([\d,]+(?:\.\d+)?)', combined):
        raw_dollar = "$" + m.group(1)           # "$72,764"
        comma_num = m.group(1)                   # "72,764" (no $, with commas)
        plain = m.group(1).replace(',', '')      # "72764"

        # Try: "$72,764" → "72,764" → "72764" → "72764.0"
        for candidate in [raw_dollar, comma_num, plain]:
            if candidate in vb:
                bb = _bbox_from_coords(page, vb[candidate])
                if bb:
                    return bb

        # Normalized float key (chart data points use "72764.0")
        try:
            float_key = str(float(plain))
            if float_key in vb:
                bb = _bbox_from_coords(page, vb[float_key])
                if bb:
                    return bb
        except ValueError:
            pass

    # Tier 2: Percentages (25%, 13%)
    for m in re.finditer(r'([\d,]+(?:\.\d+)?)\s*%', combined):
        pct_raw = m.group(0).strip()            # "25%"
        pct_num = m.group(1).replace(',', '')    # "25"

        for candidate in [pct_raw, pct_num + "%", pct_num]:
            if candidate in vb:
                bb = _bbox_from_coords(page, vb[candidate])
                if bb:
                    return bb

    # Tier 3: Plain numbers (72,764 or 18953)
    for m in re.finditer(r'(?<!\$)([\d,]+(?:\.\d+)?)', combined):
        raw = m.group(1)                         # "72,764"
        plain = raw.replace(',', '')             # "72764"

        for candidate in [raw, plain]:
            if candidate in vb:
                bb = _bbox_from_coords(page, vb[candidate])
                if bb:
                    return bb

        # Also try parenthesized negative form: (12,012)
        paren_form = f"({raw})"
        if paren_form in vb:
            bb = _bbox_from_coords(page, vb[paren_form])
            if bb:
                return bb

        try:
            float_key = str(float(plain))
            if float_key in vb:
                bb = _bbox_from_coords(page, vb[float_key])
                if bb:
                    return bb
        except ValueError:
            pass

    # Tier 4: Series label match (for chart/series chunks)
    series_label = chunk.metadata.get("series_label")
    if series_label and series_label in vb:
        bb = _bbox_from_coords(page, vb[series_label])
        if bb:
            return bb

    # Tier 5: Sentence text match (for narrative chunks with value_bboxes)
    # Match citing text against sentence-keyed entries via word overlap.
    # Handles paraphrasing: "FCF is defined as..." vs "We define FCF as..."
    for seg_text in citing_texts:
        seg_words = set(seg_text.lower().split())
        if len(seg_words) < 5:
            continue  # Too short to reliably match
        best_key, best_score = None, 0.0
        for key in vb:
            if len(key) < 20:
                continue  # Skip numeric / label keys
            key_words = set(key.lower().split())
            union = len(seg_words | key_words)
            if union == 0:
                continue
            overlap = len(seg_words & key_words) / union
            if overlap > best_score:
                best_score = overlap
                best_key = key
        if best_key and best_score > 0.35:
            bb = _bbox_from_coords(page, vb[best_key])
            if bb:
                logger.debug(
                    f"Sentence match (overlap={best_score:.2f}): "
                    f"'{seg_text[:50]}' → '{best_key[:50]}'"
                )
                return bb

    # Tier 6: Fallback to coarse region bbox
    return _extract_bbox(chunk)


# ==============================================================================
# Citation Assembly
# ==============================================================================

def assemble_cited_answer(
    segments: List[dict],
    sidecar_map: Dict[int, RetrievedChunk],
) -> Tuple[str, List[Citation]]:
    """
    Fact-based citation assembly from structured LLM output.

    Each segment = one factual claim. Dedup happens WITHIN a segment
    (multiple sources for the same claim → keep first), never across
    segments (different claims keep their own badges even if same page).

    Each badge's bbox resolves from only the segment text where it appears,
    so "$24,837" highlights the table cell while "FCF is defined as..."
    highlights the definition text — even if both come from the same page.

    Takes segments [{text, source_ids}, ...] and:
    1. Per-segment dedup: keep first valid source per claim
    2. Assigns badge numbers by order of first appearance (no global dedup)
    3. Tracks citing texts per badge (only from that badge's own segments)
    4. Builds answer text with inline [N] markers
    5. Creates Citation objects with fact-scoped fine-grained bboxes

    Returns:
        answer: Final answer string with [N] markers
        citations: List[Citation] where citations[0] = badge [1], etc.
    """
    # 1. Per-segment dedup: within each segment, keep first valid source.
    #    All sources in a segment support the same claim → one badge suffices.
    resolved_segments: List[Tuple[str, Optional[int]]] = []  # (text, kept_source_id)
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        valid_sids = [s for s in seg.get("source_ids", []) if s in sidecar_map]
        resolved_segments.append((text, valid_sids[0] if valid_sids else None))

    # 2. Assign unique badge number per cited segment (no cross-segment dedup).
    #    Each badge resolves its own fine-grained bbox from its own citing text,
    #    so "$16,836" highlights one cell while "$13,454" highlights another —
    #    even when both come from the same source chunk.
    badge_counter = 0
    segment_badges: List[Tuple[str, Optional[int]]] = []  # (text, badge_num | None)
    badge_info: List[Tuple[int, str]] = []  # (source_id, citing_text) per badge
    for text, sid in resolved_segments:
        if sid is not None:
            badge_counter += 1
            segment_badges.append((text, badge_counter))
            badge_info.append((sid, text))
        else:
            segment_badges.append((text, None))

    # 3. Build answer text with [N] markers
    parts = []
    for text, badge in segment_badges:
        if badge is not None:
            parts.append(f"{text} [{badge}]")
        else:
            parts.append(text)
    answer = " ".join(parts)

    # 4. Build Citation objects — each badge gets its own bbox resolved from
    #    only its own segment text for maximum highlighting precision.
    citations = []
    for source_id, citing_text in badge_info:
        chunk = sidecar_map[source_id]
        bbox = _resolve_fine_bbox(chunk, [citing_text])

        citations.append(Citation(
            source_file=chunk.source_file,
            page_number=chunk.page_number,
            blurb=chunk.text[:200],
            bbox=bbox,
            doc_hash=chunk.doc_hash,
        ))

    logger.info(
        f"Assembled answer: {len(segments)} segments → "
        f"{len(citations)} badges ({badge_counter} cited segments)"
    )
    return answer, citations
