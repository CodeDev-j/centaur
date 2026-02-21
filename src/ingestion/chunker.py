"""
Document Chunker: Flattening the Deal Stream for Indexing.

Converts a UnifiedDocument into a list of IndexableChunks ready for:
1. Vector indexing (Cohere embed-v4 dense + Qdrant BM25 sparse)
2. Each chunk carries rich metadata for filtered search and citation resolution.

Chunking Strategy per Item Type:
- HeaderItem:         Embed text directly (short, structural)
- NarrativeItem:      Embed text_content (qualitative arguments)
- FinancialTableItem:  Embed markdown summary of html_repr (avoid raw HTML noise)
- VisualItem:         One "summary chunk" + one chunk per MetricSeries
- ChartTableItem:     Embed title + visual_metrics labels
"""

import logging
import re
from typing import List, Optional
from uuid import uuid4

from langsmith import traceable

from src.schemas.deal_stream import (
    UnifiedDocument,
    HeaderItem,
    NarrativeItem,
    FinancialTableItem,
    VisualItem,
    ChartTableItem,
)
from src.schemas.vision_output import MetricSeries
from src.storage.vector_driver import IndexableChunk

logger = logging.getLogger(__name__)


def _html_table_to_markdown(html: str) -> str:
    """
    Lightweight HTML table → markdown summary.
    Strips tags to produce a text representation suitable for embedding.
    Not a full converter — optimized for financial table HTML from Docling.
    """
    # Strip all tags except td/th/tr for structure
    text = re.sub(r'<br\s*/?>', ' ', html)
    text = re.sub(r'</?(?:table|thead|tbody|tfoot)[^>]*>', '', text)

    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', text, re.DOTALL)
    lines = []
    for row in rows:
        cells = re.findall(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', row, re.DOTALL)
        cleaned = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        if any(cleaned):
            lines.append(' | '.join(cleaned))

    return '\n'.join(lines) if lines else re.sub(r'<[^>]+>', ' ', html).strip()


def _series_to_text(series: MetricSeries) -> str:
    """
    Converts a MetricSeries to an embedding-friendly text representation.
    Example: "Series: Revenue. Data: 2022=$12.4M, 2023=$13.1M, 2024=$14.0M"
    """
    parts = [f"Series: {series.series_label}."]

    if series.accounting_basis:
        parts.append(f"Basis: {series.accounting_basis}.")
    if series.data_provenance:
        parts.append(f"Provenance: {series.data_provenance}.")
    if series.series_nature != "level":
        parts.append(f"Nature: {series.series_nature}.")

    if series.data_points:
        dp_strs = []
        for dp in series.data_points:
            if dp.numeric_value is not None:
                val = f"{dp.numeric_value}"
                if dp.currency and dp.currency != "None":
                    val = f"{dp.currency} {val}"
                if dp.magnitude and dp.magnitude != "None":
                    val = f"{val}{dp.magnitude}"
                dp_strs.append(f"{dp.label}={val}")
            else:
                dp_strs.append(f"{dp.label}=N/A")
        parts.append(f"Data: {', '.join(dp_strs)}")

    return ' '.join(parts)


def _inject_bbox(metadata: dict, source) -> None:
    """Copies normalized bbox from SourceRef into chunk metadata (if present)."""
    if source.bbox_x is not None:
        metadata["bbox_x"] = source.bbox_x
        metadata["bbox_y"] = source.bbox_y
        metadata["bbox_width"] = source.bbox_width
        metadata["bbox_height"] = source.bbox_height


def _inject_value_bboxes(metadata: dict, item) -> None:
    """Copies value_bboxes from item into chunk metadata (if present)."""
    if hasattr(item, 'value_bboxes') and item.value_bboxes:
        metadata["value_bboxes"] = item.value_bboxes


def _make_chunk(
    doc: UnifiedDocument,
    item_id: str,
    item_type: str,
    text: str,
    page_number: int,
    metadata: dict,
) -> IndexableChunk:
    """Factory for IndexableChunk with auto-generated chunk_id."""
    return IndexableChunk(
        chunk_id=str(uuid4()),
        doc_id=doc.doc_id,
        doc_hash=doc.items[0].source.file_hash if doc.items else "",
        source_file=doc.filename,
        page_number=page_number,
        item_id=item_id,
        item_type=item_type,
        text=text,
        metadata=metadata,
    )


def _chunk_header(doc: UnifiedDocument, item: HeaderItem) -> List[IndexableChunk]:
    """HeaderItem → single chunk of header text."""
    text = item.text.strip()
    if not text:
        return []
    metadata = {"level": item.level}
    _inject_bbox(metadata, item.source)
    return [_make_chunk(
        doc=doc,
        item_id=item.id,
        item_type="header",
        text=text,
        page_number=item.source.page_number,
        metadata=metadata,
    )]


def _chunk_narrative(doc: UnifiedDocument, item: NarrativeItem) -> List[IndexableChunk]:
    """NarrativeItem → single chunk of text_content."""
    text = item.text_content.strip()
    if not text:
        return []
    metadata = {
        "is_strategic_claim": item.is_strategic_claim,
        "sentiment": item.sentiment,
    }
    if item.category:
        metadata["category"] = item.category
    _inject_bbox(metadata, item.source)
    _inject_value_bboxes(metadata, item)
    return [_make_chunk(
        doc=doc,
        item_id=item.id,
        item_type="narrative",
        text=text,
        page_number=item.source.page_number,
        metadata=metadata,
    )]


def _chunk_financial_table(doc: UnifiedDocument, item: FinancialTableItem) -> List[IndexableChunk]:
    """FinancialTableItem → single chunk with markdown table summary."""
    md = _html_table_to_markdown(item.html_repr)
    if not md.strip():
        return []
    metadata = {
        "accounting_basis": item.accounting_basis,
    }
    if item.periodicity:
        metadata["periodicity"] = item.periodicity
    if item.currency:
        metadata["currency"] = item.currency
    _inject_bbox(metadata, item.source)
    _inject_value_bboxes(metadata, item)
    return [_make_chunk(
        doc=doc,
        item_id=item.id,
        item_type="financial_table",
        text=md,
        page_number=item.source.page_number,
        metadata=metadata,
    )]


def _chunk_visual(doc: UnifiedDocument, item: VisualItem) -> List[IndexableChunk]:
    """
    VisualItem → one summary chunk + one chunk per MetricSeries.
    Summary chunk captures title + summary + series labels for broad retrieval.
    Per-series chunks capture granular data for precise numeric lookup.
    """
    chunks = []
    page = item.source.page_number
    archetype = item.archetype.value if hasattr(item.archetype, 'value') else str(item.archetype)

    # 1. Summary chunk
    series_labels = [s.series_label for s in item.metrics]
    summary_parts = [item.title]
    if item.summary:
        summary_parts.append(item.summary)
    if series_labels:
        summary_parts.append(f"Series: {', '.join(series_labels)}")

    summary_text = '. '.join(summary_parts)
    summary_meta = {
        "archetype": archetype,
        "has_metrics": bool(item.metrics),
        "chunk_role": "summary",
    }
    _inject_bbox(summary_meta, item.source)
    _inject_value_bboxes(summary_meta, item)
    chunks.append(_make_chunk(
        doc=doc,
        item_id=item.id,
        item_type="visual",
        text=summary_text,
        page_number=page,
        metadata=summary_meta,
    ))

    # 2. Per-series chunks
    for series in item.metrics:
        series_text = _series_to_text(series)
        meta = {
            "archetype": archetype,
            "chunk_role": "series",
            "series_label": series.series_label,
            "series_nature": series.series_nature,
        }
        if series.accounting_basis:
            meta["accounting_basis"] = series.accounting_basis
        if series.data_provenance:
            meta["data_provenance"] = series.data_provenance
        if series.periodicity:
            meta["periodicity"] = series.periodicity
        if series.source_region_id is not None:
            meta["source_region_id"] = series.source_region_id
        _inject_bbox(meta, item.source)
        # Per-series: filter value_bboxes to only this series' label + numeric values
        if hasattr(item, 'value_bboxes') and item.value_bboxes:
            series_keys = {series.series_label}
            for dp in series.data_points:
                if dp.numeric_value is not None:
                    series_keys.add(str(float(dp.numeric_value)))
            filtered_vb = {k: v for k, v in item.value_bboxes.items() if k in series_keys}
            if filtered_vb:
                meta["value_bboxes"] = filtered_vb
        else:
            _inject_value_bboxes(meta, item)

        chunks.append(_make_chunk(
            doc=doc,
            item_id=item.id,
            item_type="visual",
            text=series_text,
            page_number=page,
            metadata=meta,
        ))

    return chunks


def _chunk_chart_table(doc: UnifiedDocument, item: ChartTableItem) -> List[IndexableChunk]:
    """ChartTableItem → single chunk with title + visual_metrics labels."""
    parts = [item.title]
    for series in item.visual_metrics:
        parts.append(f"Series: {series.series_label}")
        if series.data_points:
            dp_labels = [dp.label for dp in series.data_points[:10]]
            parts.append(f"Labels: {', '.join(dp_labels)}")

    text = '. '.join(parts)
    archetype = item.archetype.value if hasattr(item.archetype, 'value') else str(item.archetype)

    meta = {
        "archetype": archetype,
        "has_metrics": bool(item.visual_metrics),
    }
    _inject_bbox(meta, item.source)
    _inject_value_bboxes(meta, item)

    return [_make_chunk(
        doc=doc,
        item_id=item.id,
        item_type="chart_table",
        text=text,
        page_number=item.source.page_number,
        metadata=meta,
    )]


# ==============================================================================
# PUBLIC API
# ==============================================================================

@traceable(name="Flatten UnifiedDocument", run_type="tool")
def flatten_document(doc: UnifiedDocument) -> List[IndexableChunk]:
    """
    Converts a UnifiedDocument into a flat list of IndexableChunks.
    Each chunk is a self-contained text passage with rich metadata
    for hybrid search and citation resolution.
    """
    chunks: List[IndexableChunk] = []

    for item in doc.items:
        if isinstance(item, HeaderItem):
            chunks.extend(_chunk_header(doc, item))
        elif isinstance(item, NarrativeItem):
            chunks.extend(_chunk_narrative(doc, item))
        elif isinstance(item, FinancialTableItem):
            chunks.extend(_chunk_financial_table(doc, item))
        elif isinstance(item, VisualItem):
            chunks.extend(_chunk_visual(doc, item))
        elif isinstance(item, ChartTableItem):
            chunks.extend(_chunk_chart_table(doc, item))
        else:
            logger.warning(f"Unknown item type: {type(item).__name__}, skipping.")

    logger.info(
        f"Flattened {doc.filename}: {len(doc.items)} items → {len(chunks)} chunks"
    )
    return chunks
