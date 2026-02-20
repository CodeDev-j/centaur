"""
PDF Ingestion Module for the Chiron Financial Forensic Pipeline.

==============================================================================
ARCHITECTURE OVERVIEW (The "Cognitive Contract")
==============================================================================
This module implements a hybrid "Split-Brain" parsing strategy to maximize data
fidelity from complex financial documents (CIMs, Lender Presentations):

1. THE LOGICAL BRAIN (Docling):
   - Handles the rigorous structure.
   - Extracts complex financial tables with row-level hierarchy.
   - Captures Document Outline (Headers) and Narrative Text (Body).

2. THE VISUAL BRAIN (VLM):
   - Handles the "unstructured reality".
   - Performs forensic analysis on Charts, Waterfalls, and Football Fields.
   - Uses a "Glance (Layout) -> Read (Extractor)" two-pass architecture.

3. THE CONFLICT RESOLUTION LAYER:
   - "Split-Brain" logic resolves overlaps (e.g., if Docling sees a table but
     the VLM identifies it as a 'Valuation Field' chart, the VLM wins).
   - Coordinates normalization ensures pixel-perfect citation grounding.

4. SAFETY & RESILIENCE:
   - Concurrency control (Semaphores) prevents API rate limits.
   - "Quarantine Pattern" prevents single-item failures from crashing the job.
   - Strict resource management (File handles) prevents memory leaks.
   - Fault Tolerance: Individual page failures are isolated; they do not kill the batch.
==============================================================================
"""

import asyncio
import base64
import hashlib
import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args
from uuid import uuid4

# --- Third-Party Imports ---
import fitz  # PyMuPDF: Fast PDF rendering and low-level access
import numpy as np
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    TableItem,
    SectionHeaderItem,
    TextItem,
    ListItem
)
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langsmith import traceable
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from pydantic import ValidationError

# --- Internal Core Components ---
from src.storage.blob_driver import BlobDriver
from src.tools.layout_analyzer import layout_analyzer
from src.tools.visual_extractor import visual_extractor
from src.utils.color_math import ColorResolver, LegendBinding
from src.utils.geometry import SpatialIndex
from src.utils.text_forensics import extract_table_metadata

# --- Schema Definitions (The "Truth") ---
from src.schemas.layout_output import VisualArchetype
from src.schemas.deal_stream import (
    UnifiedDocument,
    DocItem,
    FinancialTableItem,
    VisualItem,
    NarrativeItem,
    ChartTableItem,
    HeaderItem,
    SourceRef,
    RoutingSignals,
    FinancialTableRow
)
# Single Source of Truth for Enums
from src.schemas.enums import PeriodicityType

logger = logging.getLogger(__name__)

# ==============================================================================
# ⚙️ SYSTEM CONFIGURATION
# ==============================================================================

# Thread pool for CPU-bound OCR tasks to avoid blocking the async event loop.
# We limit to 2 workers to prevent CPU starvation of the main thread.
_OCR_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# Semaphore to control VLM concurrency.
# Processing 100 pages strictly in parallel would hit API rate limits immediately.
# 5 is a safe "sweet spot" for throughput vs. stability.
MAX_CONCURRENT_PAGES = 5

# Hard timeout for per-page processing (OCR + VLM + Layout).
# Prevents a single corrupt page from hanging the entire pipeline forever.
PAGE_TIMEOUT_SECONDS = 240

# Rendering settings for VLM input.
# 3.0 zoom = ~216 DPI. This is the "Goldilocks" zone: high enough to read
# tiny footnote text/axis labels, but low enough to keep token counts manageable.
RENDER_ZOOM = 3.0

# Heuristics for linking text labels to color boxes in legends.
LEGEND_SEARCH_RADIUS = (-30, -5, 30, 5)  # (x0, y0, x1, y1) expansion box
LEGEND_DISTANCE_THRESHOLD = 25  # Pixels

# ==============================================================================
# 📐 BBOX CONVERSION HELPER
# ==============================================================================

def _chart_bbox_to_normalized(bbox: List[int]) -> dict:
    """
    Converts ChartRegion's 0-1000 integer grid bbox to normalized 0-1 kwargs
    suitable for SourceRef.

    ChartRegion format: [ymin, xmin, ymax, xmax] (0-1000)
    SourceRef format:   bbox_x, bbox_y, bbox_width, bbox_height (0.0-1.0)
    """
    ymin, xmin, ymax, xmax = bbox
    return {
        "bbox_x": xmin / 1000.0,
        "bbox_y": ymin / 1000.0,
        "bbox_width": (xmax - xmin) / 1000.0,
        "bbox_height": (ymax - ymin) / 1000.0,
    }


def _docling_bbox_to_normalized(item, pdf_doc: fitz.Document) -> dict:
    """
    Extracts normalized 0-1 bbox from a Docling item's provenance data.
    Returns empty dict if provenance is unavailable.

    Docling BoundingBox: l, t, r, b in PDF points (default TOPLEFT origin).
    Normalizes against fitz page dimensions to produce SourceRef kwargs.
    """
    if not item.prov:
        return {}

    prov = item.prov[0]
    bbox = prov.bbox
    if not bbox:
        return {}

    page_no = prov.page_no
    if page_no < 1 or page_no > len(pdf_doc):
        return {}

    page = pdf_doc[page_no - 1]
    pw, ph = page.rect.width, page.rect.height
    if pw == 0 or ph == 0:
        return {}

    # Docling defaults to TOPLEFT origin. Handle BOTTOMLEFT if present.
    if hasattr(bbox, "coord_origin") and bbox.coord_origin.value == "BOTTOMLEFT":
        bbox = bbox.to_top_left_origin(ph)

    return {
        "bbox_x": bbox.l / pw,
        "bbox_y": bbox.t / ph,
        "bbox_width": bbox.width / pw,
        "bbox_height": bbox.height / ph,
    }


# ==============================================================================
# 🔢 VALUE-TO-BBOX SPATIAL BINDING
# ==============================================================================

def _normalize_financial_str(s: str) -> Optional[float]:
    """Strip $, commas, %, superscripts from a string and return as float."""
    clean = re.sub(r'[^\d.\-]', '', s)
    try:
        return float(clean) if clean and clean != '.' and clean != '-' else None
    except ValueError:
        return None


def _word_inside_region(word: Dict, rx: float, ry: float, rw: float, rh: float) -> bool:
    """Check if a word record's center falls inside a normalized 0-1 region."""
    cx = word["bbox_x"] + word["bbox_w"] / 2
    cy = word["bbox_y"] + word["bbox_h"] / 2
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


def _build_value_bboxes(
    metrics: List[Any],
    word_bboxes: List[Dict],
    chart_regions: List[Any],
) -> Optional[Dict[str, List[List[float]]]]:
    """
    Matches VLM-extracted MetricSeries values against OCR word bboxes
    to build a per-value spatial dictionary.

    Keys are normalized float strings for numbers (e.g., "8090.0") and
    verbatim series_label strings for labels. Values are lists of
    [bbox_x, bbox_y, bbox_w, bbox_h] in 0-1 normalized coordinates.
    Multiple bboxes per key handle duplicate values on the page.

    OCR words are filtered to those inside ChartRegion bboxes to prevent
    false matches from footnotes or adjacent charts.
    """
    if not metrics or not word_bboxes:
        return None

    result: Dict[str, List[List[float]]] = {}

    # Build per-region filtered word lists
    region_words: List[List[Dict]] = []
    for chart in chart_regions:
        ymin, xmin, ymax, xmax = chart.bbox
        rx, ry = xmin / 1000.0, ymin / 1000.0
        rw, rh = (xmax - xmin) / 1000.0, (ymax - ymin) / 1000.0
        filtered = [w for w in word_bboxes if _word_inside_region(w, rx, ry, rw, rh)]
        region_words.append(filtered)

    # Fall back to all words if no chart regions
    all_region_words = []
    for rw_list in region_words:
        all_region_words.extend(rw_list)
    search_words = all_region_words if all_region_words else word_bboxes

    for series in metrics:
        # A. Match series_label (multi-word AABB union)
        label = series.series_label
        label_tokens = label.split()
        if label_tokens:
            matched_bboxes = []
            remaining = list(search_words)
            for token in label_tokens:
                token_lower = token.lower().rstrip('*†')
                for w in remaining:
                    if w["text"].lower().rstrip('*†') == token_lower:
                        matched_bboxes.append(w)
                        remaining.remove(w)
                        break

            if matched_bboxes:
                # AABB union of all matched words
                x0 = min(w["bbox_x"] for w in matched_bboxes)
                y0 = min(w["bbox_y"] for w in matched_bboxes)
                x1 = max(w["bbox_x"] + w["bbox_w"] for w in matched_bboxes)
                y1 = max(w["bbox_y"] + w["bbox_h"] for w in matched_bboxes)
                result.setdefault(label, []).append([x0, y0, x1 - x0, y1 - y0])

        # B. Match data point numeric values
        for dp in series.data_points:
            if dp.numeric_value is None:
                continue

            target_float = float(dp.numeric_value)
            key = str(target_float)

            # Try matching against OCR words in relevant regions
            for w in search_words:
                ocr_float = _normalize_financial_str(w["text"])
                if ocr_float is not None and abs(ocr_float - target_float) < 0.01:
                    bbox = [w["bbox_x"], w["bbox_y"], w["bbox_w"], w["bbox_h"]]
                    # Avoid duplicate bboxes for the same key
                    existing = result.get(key, [])
                    if not any(
                        abs(b[0] - bbox[0]) < 0.005 and abs(b[1] - bbox[1]) < 0.005
                        for b in existing
                    ):
                        result.setdefault(key, []).append(bbox)

    return result if result else None


def _build_sentence_bboxes(
    text: str,
    page: fitz.Page,
    item_bbox: dict,
) -> Optional[Dict[str, List[List[float]]]]:
    """
    Builds per-sentence, per-line bounding boxes for narrative text.

    Splits text into sentences using nltk (handles "Inc.", "U.S.", etc.),
    matches each sentence's words against PyMuPDF's digital text layer,
    and groups matched words by visual line (block_no, line_no) to produce
    per-line AABB rectangles. This prevents multi-line sentences from
    creating a single oversized highlight box.

    Args:
        text: The narrative text content.
        page: The fitz.Page for word extraction.
        item_bbox: Normalized 0-1 bbox dict (bbox_x, bbox_y, etc.) to filter words.

    Returns:
        Dict mapping sentence text to list of [bbox_x, bbox_y, bbox_w, bbox_h].
        None if no sentences could be matched.
    """
    from nltk.tokenize import sent_tokenize

    if not text or len(text) < 10:
        return None

    pw, ph = page.rect.width, page.rect.height
    if pw == 0 or ph == 0:
        return None

    # Get all digital text words with line grouping info
    raw_words = page.get_text("words")
    page_words = []
    for w in raw_words:
        rec = {
            "text": w[4],
            "bbox_x": w[0] / pw, "bbox_y": w[1] / ph,
            "bbox_w": (w[2] - w[0]) / pw, "bbox_h": (w[3] - w[1]) / ph,
            "block_no": int(w[5]), "line_no": int(w[6]),
        }
        # Filter to words within the item's bbox region (if available)
        if item_bbox and "bbox_x" in item_bbox:
            cx = rec["bbox_x"] + rec["bbox_w"] / 2
            cy = rec["bbox_y"] + rec["bbox_h"] / 2
            bx = item_bbox["bbox_x"]
            by = item_bbox["bbox_y"]
            bw = item_bbox.get("bbox_width", 1.0)
            bh = item_bbox.get("bbox_height", 1.0)
            if not (bx <= cx <= bx + bw and by <= cy <= by + bh):
                continue
        page_words.append(rec)

    if not page_words:
        return None

    # Split into sentences
    sentences = sent_tokenize(text)
    result: Dict[str, List[List[float]]] = {}

    # Track consumed word index for greedy sequential matching
    word_cursor = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sent_tokens = sentence.split()
        matched_words = []

        # Greedy sequential match against page words
        local_cursor = word_cursor
        for token in sent_tokens:
            token_clean = token.lower().rstrip('.,;:!?')
            while local_cursor < len(page_words):
                pw_text = page_words[local_cursor]["text"].lower().rstrip('.,;:!?')
                if pw_text == token_clean or pw_text.startswith(token_clean):
                    matched_words.append(page_words[local_cursor])
                    local_cursor += 1
                    break
                local_cursor += 1

        if not matched_words:
            continue

        # Advance cursor past matched words for next sentence
        word_cursor = local_cursor

        # Group by visual line (block_no, line_no) and compute per-line AABB
        line_groups: Dict[Tuple[int, int], List[Dict]] = {}
        for mw in matched_words:
            line_key = (mw["block_no"], mw["line_no"])
            line_groups.setdefault(line_key, []).append(mw)

        line_bboxes = []
        for words_in_line in line_groups.values():
            x0 = min(w["bbox_x"] for w in words_in_line)
            y0 = min(w["bbox_y"] for w in words_in_line)
            x1 = max(w["bbox_x"] + w["bbox_w"] for w in words_in_line)
            y1 = max(w["bbox_y"] + w["bbox_h"] for w in words_in_line)
            line_bboxes.append([x0, y0, x1 - x0, y1 - y0])

        if line_bboxes:
            result.setdefault(sentence, []).extend(line_bboxes)

    return result if result else None


# ==============================================================================
# 📄 PDF PARSER
# ==============================================================================
class PDFParser:
    """
    Orchestrates the forensic ingestion of PDF documents.
    Acts as the central controller for Docling (Logic) and VLM (Vision).
    """

    def __init__(self):
        # 1. Initialize RapidOCR for visual text fallback ("The Safety Net")
        # Used when digital text is missing/corrupt (e.g., "Zombie Charts")
        self.ocr_engine = RapidOCR()

        # 2. Configure Docling for rigorous Table Extraction
        # We enable internal OCR specifically for table cell content to handle
        # scanned PDFs seamlessly.
        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = True
        pipeline_opts.do_table_structure = True
        
        # Optimized Compute: Disable internal image generation (we use Fitz later)
        pipeline_opts.generate_page_images = False
        pipeline_opts.generate_picture_images = False
        
        # Unified Constant: Use RENDER_ZOOM to prevent drift
        pipeline_opts.images_scale = RENDER_ZOOM
        
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
        )

    # ==========================================================================
    # 📐 VECTOR & SPATIAL HELPERS
    # ==========================================================================

    def _is_valid_color(self, p: Tuple[int, int, int]) -> bool:
        """
        Filters out background/text colors (White, Black, Gray) to focus 
        strictly on data series colors (Bars, Lines, Pies).
        """
        r, g, b = p
        # Filter White-ish backgrounds
        if r > 240 and g > 240 and b > 240:
            return False
        # Filter Black-ish text
        if r < 20 and g < 20 and b < 20:
            return False
        # Filter Gray-ish gridlines (Low Saturation)
        if abs(r - g) < 15 and abs(g - b) < 15:
            return False
        return True

    def _rect_distance(self, r1: fitz.Rect, r2: fitz.Rect) -> float:
        """
        Calculates the Euclidean distance between two rectangles.
        Returns 0 if they intersect. Used for Legend-to-Swatch linking.
        """
        x_dist = max(0, r1.x0 - r2.x1, r2.x0 - r1.x1)
        y_dist = max(0, r1.y0 - r2.y1, r2.y0 - r1.y1)
        return (x_dist**2 + y_dist**2)**0.5

    def _resolve_split_brain(
        self,
        docling_items: List[DocItem],
        layout_charts: List[Any],
        page_height: int,
        page_width: int
    ) -> List[DocItem]:
        """
        [CRITICAL] Resolves conflicts between Docling (Logic) and VLM (Vision).
        
        The "Split-Brain" Problem:
        Docling sees a grid of numbers and calls it a Table.
        The VLM sees the same grid, recognizes it's a "Valuation Football Field",
        and extracts it as a Chart.
        
        Resolution Strategy:
        1. Identify "VLM Zones": Regions claimed by high-priority visual archetypes.
        2. Check Intersection: If a Docling table falls inside a VLM zone,
           we discard the Docling table (trusting the VLM's semantic understanding).
        """
        if not layout_charts or not docling_items:
            return docling_items

        filtered_items = []

        # 1. Map VLM Claims (High-Priority Visuals)
        vlm_zones = []
        for chart in layout_charts:
            # We only prioritize VLM for complex artifacts where visual context matters.
            if chart.visual_type in [
                VisualArchetype.WATERFALL,
                VisualArchetype.VALUATION_FIELD,
                VisualArchetype.MARKET_MAP
            ]:
                # Coordinate Normalization Logic
                # The Layout Analyzer returns 1000x1000 coordinates.
                # We normalize them to 0.0-1.0 float space for comparison.
                ymin, xmin, ymax, xmax = chart.bbox
                vlm_zones.append(
                    fitz.Rect(xmin / 1000, ymin / 1000, xmax / 1000, ymax / 1000)
                )

        # 2. Filter Docling Items
        for item in docling_items:
            # Non-tables (headers, text) are always kept
            if not isinstance(item, FinancialTableItem):
                filtered_items.append(item)
                continue

            # [TODO] BBox Intersection Check
            # Currently, strict intersection is disabled until we can reliably
            # map Docling's provenance coordinates to our normalized space.
            # For now, we adopt a "Data Safety" approach: Duplicate is better than Lost.
            # We append the table even if it might overlap.
            filtered_items.append(item)

        return filtered_items

    # ==========================================================================
    # 👁️ TEXT EXTRACTION LAYERS
    # ==========================================================================

    # Async def supports non-blocking OCR
    async def _extract_text_spatial(
        self,
        page: fitz.Page,
        img_np: np.ndarray,
        zoom_factor: float
    ) -> Tuple[str, List[Dict]]:
        """
        Generates a 'Spatial Grid' representation of the page text.
        Strategy:
        1. Extract Digital Text (Ground Truth) from the PDF layer.
        2. Run OCR (Visual Fill) to catch 'Zombie Text' embedded in images.
        3. Merge results, discarding OCR duplicates that overlap digital text.

        Returns:
            spatial_string: Text grid for VLM context (unchanged format).
            word_records: Per-word bboxes normalized 0-1 for value_bboxes binding.
        """
        spatial_lines = []
        word_records: List[Dict] = []
        pw, ph = page.rect.width, page.rect.height

        # --- LAYER 1: DIGITAL TEXT (Ground Truth) ---
        # Returns: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        digital_words = page.get_text("words")
        digital_rects = []  # For collision detection

        # Capture all digital text first
        for w in digital_words:
            # Create unscaled rect for collision logic (PDF Coordinate Space)
            r = fitz.Rect(w[0], w[1], w[2], w[3])
            digital_rects.append(r)

            # Formatted Output (Scaled to Image Space for LLM context)
            x_img = int(w[0] * zoom_factor)
            y_img = int(w[1] * zoom_factor)
            spatial_lines.append(f"[{y_img}, {x_img}] {w[4]}")

            # Structured word record (normalized 0-1)
            if pw > 0 and ph > 0:
                word_records.append({
                    "text": w[4],
                    "bbox_x": w[0] / pw, "bbox_y": w[1] / ph,
                    "bbox_w": (w[2] - w[0]) / pw, "bbox_h": (w[3] - w[1]) / ph,
                    "block_no": int(w[5]), "line_no": int(w[6]),
                })

        # --- LAYER 2: OCR FILL (The Safety Net) ---
        # We run OCR to catch text embedded in images (Zombie Charts)
        try:
            # Explicitly get the running loop
            loop = asyncio.get_running_loop()
            
            # Offload blocking OCR to executor
            ocr_results, _ = await loop.run_in_executor(
                _OCR_EXECUTOR, self.ocr_engine, img_np
            )
            
            if ocr_results:
                for res in ocr_results:
                    box, text, score = res
                    # RapidOCR Box: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    x1_img, y1_img = box[0]
                    x2_img, y2_img = box[2]

                    # Map Image Coordinates BACK to PDF Coordinates for check
                    x1_pdf = x1_img / zoom_factor
                    y1_pdf = y1_img / zoom_factor
                    x2_pdf = x2_img / zoom_factor
                    y2_pdf = y2_img / zoom_factor

                    ocr_rect = fitz.Rect(x1_pdf, y1_pdf, x2_pdf, y2_pdf)

                    # COLLISION CHECK: Does OCR box overlap with Digital Text?
                    # If yes, we assume Digital Text is better and discard OCR.
                    is_duplicate = False
                    for d_rect in digital_rects:
                        # Check for significant overlap (intersection area)
                        if ocr_rect.intersects(d_rect):
                            intersect = ocr_rect & d_rect
                            # If overlap > 30% of smaller box, discard OCR
                            overlap_ratio = intersect.get_area() / min(
                                ocr_rect.get_area(), d_rect.get_area()
                            )
                            if overlap_ratio > 0.3:
                                is_duplicate = True
                                break

                    if not is_duplicate:
                        # It's unique (e.g., inside a chart image). Keep it.
                        spatial_lines.append(
                            f"[{int(y1_img)}, {int(x1_img)}] {text}"
                        )
                        # Structured word record (normalized 0-1)
                        if pw > 0 and ph > 0:
                            word_records.append({
                                "text": text,
                                "bbox_x": x1_pdf / pw, "bbox_y": y1_pdf / ph,
                                "bbox_w": (x2_pdf - x1_pdf) / pw,
                                "bbox_h": (y2_pdf - y1_pdf) / ph,
                                "block_no": -1, "line_no": -1,
                            })

        except Exception as e:
            logger.warning(f"⚠️ OCR Layer failed: {e}")

        # Sort spatially: Left-right (column order), then Top-down within each column.
        # X-first ensures chart values appear in column order regardless of whether
        # bar labels sit above (positive bars) or below (negative bars) the axis.
        # Y-first (reading order) caused column-shift errors on waterfalls: positive
        # bar labels float high (low Y) and were greedily assigned to wrong columns
        # before the negative bar labels (high Y) were reached.
        def parse_sort_key(line):
            try:
                coords = line.split("]")[0].strip("[").split(",")
                return int(coords[1]), int(coords[0])  # (X, Y)
            except Exception:
                return (0, 0)

        spatial_lines.sort(key=parse_sort_key)
        return "\n".join(spatial_lines), word_records

    def _generate_vector_hints(
        self,
        page: fitz.Page,
        layout: Any = None
    ) -> str:
        """
        Scans vector graphics (shapes) to find semantic links between
        legend text labels and color swatches.
        """
        try:
            # 1. Filter: Only look for keys detected by Layout Analyzer
            valid_keys = set()
            if layout and layout.charts:
                for chart in layout.charts:
                    for key in chart.legend_keys:
                        valid_keys.add(key.lower().strip())

            # 2. Build Spatial Index of all vector shapes
            spatial_index = SpatialIndex(page.rect)
            for d in page.get_drawings():
                spatial_index.insert(d)

            # 3. Iterate Text Blocks to find potential Legend Labels
            text_blocks = page.get_text("dict")["blocks"]
            raw_legend_bindings: List[LegendBinding] = []

            for block in text_blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        # Skip noise (single chars, numbers)
                        if len(text) < 2 or any(c.isdigit() for c in text):
                            continue

                        # Semantic Check
                        is_confirmed = (
                            (text.lower() in valid_keys) if valid_keys else False
                        )

                        # Define Search Area (Small buffer around text)
                        span_rect = fitz.Rect(span["bbox"])
                        search_rect = span_rect + LEGEND_SEARCH_RADIUS

                        # Query Spatial Index for nearby shapes
                        nearby_shapes = spatial_index.query(search_rect)
                        candidates = []

                        for shape in nearby_shapes:
                            if shape.get('fill') is None:
                                continue

                            # Validate Color
                            r, g, b = shape['fill']
                            rgb_255 = (int(r * 255), int(g * 255), int(b * 255))
                            if not self._is_valid_color(rgb_255):
                                continue

                            # Check Distance
                            s_rect = fitz.Rect(shape['rect'])
                            dist = self._rect_distance(span_rect, s_rect)

                            # Tight binding only
                            if dist < LEGEND_DISTANCE_THRESHOLD:
                                hex_c = '#{:02x}{:02x}{:02x}'.format(*rgb_255)
                                candidates.append((dist, rgb_255, hex_c))

                        if candidates:
                            # Closest shape wins
                            candidates.sort(key=lambda x: x[0])
                            best = candidates[0]
                            # Store dictionary for updated ColorResolver
                            binding: LegendBinding = {
                                "text": text,
                                "rgb": best[1],
                                "hex": best[2],
                                "confirmed": is_confirmed
                            }
                            raw_legend_bindings.append(binding)

            # 4. Resolve Color Names (RGB -> "Dark Blue")
            if not raw_legend_bindings:
                return ""

            hints = ColorResolver.resolve_names(raw_legend_bindings)
            return "\n".join(hints)

        except Exception as e:
            logger.warning(f"Vector hint generation failed: {e}")
            return ""

    # ==========================================================================
    # 🧱 CONTENT PROCESSORS
    # ==========================================================================

    async def _process_complex_table(
        self,
        item,
        doc,
        doc_hash: str,
        filename: str,
        bbox_kwargs: dict = None,
        pdf_doc: fitz.Document = None
    ) -> List[DocItem]:
        """
        Handles high-fidelity table extraction via Docling.
        Constructs hierarchical FinancialTableItem objects.
        """
        df = item.export_to_dataframe(doc)
        if df.empty:
            return []

        # Store HTML representation for the UI
        html_content = df.to_html()
        chunk_id = str(uuid4())
        await BlobDriver.save_json({"html": html_content}, "tables", f"{chunk_id}.json")

        page_no = item.prov[0].page_no if item.prov else 1

        # Extract per-cell bboxes from Docling's internal table structure
        cell_bboxes: Dict[str, List[List[float]]] = {}
        if pdf_doc and hasattr(item, 'data') and hasattr(item.data, 'table_cells'):
            page_obj = pdf_doc[page_no - 1]
            pw, ph = page_obj.rect.width, page_obj.rect.height
            if pw > 0 and ph > 0:
                for cell in item.data.table_cells:
                    if cell.bbox and cell.text.strip():
                        cb = cell.bbox
                        if hasattr(cb, "coord_origin") and cb.coord_origin.value == "BOTTOMLEFT":
                            cb = cb.to_top_left_origin(ph)
                        key = cell.text.strip()
                        cell_bboxes.setdefault(key, []).append([
                            cb.l / pw, cb.t / ph, cb.width / pw, cb.height / ph
                        ])
        
        # Merge Headers and Footnotes for Metadata Detection
        headers = " ".join([str(c) for c in df.columns])
        footnotes = ""
        if hasattr(item, "captions"):
            footnotes += " ".join([getattr(c, "text", str(c)) for c in item.captions or []])
        if hasattr(item, "footnotes"):
            footnotes += " ".join([getattr(f, "text", str(f)) for f in item.footnotes or []])

        # Call external utility instead of internal method
        meta = extract_table_metadata(headers, footnotes)
        
        # Build Typed Rows
        rows = [
            FinancialTableRow(
                cells=[str(c) for c in r], 
                row_type="header" if i == 0 else "data"
            ) for i, r in enumerate(df.values)
        ]

        # Safe access and validation against Literal definition
        # Using .get() prevents KeyError, checking get_args prevents TypeError
        p_val = meta.get("periodicity")
        if p_val in get_args(PeriodicityType):
            safe_periodicity = p_val
        else:
            safe_periodicity = "Unknown"

        try:
            return [FinancialTableItem(
                id=chunk_id,
                source=SourceRef(
                    file_hash=doc_hash,
                    page_number=page_no,
                    **(bbox_kwargs or {})
                ),
                value_bboxes=cell_bboxes or None,
                html_repr=html_content,
                rows=rows,
                accounting_basis=meta.get("accounting_basis"),
                periodicity=safe_periodicity,
                currency=meta.get("currency")
            )]
        except ValidationError as e:
            logger.warning(f"⚠️ Table validation failed in {filename} (pg {page_no}): {e}")
            return []

    async def _process_page_vlm(
        self,
        page_num: int,
        pdf_doc: fitz.Document,
        page_table_items: Dict[int, List[DocItem]],
        doc_hash: str
    ) -> Tuple[List[DocItem], List[Dict], RoutingSignals]:
        """
        Runs the Visual Pipeline on a single page.
        Isolated for safe concurrency.
        """
        # Note: fitz is not thread-safe for write ops, but read ops are generally safe.
        # We access the page object within this async context.
        page = pdf_doc[page_num - 1]
        page_w, page_h = page.rect.width, page.rect.height
        local_items, local_quarantine = [], []
        signals = RoutingSignals()

        # 1. Render Page to Image (Use Fitz directly)
        zoom = RENDER_ZOOM
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        
        # Get raw PNG bytes directly from Fitz (fastest)
        # Eliminates the Fitz -> PIL -> PNG double-encoding bottleneck
        img_bytes = pix.tobytes("png")
        
        # Create PIL Image only for OCR context (requires numpy array)
        pil_img = Image.open(io.BytesIO(img_bytes)) 
        
        # Encode for VLM
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # 2. Layout Analysis (The "Glance")
        layout = await layout_analyzer.scan(img_b64)
        
        # 3. Resolve Split-Brain (Docling vs VLM)
        # Passing dimensions for future bbox normalization
        resolved_tables = self._resolve_split_brain(
            page_table_items.get(page_num, []), 
            layout.charts, 
            page_h, 
            page_w
        )
        local_items.extend(resolved_tables)

        # [Fix 3] Skip VLM for pages with no non-TABLE chart regions.
        # Title slides, text-only pages, and pure-table pages are handled by Docling.
        non_table_charts = [
            c for c in layout.charts
            if c.visual_type != VisualArchetype.TABLE
        ]
        if not non_table_charts:
            return local_items, local_quarantine, signals

        # 4. Build Context (OCR + Vector Hints)
        ocr_context, word_bboxes = await self._extract_text_spatial(page, np.array(pil_img), zoom)
        vector_hints = self._generate_vector_hints(page, layout=layout)
        full_context = f"[OCR SPATIAL GRID]\n{ocr_context}\n\n[VECTOR LEGEND HINTS]\n{vector_hints}"

        # 5. Visual Extraction (The "Read")
        analysis = await visual_extractor.analyze_full_page(img_bytes, full_context, layout)

        # 6. Build value-level bboxes by matching VLM metrics to OCR words
        vb = _build_value_bboxes(analysis.metrics, word_bboxes, layout.charts) if analysis else None

        if analysis:
            eid = str(uuid4())

            # [Fix 4] Summary is already on VisualItem.summary — no separate
            # NarrativeItem needed. Eliminates duplication and incorrect
            # is_strategic_claim=True on factual data summaries.

            # A. Process Insights (Narrative)
            for insight in analysis.insights:
                try:
                    local_items.append(NarrativeItem(
                        id=str(uuid4()),
                        layout_cluster_id=eid,
                        source=SourceRef(file_hash=doc_hash, page_number=page_num),
                        text_content=insight.content,
                        sentiment=(
                            "Positive" if insight.stated_direction == "positive_contributor"
                            else "Negative" if insight.stated_direction == "negative_contributor"
                            else "Unknown"
                        ),
                        is_strategic_claim=(insight.category == "Strategic"),
                        category=insight.category  # [Fix 6c] Propagate full taxonomy
                    ))
                except ValidationError as e:
                    local_quarantine.append({
                        "type": "insight", "error": str(e),
                        "content": insight.model_dump()
                    })

            # B. Process Charts & Visuals
            if layout.charts:
                # [Fix 2] Check if VLM assigned region IDs to metrics.
                # When all metrics are untagged and multiple charts exist,
                # emit a single VisualItem to prevent duplication.
                all_untagged = (
                    all(m.source_region_id is None for m in analysis.metrics)
                    if analysis.metrics else True
                )

                if all_untagged and len(layout.charts) > 1:
                    # VLM didn't partition by region — emit single VisualItem
                    primary = layout.charts[0]
                    try:
                        local_items.append(VisualItem(
                            id=str(uuid4()),
                            layout_cluster_id=eid,
                            source=SourceRef(
                                file_hash=doc_hash,
                                page_number=page_num,
                                layout_id=str(primary.region_id),
                                **_chart_bbox_to_normalized(primary.bbox)
                            ),
                            archetype=primary.visual_type,
                            title=(
                                analysis.title
                                if analysis.title and analysis.title.lower() != "untitled"
                                else primary.title or analysis.title
                            ),
                            summary=analysis.summary,
                            metrics=analysis.metrics,
                            value_bboxes=vb
                        ))
                    except ValidationError as e:
                        local_quarantine.append({
                            "type": "chart", "error": str(e),
                            "bbox": primary.bbox
                        })

                    # Propagate routing signals from all charts
                    for chart in layout.charts:
                        if chart.visual_type in (VisualArchetype.WATERFALL, VisualArchetype.VALUATION_FIELD):
                            signals.has_valuation_models = True
                else:
                    for chart in layout.charts:
                        # [Fix 5] Skip TABLE charts when Docling already captured the data.
                        if chart.visual_type == VisualArchetype.TABLE:
                            has_docling_table = any(
                                isinstance(t, FinancialTableItem)
                                for t in resolved_tables
                            )
                            if has_docling_table:
                                continue

                        # Filter metrics by Region ID.
                        chart_metrics = [
                            m for m in analysis.metrics
                            if m.source_region_id == chart.region_id or m.source_region_id is None
                        ]

                        try:
                            bbox_kwargs = _chart_bbox_to_normalized(chart.bbox)
                            if chart.visual_type == VisualArchetype.VALUATION_FIELD:
                                local_items.append(ChartTableItem(
                                    id=str(uuid4()),
                                    layout_cluster_id=eid,
                                    source=SourceRef(
                                        file_hash=doc_hash,
                                        page_number=page_num,
                                        layout_id=str(chart.region_id),
                                        **bbox_kwargs
                                    ),
                                    archetype=chart.visual_type,
                                    title=chart.title,
                                    visual_metrics=chart_metrics,
                                    value_bboxes=vb
                                ))
                                signals.has_valuation_models = True
                            else:
                                local_items.append(VisualItem(
                                    id=str(uuid4()),
                                    layout_cluster_id=eid,
                                    source=SourceRef(
                                        file_hash=doc_hash,
                                        page_number=page_num,
                                        layout_id=str(chart.region_id),
                                        **bbox_kwargs
                                    ),
                                    archetype=chart.visual_type,
                                    title=chart.title,
                                    summary=analysis.summary,
                                    metrics=chart_metrics,
                                    value_bboxes=vb
                                ))

                            if chart.visual_type == VisualArchetype.WATERFALL:
                                signals.has_valuation_models = True

                        except ValidationError as e:
                            local_quarantine.append({
                                "type": "chart", "error": str(e), "bbox": chart.bbox
                            })

        return local_items, local_quarantine, signals

    # Using process_outputs to mask the massive return value from LangSmith
    @traceable(
        name="Parse PDF File", 
        run_type="parser",
        process_outputs=lambda x: f"<UnifiedDocument: {len(x.items)} items, {len(x.quarantined_items)} failures>" if x else "None"
    )
    async def parse(self, file_path: Path) -> UnifiedDocument:
        """
        Main Entry Point.
        Performs parallel ingestion of the PDF file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"🎨 Starting Centaur ingestion for: {file_path.name}")

        valid_items, quarantined_items = [], []
        doc_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        doc_id = str(uuid4())
        master_signals = RoutingSignals()

        # Open fitz doc early — used for both Docling bbox normalization and VLM
        pdf_doc = fitz.open(file_path)
        try:

            # --- STEP 1: DOCLING (Logical Structure) ---
            # Run in executor to avoid blocking the event loop with CPU work
            loop = asyncio.get_running_loop()
            docling_res = await loop.run_in_executor(
                None, self.converter.convert, file_path
            )

            page_table_items = {}
            # Extract ALL content types (Tables, Headers, Text)
            for item, _ in docling_res.document.iterate_items():
                p_no = item.prov[0].page_no if item.prov else 1
                bbox_kw = _docling_bbox_to_normalized(item, pdf_doc)

                if isinstance(item, TableItem):
                    new_t = await self._process_complex_table(
                        item, docling_res.document, doc_hash, file_path.name,
                        bbox_kwargs=bbox_kw, pdf_doc=pdf_doc
                    )
                    page_table_items.setdefault(p_no, []).extend(new_t)

                elif isinstance(item, SectionHeaderItem) and item.text.strip():
                    valid_items.append(HeaderItem(
                        id=str(uuid4()),
                        source=SourceRef(file_hash=doc_hash, page_number=p_no, **bbox_kw),
                        text=item.text.strip(),
                        level=getattr(item, 'level', 1)
                    ))

                elif isinstance(item, (TextItem, ListItem)) and len(item.text.strip()) > 15:
                    # Basic noise filter (>15 chars) to avoid page numbers/artifacts
                    narr_text = item.text.strip()
                    sentence_bboxes = _build_sentence_bboxes(
                        narr_text, pdf_doc[p_no - 1], bbox_kw
                    )
                    valid_items.append(NarrativeItem(
                        id=str(uuid4()),
                        source=SourceRef(file_hash=doc_hash, page_number=p_no, **bbox_kw),
                        text_content=narr_text,
                        value_bboxes=sentence_bboxes
                    ))

            # --- STEP 2: VLM (Visual Intelligence) ---
            # Concurrency Control via Semaphore
            sem = asyncio.Semaphore(MAX_CONCURRENT_PAGES)
            tasks = []
            total_pages = len(pdf_doc)

            async def protected_process(p):
                async with sem:
                    # Timeout Wrapper
                    # If a page hangs (e.g. corrupt image), we kill it after N seconds
                    return await asyncio.wait_for(
                        self._process_page_vlm(
                            p, pdf_doc, page_table_items, doc_hash
                        ),
                        timeout=PAGE_TIMEOUT_SECONDS
                    )

            for p in range(1, total_pages + 1):
                tasks.append(protected_process(p))

            # return_exceptions=True
            # If one page crashes/times out, we do NOT want to kill the whole batch.
            # We will catch the exception in the result loop.
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                page_idx = i + 1
                if isinstance(result, Exception):
                    # Handle page-level failures gracefully
                    logger.error(f"❌ Page {page_idx} failed: {result}")
                    quarantined_items.append({
                        "type": "page_failure",
                        "page": page_idx,
                        "error": str(result)
                    })
                    continue
                
                # Unpack successful result
                p_items, p_quarantine, p_signals = result
                valid_items.extend(p_items)
                quarantined_items.extend(p_quarantine)
                if p_signals.has_valuation_models:
                    master_signals.has_valuation_models = True

        finally:
            # Always close the file handle
            pdf_doc.close()

        # --- STEP 3: CLASSIFICATION ---
        if any(isinstance(i, FinancialTableItem) and "Pro Forma" in i.accounting_basis for i in valid_items):
            master_signals.has_pro_forma_adjustments = True
            master_signals.detected_artifact_type = "CIM"
            master_signals.deal_relevance_score = 1.0
        elif master_signals.has_valuation_models:
            master_signals.detected_artifact_type = "FinancialModel"
            master_signals.deal_relevance_score = 0.9

        logger.info(f"✅ Ingested {len(valid_items)} items across {total_pages} pages.")

        return UnifiedDocument(
            doc_id=doc_id, 
            filename=file_path.name, 
            items=valid_items,
            quarantined_items=quarantined_items, 
            signals=master_signals
        )

# Singleton Instance
pdf_parser = PDFParser()