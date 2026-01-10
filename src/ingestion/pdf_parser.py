import logging
import hashlib
import math
import collections
import re
import statistics
import string
import asyncio
import io
import colorsys
import numpy as np
from uuid import uuid4
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

# Third-party
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import DocItem, TableItem, PictureItem
from rapidocr_onnxruntime import RapidOCR
from langsmith import traceable

# Internal
from src.schemas.documents import IngestedChunk
from src.schemas.citation import BoundingBox
from src.storage.blob_driver import BlobDriver
from src.tools.vision import vision_tool
from src.config import SystemPaths

logger = logging.getLogger(__name__)

# Separate executor for OCR/Image ops to prevent starvation of the main loop
_OCR_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_IMG_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# ==============================================================================
# 🎨 INTELLIGENT SHADE RESOLVER
# ==============================================================================
class ColorResolver:
    """
    Resolves ambiguous colors (e.g. three shades of red) into semantic names
    like 'Dark Red', 'Medium Red', 'Light Red'.
    """
    @staticmethod
    def get_luminance(rgb: Tuple[int, int, int]) -> float:
        """Calculates perceived brightness (Standard Rec. 709)."""
        return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])

    @staticmethod
    def get_base_hue(rgb: Tuple[int, int, int]) -> str:
        """
        Robustly maps RGB to a Base Hue Bucket using HSL.
        Prevents 'Dark Red' from being misclassified as 'Black' or 'Brown'.
        """
        r, g, b = rgb
        h, _, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        if s < 0.10: return "Gray"
        deg = h * 360
        if deg < 15 or deg > 345: return "Red"
        if deg < 45: return "Orange"
        if deg < 70: return "Yellow"
        if deg < 150: return "Green"
        if deg < 200: return "Cyan"
        if deg < 260: return "Blue"
        if deg < 300: return "Purple"
        return "Pink"

    @classmethod
    def resolve_names(cls, bindings: List[Tuple[str, Tuple[int, int, int], str]]) -> List[str]:
        """
        Input: List of (Label, RGB_Tuple, Hex_String)
        Output: List of strings "Legend Key: 'Label' == Hex (Semantic Name)"
        """
        # 1. Group by Base Hue
        groups = collections.defaultdict(list)
        for label, rgb, hex_c in bindings:
            base = cls.get_base_hue(rgb)
            groups[base].append({
                "label": label, 
                "rgb": rgb, 
                "hex": hex_c, 
                "lum": cls.get_luminance(rgb)
            })

        results = []

        # 2. Process each group to resolve collisions
        for base_name, items in groups.items():
            if len(items) == 1:
                # No collision: Just use the Base Name
                i = items[0]
                results.append(f"Legend: '{i['label']}' is {base_name} ({i['hex']})")
            else:
                # Collision: Sort by Luminance (Darkest to Lightest)
                items.sort(key=lambda x: x["lum"])
                
                count = len(items)
                for idx, item in enumerate(items):
                    if count == 2:
                        mod = "Dark" if idx == 0 else "Light"
                    elif count == 3:
                        mod = ["Dark", "Medium", "Light"][idx]
                    else:
                        mod = f"Luminance-{int((idx / (count - 1)) * 100)}"
                    results.append(f"Legend: '{item['label']}' is {mod} {base_name} ({item['hex']})")
        return results

# ==============================================================================
# 🧲 SPATIAL INDEX & ANCHOR (The Snap Logic)
# ==============================================================================
class SpatialIndex:
    def __init__(self, page_rect: fitz.Rect, cell_size: int = 50):
        self.page_w = page_rect.width
        self.page_h = page_rect.height
        self.cell_size = cell_size
        self.cols = math.ceil(self.page_w / cell_size)
        self.rows = math.ceil(self.page_h / cell_size)
        self.grid = collections.defaultdict(list)

    def _get_keys(self, rect: fitz.Rect):
        start_col = max(0, int(rect.x0 // self.cell_size))
        end_col = min(self.cols, int(rect.x1 // self.cell_size) + 1)
        start_row = max(0, int(rect.y0 // self.cell_size))
        end_row = min(self.rows, int(rect.y1 // self.cell_size) + 1)
        for c in range(start_col, end_col):
            for r in range(start_row, end_row):
                yield (c, r)

    def insert(self, shape: dict):
        rect = fitz.Rect(shape['rect'])
        for key in self._get_keys(rect):
            self.grid[key].append(shape)

    def query(self, rect: fitz.Rect) -> List[dict]:
        candidates = set()
        results = []
        for key in self._get_keys(rect):
            if key in self.grid:
                for shape in self.grid[key]:
                    s_id = id(shape)
                    if s_id not in candidates:
                        candidates.add(s_id)
                        results.append(shape)
        return results

    def snap_box(self, fuzzy_box: fitz.Rect, threshold: float = 0.4) -> Tuple[fitz.Rect, List[dict]]:
        """
        The Anchor: Expands a fuzzy VLM box to include any vector text/objects it partially touches.
        Prevents cutting off labels like '2025E' or axes.
        Returns: (SnappedRect, ListOfItemsInside)
        """
        snapped = fitz.Rect(fuzzy_box)
        # Search wider area to catch hanging labels
        search_rect = fuzzy_box + (-30, -30, 30, 30)
        candidates = self.query(search_rect)
        
        valid_items = []
        
        for obj in candidates:
            obj_rect = fitz.Rect(obj['rect'])
            if obj_rect.get_area() > (self.page_w * self.page_h * 0.9): continue

            intersection = (obj_rect & fuzzy_box).get_area()
            obj_area = obj_rect.get_area()
            
            # Smart Inclusion Logic:
            # 1. Shapes: Snap if box covers >40% OR shape is mostly inside.
            # 2. Text: Snap if it physically touches or intersects the box (critical for labels).
            is_text = (obj.get('type') == 'text')
            shape_match = (not is_text) and (intersection > obj_area * threshold or obj_rect.intersects(fuzzy_box))
            text_match = is_text and (intersection > obj_area * 0.15) # Lower threshold for text labels

            if shape_match or text_match:
                if snapped.is_empty: 
                    snapped = obj_rect
                else: 
                    snapped.include_rect(obj_rect)
                valid_items.append(obj)
                
        # If no vectors found (Raster PDF), return original fuzzy box
        if snapped.is_empty:
            return fuzzy_box, []
            
        # Add small final buffer to the snapped box
        return snapped + (-10, -10, 10, 10), valid_items

# ==============================================================================
# 🧠 INTELLIGENT CLASSIFIER (OPTIMIZED)
# ==============================================================================
@traceable(name="Classify Region", run_type="chain")
def classify_region(text_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Universal classifier for Financial Structures vs. Prose.
    """
    if not text_items:
        # [SAFETY] Return default structure to prevent KeyError
        return {
            "decision": "chart", "score": 0.0, "debug": "No text", 
            "evidence": {"layout_structure": "Image/Graphic", "chart_type_hint": "Visual Only"}, 
            "features": {}
        }

    # --- PRE-PROCESSING ---
    full_text = " ".join([t.get('text', '') for t in text_items])
    tokens = full_text.split()
    total_tokens = len(tokens) if tokens else 1
    
    # --- VARIABLE 1: Alpha-Token Ratio ---
    clean_tokens = [t.strip(string.punctuation) for t in tokens]
    alpha_count = sum(1 for t in clean_tokens if t.isalpha() and len(t) > 1)
    v1_structure_signal = 1.0 - (alpha_count / total_tokens)

    # --- VARIABLE 2: Universal Entity Density (RESTORED ROBUST REGEX) ---
    score_v2_raw = 0.0
    # Currency / Financials
    currency_pattern = r'[$€£¥]|\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY|HKD|SGD|NZD|KRW)\b|\(\d{1,3}(?:,\d{3})*\)|\d+(?:\.\d+)?[x%]'
    score_v2_raw += len(re.findall(currency_pattern, full_text, re.IGNORECASE)) * 2.0
    # Asset Units & Safe CUSIPs
    safe_asset_regex = r'\b(sq\s?ft|sf|m2|psf|keys|MW|GWh|TEU|dwt|mt|bbl)\b|\b(?=.*\d)(?=.*[A-Z])[A-Z0-9]{9,12}\b'
    score_v2_raw += len(re.findall(safe_asset_regex, full_text, re.IGNORECASE)) * 3.0
    # Dates/Years
    score_v2_raw += len(re.findall(r'\b(?:FY|Q[1-4]|LTM)\d{2}\b|\b2[0O]\d{2}[E]?\b', full_text)) * 1.5
    
    v2_data_density = min(score_v2_raw / total_tokens, 1.0)

    # --- VARIABLE 3: Geometric Alignment ---
    x_starts = sorted([t.get('bbox', [0])[0] for t in text_items])
    v3_grid_score = 0.0
    
    if len(x_starts) > 5:
        clusters = []
        if x_starts:
            current = [x_starts[0]]
            region_width = x_starts[-1] - x_starts[0]
            jitter_tol = max(15, region_width * 0.02)
            
            for x in x_starts[1:]:
                if x - current[-1] < jitter_tol:
                    current.append(x)
                else:
                    clusters.append(current)
                    current = [x]
            clusters.append(current)
        
        valid_cols = [c for c in clusters if len(c) > len(x_starts) * 0.10]
        num_peaks = len(valid_cols)
        
        if num_peaks >= 3:
            v3_grid_score = 1.0
        elif num_peaks == 2:
            # Rent Roll Logic
            mid_x = (x_starts[0] + x_starts[-1]) / 2
            right_col_text = " ".join([t.get('text', '') for t in text_items if t.get('bbox', [0])[0] > mid_x])
            digit_ratio = sum(c.isdigit() for c in right_col_text) / (len(right_col_text) or 1)
            right_tokens = right_col_text.split()
            title_ratio = sum(1 for w in right_tokens if w.istitle() or w.isupper()) / (len(right_tokens) or 1)
            
            if digit_ratio > 0.4 or title_ratio > 0.25:
                v3_grid_score = 0.85
            else:
                v3_grid_score = 0.20

    # --- VARIABLE 4: Guardrails ---
    has_toc = len(re.findall(r'\.{5,}\s*\d+$', full_text, re.MULTILINE)) > 1
    header_sample = full_text[:200].lower()
    is_legal = re.search(r'^(article|section)\s+\d', header_sample)
    is_note = "note" in header_sample 
    legal_header = re.search(r'forward[- ]looking|risk factors|disclaimer', header_sample, re.IGNORECASE)
    v4_penalty = 0.0
    if has_toc: v4_penalty = 1.0 
    elif (is_legal and not is_note) or legal_header: v4_penalty = 0.8 

    # --- FINAL SCORING ---
    raw_score = (0.35 * v1_structure_signal) + (0.30 * v3_grid_score) + (0.35 * v2_data_density)
    final_score = raw_score * (1.0 - v4_penalty)
    
    # --- TELEMETRY ---
    waterfall_keywords = {'bridge', 'offset', 'impact', 'walk', 'evolution', 'decrease', 'increase'}
    found_waterfall_terms = [w for w in waterfall_keywords if w in full_text.lower()]

    evidence = {
        "content_style": "Financial/Asset Data" if v2_data_density > 0.3 else "Prose/Text",
        "layout_structure": "Grid" if v3_grid_score == 1.0 else ("Key-Value" if v3_grid_score == 0.85 else "Freeform"),
        "chart_type_hint": "Waterfall/Bridge" if found_waterfall_terms else ("Financial Structure" if final_score > 0.5 else "Standard"),
        "semantic_keywords": found_waterfall_terms,
        "risk_flag": "TOC/Legal" if v4_penalty > 0.5 else "None"
    }

    return {
        "decision": "chart" if final_score > 0.50 else "general",
        "score": round(final_score, 3),
        "evidence": evidence,
        "features": {
            "alpha_inv": round(v1_structure_signal, 2),
            "asset_density": round(v2_data_density, 2),
            "grid_score": round(v3_grid_score, 2)
        }
    }

# ==============================================================================
# 📄 PDF PARSER (Vision-Guided Architecture)
# ==============================================================================
class PDFParser:
    def __init__(self):
        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.do_table_structure = True
        opts.generate_page_images = False  # Manual handling for thread safety
        
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        self.ocr_engine = RapidOCR()

    @traceable(name="Parse PDF File", run_type="parser")
    async def parse(self, file_path: Path) -> List[IngestedChunk]:
        if not file_path.exists(): 
            raise FileNotFoundError(f"{file_path}")
        
        logger.info(f"🎨 Starting Centaur Ingestion: {file_path.name}")
        loop = asyncio.get_running_loop()
        
        # 1. Docling First (Get Tables & Prose)
        # We still use Docling for the heavy lifting of extracting clean text tables
        result = await loop.run_in_executor(None, self.converter.convert, file_path)
        doc = result.document
        with open(file_path, "rb") as f:
            doc_hash = hashlib.sha256(f.read()).hexdigest()
            
        chunks = []
        
        # Track areas we have already processed (e.g. Docling tables) so VLM doesn't duplicate
        processed_masks = collections.defaultdict(list) 

        # 2. Process Docling Tables (High Confidence)
        for item, _ in doc.iterate_items():
            if isinstance(item, TableItem):
                df_string = item.export_to_dataframe(doc).to_string()
                num_cells = 0
                if hasattr(item, 'data') and hasattr(item.data, 'grid'):
                    num_cells = len(item.data.grid) * (len(item.data.grid[0]) if item.data.grid else 0)
                
                # Filter out visual tables (Heatmaps/Gantts)
                if len(df_string) < 100 and num_cells < 6: 
                    continue 

                table_chunks = await self._process_complex_table(item, doc, doc_hash, file_path.name)
                chunks.extend(table_chunks)
                
                # Masking (Convert Bottom-Left to Top-Left)
                if item.prov:
                    p = item.prov[0]
                    # Mask this area
                    pg_h = doc.pages[p.page_no].size.height
                    tl_y0 = pg_h - p.bbox.t
                    tl_y1 = pg_h - p.bbox.b
                    r = fitz.Rect(p.bbox.l, tl_y0, p.bbox.r, tl_y1)
                    processed_masks[p.page_no].append(r)

        # 3. Vision-Guided Extraction
        with fitz.open(file_path) as pdf_doc:
            for page_idx in range(len(pdf_doc)):
                page = pdf_doc[page_idx]
                page_no = page_idx + 1
                
                # [PERFORMANCE] Render ONE high-res image per page in main thread
                # This avoids passing fitz.Page to threads (Thread Safety Fix)
                pix_hires = page.get_pixmap(dpi=200)
                img_hires_bytes = pix_hires.tobytes("png") 
                
                # Create low-res for Layout Detection (speed)
                pix_lowres = page.get_pixmap(dpi=72)
                
                # B. The Scout: Layout Detection
                try:
                    layout_regions = await vision_tool.detect_layout(pix_lowres.tobytes("jpeg"))
                except Exception as e:
                    logger.warning(f"Router failed on p{page_no}: {e}")
                    layout_regions = []

                # [SAFETY] Cap at 5 largest regions to prevent processing "Icon Storms"
                if len(layout_regions) > 5:
                    layout_regions.sort(key=lambda x: (x.get('box_2d')[2]-x.get('box_2d')[0])*(x.get('box_2d')[3]-x.get('box_2d')[1]), reverse=True)
                    layout_regions = layout_regions[:5]

                # B. Build Spatial Index
                spatial = SpatialIndex(page.rect)
                raw_text_blocks = []
                
                for d in page.get_drawings(): 
                    spatial.insert({'rect': d['rect'], 'fill': d.get('fill'), 'stroke': d.get('color'), 'type': 'vector'})
                
                for b in page.get_text("dict")["blocks"]:
                    for l in b.get("lines", []):
                        for s in l.get("spans", []):
                            spatial.insert({
                                'rect': s['bbox'], 'text': s['text'], 'type': 'text',
                                'center_x': (s['bbox'][0]+s['bbox'][2])/2
                            })
                            raw_text_blocks.append((s['bbox'][1], s['text']))

                slide_title = self._extract_page_title(raw_text_blocks)

                # C. Process Regions
                for region in layout_regions:
                    label = region.get('label', 'Chart')
                    vlm_box_norm = region.get('box_2d', []) 
                    if not vlm_box_norm: continue

                    # Normalize VLM coords -> PDF coords
                    # Note: Check if your VLM output matches [ymin, xmin, ymax, xmax] or [xmin, ymin...]
                    # Standard GPT-4o is typically [ymin, xmin, ymax, xmax]
                    y0, x0, y1, x1 = vlm_box_norm
                    w, h = page.rect.width, page.rect.height
                    fuzzy_rect = fitz.Rect(
                        x0/1000*w - 15, y0/1000*h - 15, 
                        x1/1000*w + 15, y1/1000*h + 15
                    )
                    fuzzy_rect &= page.rect

                    # D. The Snap
                    snapped_rect, valid_items = spatial.snap_box(fuzzy_rect)
                    
                    overlap = False
                    for m in processed_masks[page_no]:
                        if (snapped_rect & m).get_area() > (snapped_rect.get_area() * 0.5):
                            overlap = True
                            break
                    if overlap: continue
                    processed_masks[page_no].append(snapped_rect)

                    # E. The Analyst
                    chunk = await self._process_snapped_visual(
                        img_hires_bytes, page.rect, snapped_rect, valid_items, 
                        spatial, doc_hash, file_path.name, label, page_no, slide_title
                    )
                    if chunk: chunks.append(chunk)

        # 4. Cleanup: Prose
        for item, _ in doc.iterate_items():
            if getattr(item, 'label', None) and item.label.value in ["page_header", "page_footer"]: continue
            if hasattr(item, "text") and item.text.strip():
                p = item.prov[0]
                pg_h = doc.pages[p.page_no].size.height
                tl_y0 = pg_h - p.bbox.t
                tl_y1 = pg_h - p.bbox.b
                r = fitz.Rect(p.bbox.l, tl_y0, p.bbox.r, tl_y1)
                
                overlap = False
                for m in processed_masks[p.page_no]:
                    if (r & m).get_area() > (r.get_area() * 0.5):
                        overlap = True
                        break
                if overlap: continue
                
                c = self._create_standard_chunk(item, doc, doc_hash, file_path.name)
                if c: chunks.append(c)

        logger.info(f"✅ Extracted {len(chunks)} chunks from {file_path.name}")
        return chunks

    def _extract_page_title(self, text_blocks: List[Tuple[float, str]]) -> str:
        """Finds topmost text line (likely title)."""
        if not text_blocks: return ""
        text_blocks.sort(key=lambda x: x[0])
        for _, text in text_blocks:
            if len(text.strip()) > 3: return text.strip()
        return ""

    def _crop_and_ocr_sync(self, img_bytes: bytes, page_rect: fitz.Rect, crop_rect: fitz.Rect):
        """Thread-safe image processing (No fitz objects)."""
        try:
            full_img = Image.open(io.BytesIO(img_bytes))
            scale_x = full_img.width / page_rect.width
            scale_y = full_img.height / page_rect.height
            
            crop_px = (
                (crop_rect.x0 - page_rect.x0) * scale_x,
                (crop_rect.y0 - page_rect.y0) * scale_y,
                (crop_rect.x1 - page_rect.x0) * scale_x,
                (crop_rect.y1 - page_rect.y0) * scale_y
            )
            crop_img = full_img.crop(crop_px)
            crop_arr = np.array(crop_img)
            return crop_img, crop_arr
        except Exception as e:
            logger.error(f"Crop failed: {e}")
            return None, None

    async def _process_snapped_visual(self, img_bytes, page_rect, rect, vector_items, spatial, doc_hash, fname, label, page_no, slide_title):
        try:
            loop = asyncio.get_running_loop()
            
            # 1. Thread-safe Crop
            img_io, img_arr = await loop.run_in_executor(_IMG_EXECUTOR, self._crop_and_ocr_sync, img_bytes, page_rect, rect)
            if img_io is None: return None

            # 2. OCR Fallback
            if not any(i.get('type') == 'text' for i in vector_items):
                ocr_res = await loop.run_in_executor(_OCR_EXECUTOR, lambda: self.ocr_engine(img_arr))
                if ocr_res:
                    sx, sy = rect.width / img_io.width, rect.height / img_io.height
                    for r in ocr_res:
                        poly, txt = r[0], r[1]
                        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
                        l, t, r_x, b = min(xs), min(ys), max(xs), max(ys)
                        pdf_r = [rect.x0 + l*sx, rect.y0 + t*sy, rect.x0 + r_x*sx, rect.y0 + b*sy]
                        vector_items.append({'type': 'text', 'text': txt, 'rect': pdf_r, 'center_x': (pdf_r[0]+pdf_r[2])/2})

            raw_text_parts = [i['text'] for i in vector_items if i.get('type') == 'text']
            raw_text_str = " ".join(raw_text_parts)
            
            legends, data_point_hints = [], []
            
            # 3. Probe Loop
            for item in vector_items:
                if item.get('type') != 'text': continue
                txt = item['text'].strip()
                digs = sum(c.isdigit() for c in txt)
                is_numeric = (digs/len(txt) > 0.4) and len(txt) < 12 and not re.match(r'^(?:FY|Q[1-4]|20)\d{2}$', txt, re.I)
                
                match = None
                # A. Vector Probe (Exact)
                if 'rect' in item:
                    match = self._vector_radar_probe(spatial, item['rect'], rect, is_numeric=is_numeric)
                
                # B. Raster Probe (Visual)
                if not match:
                    l_rel = (item['rect'][0] - rect.x0) * (img_io.width / rect.width)
                    t_rel = (item['rect'][1] - rect.y0) * (img_io.height / rect.height)
                    r_rel = (item['rect'][2] - rect.x0) * (img_io.width / rect.width)
                    b_rel = (item['rect'][3] - rect.y0) * (img_io.height / rect.height)
                    
                    match = self._extract_dominant_colors(img_arr, [l_rel, t_rel, r_rel, b_rel])

                if match:
                    # match format: [('#hex', rgb_tuple), ...]
                    primary_hex = match[0][0]
                    primary_rgb = match[0][1]
                    
                    if is_numeric:
                        # Stacked Bar Logic: Join all found colors to hint at multi-series data
                        all_colors_str = "|".join([c[0] for c in match])
                        data_point_hints.append({"val": txt, "color": all_colors_str, "x": item.get('center_x', 0)})
                    else:
                        legends.append((txt, primary_rgb, primary_hex))

            resolved_legends = ColorResolver.resolve_names(legends)
            
            # 4. Context Build
            data_point_hints.sort(key=lambda k: k['x'])
            grouped_txt = []
            if data_point_hints:
                grps, curr = [], [data_point_hints[0]]
                for h in data_point_hints[1:]:
                    if abs(h['x'] - curr[-1]['x']) < 20: curr.append(h)
                    else: grps.append(curr); curr = [h]
                grps.append(curr)
                for i, g in enumerate(grps):
                    g_str = ", ".join([f"{x['val']}[{x['color']}]" for x in g])
                    grouped_txt.append(f"Col {i+1}: {g_str}")

            # [FIX] Wrap Telemetry in try/except so it doesn't kill the Pipeline
            try:
                cls_items = [{'text': i['text'], 'bbox': i.get('rect')} for i in vector_items if i.get('type')=='text']
                cls_res = classify_region(cls_items)
                layout_structure = cls_res['evidence']['layout_structure']
                chart_hint = cls_res['evidence']['chart_type_hint']
            except Exception as e:
                logger.warning(f"Telemetry failed: {e}")
                layout_structure = "Unknown"
                chart_hint = "General"

            context = (
                f"SLIDE TITLE: {slide_title}\n"
                f"DETECTED REGION: {label}\n"
                f"GROUND TRUTH TEXT:\n{raw_text_str[:2000]}\n\n"
                f"LEGENDS:\n{list(set(resolved_legends))}\n"
                f"DATA STRUCTURE:\n" + "\n".join(grouped_txt[:30]) + "\n\n"
                f"LAYOUT: {cls_res['evidence']['layout_structure']}\n"
                f"SEMANTIC HINTS: {cls_res['evidence']['chart_type_hint']}"
            )

            buf = io.BytesIO()
            img_io.save(buf, format="PNG")
            desc = await vision_tool.analyze(buf.getvalue(), mode="chart", context=context)

            return IngestedChunk(
                chunk_id=str(uuid4()), doc_hash=doc_hash, clean_text=f"[{label.upper()}]\n{desc}",
                raw_text=desc, page_number=page_no, metadata={"source": fname, "type": "visual", "coords": list(rect)}
            )

        except Exception as e:
            logger.error(f"Visual Error on p{page_no}: {e}")
            return None

    def _extract_dominant_colors(self, img_arr: np.ndarray, box: List[float]) -> Optional[List[Tuple[str, Tuple[int, int, int]]]]:
        """
        Improved scanner for Stacked Bars / Waterfalls using Vectorized NumPy.
        Scans neighborhood of text for unique distinct colors.
        """
        try:
            h_img, w_img, _ = img_arr.shape
            l, t, r, b = map(int, box)
            pad = 20
            l, t = max(0, l - pad), max(0, t - pad)
            r, b = min(w_img, r + pad), min(h_img, b + pad)
            
            roi = img_arr[t:b, l:r]
            if roi.size == 0: return None
            
            # Vectorized validity check (Not White, Black, Gray)
            pixels = roi.reshape(-1, 3)
            r_ch, g_ch, b_ch = pixels[:, 0], pixels[:, 1], pixels[:, 2]
            is_white = (r_ch > 240) & (g_ch > 240) & (b_ch > 240)
            is_black = (r_ch < 30) & (g_ch < 30) & (b_ch < 30)
            is_gray = (np.max(pixels, axis=1) - np.min(pixels, axis=1)) < 15
            valid_mask = ~(is_white | is_black | is_gray)
            
            valid_pixels = pixels[valid_mask]
            if valid_pixels.size == 0: return None
            
            # Quantize colors to find distinct shades (e.g. Red vs Green)
            quantized = (valid_pixels // 10) * 10
            unique_colors = np.unique(quantized, axis=0)
            
            results = []
            for c in unique_colors[:4]: # Limit to top 4 nearby colors
                rgb = tuple(int(x) for x in c)
                hex_c = '#{:02x}{:02x}{:02x}'.format(*rgb)
                results.append((hex_c, rgb))
            return results if results else None
        except Exception: return None

    def _vector_radar_probe(self, spatial, t_bbox, search_rect, is_numeric) -> Optional[List[Tuple[str, Tuple[int, int, int]]]]:
        """
        Physics-Aware Probe. Returns list of [(hex, rgb)] to match raster signature.
        Detects multiple nearby vector fills for stacked bar logic.
        """
        t_rect = fitz.Rect(t_bbox)
        probe_rect = t_rect + (-60, -60, 60, 60)
        nearby = spatial.query(probe_rect)
        
        candidates = []
        fills = [s for s in nearby if s.get('fill')]
        strokes = [s for s in nearby if s.get('stroke')]

        # 1. Fill Logic
        for s in fills:
            if not fitz.Rect(s['rect']).intersects(search_rect): continue
            rgb = (int(s['fill'][0]*255), int(s['fill'][1]*255), int(s['fill'][2]*255))
            if self._is_valid_color(rgb):
                dist = self._rect_distance(t_rect, fitz.Rect(s['rect']))
                # If very close, add to candidates
                if dist < 10.0:
                    candidates.append((dist, rgb))

        # Spider Leg Logic (Line connecting text to bar)
        if is_numeric and not candidates:
            for line in strokes:
                l_rect = fitz.Rect(line['rect'])
                if self._rect_distance(t_rect, l_rect) < 5.0:
                    for fill in fills:
                        f_rect = fitz.Rect(fill['rect'])
                        if self._rect_distance(l_rect, f_rect) < 5.0:
                            rgb = (int(fill['fill'][0]*255), int(fill['fill'][1]*255), int(fill['fill'][2]*255))
                            if self._is_valid_color(rgb):
                                candidates.append((1.0, rgb))

        if not candidates: return None
        
        # Deduplicate and format results
        # We return multiple colors if found to support stacked vector bars
        unique_results = {}
        for dist, rgb in candidates:
            hex_c = '#{:02x}{:02x}{:02x}'.format(*rgb)
            if hex_c not in unique_results:
                unique_results[hex_c] = rgb
        
        # Return top 3 closest colors
        return [(h, r) for h, r in list(unique_results.items())[:3]]

    # --- Helpers ---
    def _rect_distance(self, r1, r2):
        x_dist = max(0, r1.x0 - r2.x1, r2.x0 - r1.x1)
        y_dist = max(0, r1.y0 - r2.y1, r2.y0 - r1.y1)
        return math.sqrt(x_dist**2 + y_dist**2)

    def _is_valid_color(self, p):
        r, g, b = p
        if r>240 and g>240 and b>240: return False 
        if r<30 and g<30 and b<30: return False    
        if max(r,g,b)-min(r,g,b) < 15: return False 
        return True

    async def _process_complex_table(self, item, doc, doc_hash, filename) -> List[IngestedChunk]:
        df = item.export_to_dataframe(doc)
        if df.empty: return []
        
        chunks = []
        chunk_size = 10
        total_rows = len(df)
        header_md = f"| {' | '.join(str(c) for c in df.columns)} |\n|{'---|' * len(df.columns)}"
        
        for start_row in range(0, total_rows, chunk_size):
            end_row = min(start_row + chunk_size, total_rows)
            chunk_df = df.iloc[start_row:end_row]
            body_md = chunk_df.to_markdown(index=False, tablefmt="pipe").split('\n', 2)[-1] 
            full_text = f"Context (Headers):\n{header_md}\n\nSegment ({start_row}-{end_row}):\n{body_md}"
            
            chunk_id = str(uuid4())
            await BlobDriver.save_json({"html": df.to_html()}, "tables", f"{chunk_id}.json")

            chunks.append(IngestedChunk(
                chunk_id=chunk_id, doc_hash=doc_hash, clean_text=full_text, raw_text=full_text,
                page_number=item.prov[0].page_no if item.prov else 1, token_count=len(full_text.split()),
                metadata={"source": filename, "type": "table", "rows": f"{start_row}-{end_row}"}
            ))
        return chunks

    def _create_standard_chunk(self, item, doc, doc_hash, filename):
        if not item.prov: return None
        p = item.prov[0]
        pg = doc.pages[p.page_no]
        width, height = pg.size.width, pg.size.height
        
        # [COORD FIX] Docling (Bottom-Left Origin) -> Standard Top-Left
        y0_norm = (height - p.bbox.t) / height
        h_norm = ((height - p.bbox.b) / height) - y0_norm
        
        return IngestedChunk(
            chunk_id=str(uuid4()), doc_hash=doc_hash, clean_text=item.text.strip(), raw_text=item.text,
            page_number=p.page_no, token_count=len(item.text.split()),
            metadata={"source": filename, "type": "text"},
            primary_bbox=BoundingBox(
                page_number=p.page_no, x=p.bbox.l/width, y=y0_norm, width=(p.bbox.r-p.bbox.l)/width, height=h_norm
            )
        )