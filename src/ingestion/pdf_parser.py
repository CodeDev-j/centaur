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

# Separate executor for OCR to prevent starvation of the main loop
_OCR_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# ==============================================================================
# 🎨 INTELLIGENT SHADE RESOLVER (The Monochromatic Fix)
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
        h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
        
        # Achromatic check (Low Saturation)
        if s < 0.15: return "Gray"
        
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
                "label": label, "rgb": rgb, "hex": hex_c, "lum": cls.get_luminance(rgb)
            })

        results = []

        # 2. Process each group to resolve collisions
        for base_name, items in groups.items():
            if len(items) == 1:
                # No collision: Just use the Base Name
                i = items[0]
                results.append(f"Legend Key: '{i['label']}' == {i['hex']} ({base_name})")
            else:
                # Collision: Sort by Luminance (Darkest to Lightest)
                items.sort(key=lambda x: x["lum"])
                
                count = len(items)
                for idx, item in enumerate(items):
                    # Assign relative modifier
                    if count == 2:
                        mod = "Dark" if idx == 0 else "Light"
                    elif count == 3:
                        if idx == 0: mod = "Dark"
                        elif idx == 1: mod = "Medium"
                        else: mod = "Light"
                    else:
                        # Fallback for many shades
                        intensity = int((idx / (count - 1)) * 100)
                        mod = f"Luminance-{intensity}"
                    
                    results.append(f"Legend Key: '{item['label']}' == {item['hex']} ({mod} {base_name})")

        return results

# ==============================================================================
# 🧩 SPATIAL INDEX (Performance Optimization)
# ==============================================================================
class SpatialIndex:
    def __init__(self, page_rect: fitz.Rect, cell_size: int = 50):
        self.cell_size = cell_size
        self.cols = math.ceil(page_rect.width / cell_size)
        self.rows = math.ceil(page_rect.height / cell_size)
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
            for shape in self.grid[key]:
                s_id = id(shape)
                if s_id not in candidates:
                    candidates.add(s_id)
                    results.append(shape)
        return results

# ==============================================================================
# 🧠 INTELLIGENT CLASSIFIER (OPTIMIZED)
# ==============================================================================
@traceable(name="Classify Region", run_type="chain")
def classify_region(text_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not text_items:
        return {"decision": "chart", "score": 0.0, "evidence": {}, "features": {}}

    full_text = " ".join([t.get('text', '') for t in text_items])
    tokens = full_text.split()
    total_tokens = len(tokens) if tokens else 1
    
    # --- VARIABLE 1: Alpha-Token Ratio ---
    clean_tokens = [t.strip(string.punctuation) for t in tokens]
    alpha_count = sum(1 for t in clean_tokens if t.isalpha() and len(t) > 1)
    v1_structure_signal = 1.0 - (alpha_count / total_tokens)

    # --- VARIABLE 2: Universal Entity Density ---
    score_v2_raw = 0.0
    currency_pattern = r'[$€£¥]|\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY|HKD|SGD|NZD|KRW)\b|\(\d{1,3}(?:,\d{3})*\)|\d+(?:\.\d+)?[x%]'
    score_v2_raw += len(re.findall(currency_pattern, full_text, re.IGNORECASE)) * 2.0
    safe_asset_regex = r'\b(sq\s?ft|sf|m2|psf|keys|MW|GWh|TEU|dwt|mt|bbl)\b|\b(?=.*\d)(?=.*[A-Z])[A-Z0-9]{9,12}\b'
    score_v2_raw += len(re.findall(safe_asset_regex, full_text, re.IGNORECASE)) * 3.0
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
            mid_x = (x_starts[0] + x_starts[-1]) / 2
            right_col_text = " ".join([t.get('text', '') for t in text_items if t.get('bbox', [0])[0] > mid_x])
            digit_ratio = sum(c.isdigit() for c in right_col_text) / (len(right_col_text) or 1)
            right_tokens = right_col_text.split()
            title_ratio = sum(1 for w in right_tokens if w.istitle() or w.isupper()) / (len(right_tokens) or 1)
            if digit_ratio > 0.4 or title_ratio > 0.25: v3_grid_score = 0.85 
            else: v3_grid_score = 0.20 

    # --- VARIABLE 4: Guardrails ---
    has_toc = len(re.findall(r'\.{5,}\s*\d+$', full_text, re.MULTILINE)) > 1
    header_sample = full_text[:200].lower()
    is_legal = re.search(r'^(article|section)\s+\d', header_sample)
    is_note = "note" in header_sample 
    legal_header = re.search(r'forward[- ]looking|risk factors|disclaimer', header_sample, re.IGNORECASE)
    v4_penalty = 0.0
    if has_toc: v4_penalty = 1.0 
    elif (is_legal and not is_note) or legal_header: v4_penalty = 0.8 

    final_score = ((0.35 * v1_structure_signal) + (0.30 * v3_grid_score) + (0.35 * v2_data_density)) * (1.0 - v4_penalty)
    is_financial_structure = final_score > 0.50
    
    return {
        "decision": "chart" if is_financial_structure else "general",
        "score": round(final_score, 3),
        "evidence": {
            "layout_structure": "Grid" if v3_grid_score == 1.0 else ("KV" if v3_grid_score == 0.85 else "Freeform"),
            "risk_flag": "TOC/Legal" if v4_penalty > 0.5 else "None"
        }
    }

# ==============================================================================
# 📄 PDF PARSER
# ==============================================================================
class PDFParser:
    def __init__(self):
        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = True
        pipeline_opts.do_table_structure = True
        pipeline_opts.generate_page_images = True
        pipeline_opts.generate_picture_images = True
        pipeline_opts.images_scale = 3.0 
        
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
        )
        self.ocr_engine = RapidOCR()
        self.suppressed_items: Set[int] = set()

    @traceable(name="Parse PDF File", run_type="parser")
    async def parse(self, file_path: Path) -> List[IngestedChunk]:
        if not file_path.exists(): raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"🎨 Starting Centaur ingestion for: {file_path.name}")
        
        # [BLOCKING FIX] Offload Docling to Thread
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.converter.convert, file_path)
        
        doc = result.document
        doc_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        chunks: List[IngestedChunk] = []
        self.suppressed_items = set()

        # Open Fitz once, cheaply
        with fitz.open(file_path) as pdf_doc:
            # PASS 1: VISUALS
            for item, _ in doc.iterate_items():
                label_val = getattr(item.label, 'value', 'unknown') if hasattr(item, 'label') else 'unknown'
                if label_val in ["page_header", "page_footer"]: continue
                
                if isinstance(item, TableItem):
                    chunks.extend(await self._process_complex_table(item, doc, doc_hash, file_path.name))
                    continue
                    
                if isinstance(item, PictureItem) or label_val in ["picture", "figure", "chart"]:
                    vision_chunk = await self._process_visual(item, doc, pdf_doc, doc_hash, file_path.name)
                    if vision_chunk: chunks.append(vision_chunk)
                    continue

            # PASS 2: TEXT
            for item, _ in doc.iterate_items():
                label_val = getattr(item.label, 'value', 'unknown') if hasattr(item, 'label') else 'unknown'
                if label_val in ["page_header", "page_footer"]: continue
                if id(item) in self.suppressed_items: continue
                if isinstance(item, (TableItem, PictureItem)): continue
                if label_val in ["picture", "figure", "chart"]: continue

                if hasattr(item, "text") and item.text.strip():
                    chunk = self._create_standard_chunk(item, doc, doc_hash, file_path.name)
                    if chunk: chunks.append(chunk)

        logger.info(f"✅ Extracted {len(chunks)} chunks from {file_path.name}")
        return chunks

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
        prov = item.prov[0]
        page = doc.pages[prov.page_no]
        width, height = page.size.width, page.size.height
        l, r = sorted([prov.bbox.l, prov.bbox.r])
        t, b = sorted([prov.bbox.t, prov.bbox.b])
        
        return IngestedChunk(
            chunk_id=str(uuid4()), doc_hash=doc_hash, clean_text=item.text.strip(), raw_text=item.text,
            page_number=prov.page_no, token_count=len(item.text.split()),
            primary_bbox=BoundingBox(page_number=prov.page_no, x=l/width, y=t/height, width=(r-l)/width, height=(b-t)/height),
            metadata={"source": filename, "type": "text"}
        )

    # --- HELPERS ---

    def _get_safe_clip_rect(self, bbox, page_rect):
        w, h = bbox.r - bbox.l, bbox.t - bbox.b
        pad_w, pad_h = w * 0.1, h * 0.1
        return fitz.Rect(
            max(page_rect.x0, bbox.l - pad_w),
            max(page_rect.y0, page_rect.height - bbox.t - pad_h),
            min(page_rect.x1, bbox.r + pad_w),
            min(page_rect.y1, page_rect.height - bbox.b + pad_h)
        )

    def _rect_distance(self, r1: fitz.Rect, r2: fitz.Rect) -> float:
        """Edge-to-Edge Euclidean Distance."""
        x_dist = max(0, r1.x0 - r2.x1, r2.x0 - r1.x1)
        y_dist = max(0, r1.y0 - r2.y1, r2.y0 - r1.y1)
        return math.sqrt(x_dist**2 + y_dist**2)

    def _is_text_visible(self, span: dict, page_bg=(1, 1, 1)) -> bool:
        if span.get('alpha', 1) == 0: return False
        if span.get('flags', 0) & 8: return False
        # Allow white text (labels on bars)
        return True

    def _is_valid_color(self, p: tuple) -> bool:
        """Strict Chromatic Check."""
        r, g, b = p[0], p[1], p[2]
        if r > 240 and g > 240 and b > 240: return False # White
        if r < 20 and g < 20 and b < 20: return False    # Black
        if abs(r-g) < 15 and abs(g-b) < 15: return False # Gray
        return True

    def _vector_radar_probe(self, spatial_index: SpatialIndex, text_bbox, search_rect) -> Optional[Tuple[str, int, Tuple[int,int,int]]]:
        """
        Refined Vector Radar using Spatial Index + Edge-to-Edge Distance.
        Returns: (Hex, Score, RGB)
        """
        candidates = []
        t_rect = fitz.Rect(text_bbox)
        nearby_shapes = spatial_index.query(search_rect)
        MAX_BINDING_DIST = 150.0

        for shape in nearby_shapes:
            s_rect = fitz.Rect(shape['rect'])
            if not s_rect.intersects(search_rect): continue
            if shape.get('fill') is None: continue
            
            r, g, b = shape['fill']
            rgb_255 = (int(r*255), int(g*255), int(b*255))
            
            if not self._is_valid_color(rgb_255): continue
            if min(s_rect.width, s_rect.height) < 2.0: continue

            dist = self._rect_distance(t_rect, s_rect)
            if dist > MAX_BINDING_DIST: continue
            
            score = dist 
            hex_c = '#{:02x}{:02x}{:02x}'.format(*rgb_255)
            candidates.append((hex_c, score, rgb_255))
            
        if not candidates: return None
        candidates.sort(key=lambda x: x[1])
        return candidates[0]

    # --- RASTER PROBE HELPERS ---

    def _scan_direction(self, img: Image.Image, box: tuple, direction: str, gap_tolerance: int = 15) -> Optional[Tuple[str, int, Tuple[int,int,int]]]:
        """Gap Tolerant Ray Casting."""
        try:
            safe_box = (max(0, math.floor(box[0])), max(0, math.floor(box[1])), 
                        min(img.width, math.ceil(box[2])), min(img.height, math.ceil(box[3])))
            l, t, r, b = safe_box
            w, h = img.size
            if r <= l or b <= t: return None

            def check_pixel(px, py):
                if 0 <= px < w and 0 <= py < h:
                    c = img.getpixel((px, py))
                    # Check against simple valid color logic
                    if self._is_valid_color(c[:3]):
                        return ('#{:02x}{:02x}{:02x}'.format(*c[:3]), c[:3])
                return None

            if direction == "left":
                for x in range(l - 1, max(-1, l - gap_tolerance), -1):
                    for y in range(t, b):
                        if res := check_pixel(x, y): return (res[0], l - x, res[1])
            elif direction == "right":
                for x in range(r, min(w, r + gap_tolerance)):
                    for y in range(t, b):
                        if res := check_pixel(x, y): return (res[0], x - r, res[1])
            elif direction == "bottom":
                for y in range(b, min(h, b + gap_tolerance)):
                    for x in range(l, r):
                        if res := check_pixel(x, y): return (res[0], y - b, res[1])
            return None 
        except Exception: return None

    # --- MAIN VISUAL PROCESSOR ---

    @traceable(name="Process Visual (Cascade)", run_type="chain")
    async def _process_visual(self, item: DocItem, doc, pdf_doc, doc_hash: str, filename: str) -> Optional[IngestedChunk]:
        if not item.prov: return None
        prov = item.prov[0]
        page_no = prov.page_no
        bbox = prov.bbox 

        try:
            # 1. Image Extraction (Threaded & Stateless)
            page_ref = doc.pages[page_no].image
            if not page_ref or not page_ref.pil_image: return None
            
            loop = asyncio.get_running_loop()
            
            def _crop_in_memory():
                page_img = page_ref.pil_image
                pdf_w, pdf_h = doc.pages[page_no].size.width, doc.pages[page_no].size.height
                img_w, img_h = page_img.size
                scale_x, scale_y = img_w / pdf_w, img_h / pdf_h
                
                px_l, px_t = bbox.l * scale_x, (pdf_h - bbox.t) * scale_y
                px_r, px_b = bbox.r * scale_x, (pdf_h - bbox.b) * scale_y
                
                PAD = 60
                crop_box = (int(max(0, px_l-PAD)), int(max(0, min(px_t, px_b)-PAD)), 
                            int(min(img_w, px_r+PAD)), int(min(img_h, max(px_t, px_b)+PAD)))
                
                cropped = page_img.crop(crop_box)
                out_bytes = io.BytesIO()
                # Save as PNG initially to preserve quality before analysis; VisionTool will convert if needed.
                cropped.save(out_bytes, format='PNG') 
                return out_bytes.getvalue(), scale_x, scale_y, crop_box[0], crop_box[1]

            # [ASYNC] Offload Crop
            img_bytes, scale_x, scale_y, crop_off_x, crop_off_y = await loop.run_in_executor(None, _crop_in_memory)
            
            # 2. Vector Probe Setup
            fitz_page = pdf_doc.load_page(page_no - 1)
            clip_rect = self._get_safe_clip_rect(bbox, fitz_page.rect)
            
            # The "Periscope": Define a search area LARGER than the visual to find legends
            SEARCH_PAD = 150
            search_rect = fitz.Rect(
                max(0, clip_rect.x0 - SEARCH_PAD), 
                max(0, clip_rect.y0 - SEARCH_PAD),
                min(fitz_page.rect.width, clip_rect.x1 + SEARCH_PAD),
                min(fitz_page.rect.height, clip_rect.y1 + SEARCH_PAD)
            )
            
            # [OPTIMIZATION] Build Spatial Index
            spatial_index = SpatialIndex(fitz_page.rect)
            for d in fitz_page.get_drawings(): spatial_index.insert(d)
            
            vector_data = fitz_page.get_text("dict", clip=clip_rect, flags=fitz.TEXT_PRESERVE_IMAGES)
            text_items = []
            method = "vector"
            
            for block in vector_data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if self._is_text_visible(span):
                            bx0, by0, bx1, by1 = span["bbox"]
                            px0, py0 = (bx0 * scale_x) - crop_off_x, (by0 * scale_y) - crop_off_y
                            px1, py1 = (bx1 * scale_x) - crop_off_x, (by1 * scale_y) - crop_off_y
                            text_items.append({
                                'text': span["text"], 'bbox': [px0, py0, px1, py1],
                                'raw_bbox': fitz.Rect(bx0, by0, bx1, by1)
                            })

            # 3. Raster Fallback (Threaded) [CRITICAL SAFETY NET PRESERVED]
            if len(text_items) < 5:
                method = "raster"
                # [ASYNC] Offload OCR
                def _ocr_task(): return self.ocr_engine(np.array(Image.open(io.BytesIO(img_bytes))))
                ocr_results, _ = await loop.run_in_executor(_OCR_EXECUTOR, _ocr_task)
                if ocr_results:
                    for line in ocr_results:
                        poly, text, _ = line
                        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
                        text_items.append({'text': text, 'bbox': [min(xs), min(ys), max(xs), max(ys)]})

            # 4. Semantic Binding (THE FIX: Separate Legends vs Data)
            raw_legend_bindings = []
            data_point_hints = []
            
            for t_item in text_items:
                if not t_item['text'].strip(): continue
                text_clean = t_item['text'].strip()
                
                # [LOGIC SPLIT]
                # Is it a number? (Data Point)
                is_numeric = any(c.isdigit() for c in text_clean) and len(text_clean) < 10
                
                candidates = []
                
                if method == "vector" and 'raw_bbox' in t_item:
                    match = self._vector_radar_probe(spatial_index, t_item['raw_bbox'], search_rect)
                    if match: candidates.append(("VectorMatch", match[1], match[2], match[0]))
                else:
                    l, t, r, b = t_item['bbox']
                    PROBE_DIST = int(60 * scale_x)
                    for d in [("left", (l-PROBE_DIST, t, l, b)), ("right", (r, t, r+PROBE_DIST, b))]:
                        res = self._scan_direction(Image.open(io.BytesIO(img_bytes)), d[1], d[0])
                        if res: candidates.append((d[0], res[1], res[2], res[0]))

                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    best = candidates[0]
                    # Hex = best[3], RGB = best[2]
                    
                    if is_numeric:
                        # It's a Data Point -> Store as "Hint" not "Legend"
                        # "Value '20766' is on Dark Red"
                        # We use the raw Hex for this, LLM matches it to the Resolved Legend
                        data_point_hints.append(f"Data Point: '{text_clean}' is on/near {best[3]}")
                    else:
                        # It's Text -> Potential Legend
                        raw_legend_bindings.append((text_clean, best[2], best[3]))

            # 5. Resolve Shades (Only for Legends)
            resolved_legends = ColorResolver.resolve_names(raw_legend_bindings)

            # 6. Classify & Analyze
            classification_result = classify_region(text_items)
            mode = classification_result["decision"]
            item_label = getattr(item, "label", None)
            if item_label is None and isinstance(item, dict): item_label = item.get("label")
            label_str = item_label.value.lower() if hasattr(item_label, "value") else str(item_label).lower()
            if any(x in label_str for x in ["chart", "figure"]) and text_items: mode = "chart"

            full_context = (
                f"DETECTED LEGENDS (Definitions):\n{list(set(resolved_legends))}\n\n"
                f"DATA POINT LOCATION HINTS:\n{data_point_hints[:30]}\n" # Limit to avoid context overflow
                f"LAYOUT: {classification_result['evidence']['layout_structure']}"
            )

            description = await vision_tool.analyze(
                image_data=img_bytes, 
                mode=mode, 
                context=full_context
            )
            
            chunk_id = str(uuid4())
            caption_str = ""
            if hasattr(item, "captions") and item.captions:
                caption_str = " | ".join([c.text for c in item.captions])

            return IngestedChunk(
                chunk_id=chunk_id, doc_hash=doc_hash,
                clean_text=f"[{mode.upper()} ANALYSIS]\nTitle: {caption_str}\n{description}", 
                raw_text=description, page_number=page_no, token_count=len(description.split()),
                metadata={"source": filename, "type": "visual", "mode": mode}
            )

        except Exception as e:
            logger.error(f"Visual processing failed: {e}")
            return None