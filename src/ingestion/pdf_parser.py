import logging
import hashlib
import asyncio
import io
import math
import collections
import colorsys
import difflib
import numpy as np
import base64
import re
from uuid import uuid4
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor

# Third-party
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import TableItem
from rapidocr_onnxruntime import RapidOCR
from langsmith import traceable

# Internal
from src.schemas.documents import IngestedChunk
from src.storage.blob_driver import BlobDriver
from src.tools.vision import vision_tool
from src.tools.layout_scanner import layout_scanner

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# ProcessPool for CPU-heavy tasks to bypass GIL. 
# 4 workers is usually optimal for standard 8-core machines.
_PROCESS_EXECUTOR = ProcessPoolExecutor(max_workers=4)

# --- WORKER FUNCTIONS ---
def _render_and_ocr_worker(file_path: str, page_num: int) -> Tuple[Optional[bytes], List[Any]]:
    try:
        with fitz.open(file_path) as doc:
            if page_num >= len(doc): return None, []
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200) 
            img_bytes = pix.tobytes("png")
            
            ocr = RapidOCR()
            # Returns list of [ [[x,y]...], text, score ]
            ocr_results, _ = ocr(img_bytes)
            
            return img_bytes, ocr_results
    except Exception as e:
        print(f"Worker Error on p{page_num + 1}: {e}")
        return None, []

# ==============================================================================
# 🎨 SMART COLOR EXTRACTOR
# ==============================================================================
class SmartColorExtractor:
    """
    Advanced color picker that performs Proximity-Weighted Omni-Scanning.
    It prioritizes saturation and penalizes physical distance to solve
    'Sandwich' legends and 'Stacked' labels.
    """
    
    @staticmethod
    def get_color_with_distance(image_crop: Image.Image, scan_direction: str) -> Optional[Dict[str, Any]]:
        """
        Analyzes a crop to find a dominant color AND its distance from the scan edge.
        scan_direction: 'left', 'right', 'up' (affects how we measure distance)
        """
        # 1. Quantize to reduce noise/artifacts
        try:
            quantized = image_crop.quantize(colors=32, method=2)
            colors = quantized.convert("RGB").getcolors(maxcolors=256)
        except Exception:
            return None

        if not colors: return None
        
        # 2. Find the "Best" Color based on Saturation (Vibrancy)
        best_candidate = None
        best_score = -1
        
        for count, rgb in colors:
            r, g, b = rgb
            h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
            
            # Filter Achromatic (Gray/White) and Extreme Luminance (Black Text/Paper White)
            if s < 0.10 or l > 0.95 or l < 0.05: continue
            
            # Score = Count * Saturation (Favor vibrant legend keys over background noise)
            score = count * (s * 2.0)
            if score > best_score:
                best_score = score
                best_candidate = rgb

        if not best_candidate: return None

        # 3. Calculate "Gap Distance" (Crucial for Tie-Breaking)
        # We scan the crop to find WHERE this color first appears relative to the text edge.
        width, height = image_crop.size
        pixels = image_crop.load()
        min_dist = 1000 # Start high
        
        check_rgb = best_candidate
        
        # Helper: Euclidean distance tolerance for color matching
        def is_match(p):
            return sum(abs(p[i] - check_rgb[i]) for i in range(3)) < 30

        if scan_direction == 'left':
            # Text is at X=RightEdge. Scan backwards from Width->0
            for x in range(width - 1, -1, -1):
                for y in range(0, height, 5): # Sparsely scan Y for speed
                    if is_match(pixels[x, y]):
                        dist = width - x
                        if dist < min_dist: min_dist = dist
                        
        elif scan_direction == 'right':
            # Text is at X=0. Scan forward from 0->Width
            for x in range(0, width):
                for y in range(0, height, 5):
                    if is_match(pixels[x, y]):
                        dist = x
                        if dist < min_dist: min_dist = dist

        elif scan_direction == 'up':
            # Text is at Y=Height. Scan upwards from Height->0
            for y in range(height - 1, -1, -1):
                for x in range(0, width, 5):
                    if is_match(pixels[x, y]):
                        dist = height - y
                        if dist < min_dist: min_dist = dist

        # 4. Return Data
        return {
            "rgb": best_candidate,
            "saturation_score": best_score,
            "gap_distance": min_dist if min_dist != 1000 else 40 # Cap at max scan depth
        }

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
    def resolve_palette(cls, bindings: List[Tuple[str, Tuple[int, int, int]]]) -> List[str]:
        """
        Input: List of (TextLabel, RGB_Tuple)
        Output: List of strings "Hint: 'Label' is Dark Red (#hex)"
        """
        # 1. Group by Base Hue
        groups = collections.defaultdict(list)
        for label, rgb in bindings:
            hex_c = '#{:02x}{:02x}{:02x}'.format(*rgb)
            base = cls.get_base_hue(rgb)
            lum = cls.get_luminance(rgb)
            groups[base].append({"label": label, "hex": hex_c, "lum": lum})

        results = []
        # 2. Process each group to resolve collisions (Dark/Light)
        for base_name, items in groups.items():
            if len(items) == 1:
                i = items[0]
                results.append(f"- '{i['label']}' matches {base_name} ({i['hex']})")
            else:
                # Sort by Luminance (Darkest to Lightest)
                items.sort(key=lambda x: x["lum"])
                count = len(items)
                for idx, item in enumerate(items):
                    mod = f"Luminance-{int((idx / (count - 1)) * 100)} " if count > 2 else ("Dark " if idx == 0 else "Light ")
                    results.append(f"- '{item['label']}' matches {mod}{base_name} ({item['hex']})")
        return results

# ==============================================================================
# 📄 MAIN PDF PARSER
# ==============================================================================
class PDFParser:
    # Compile regex once at class level
    # Matches pure numbers, years (2024), percentages (15%), currency ($50), or simple floats (1.5)
    _IGNORE_LEGEND_PATTERN = re.compile(r'^(\d{4}|Q[1-4]|\d+(\.\d+)?%?|[\$€£]\d+.*)$', re.IGNORECASE)

    def __init__(self):
        # Docling setup for pure text/table extraction
        # Note: We disable image generation here to save CPU (we do it in fitz)
        opts = PdfPipelineOptions()
        opts.do_ocr = True 
        opts.do_table_structure = True
        opts.generate_page_images = False 
        
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

    def _extract_dominant_colors(self, img_arr: np.ndarray, box: List[float]) -> Optional[str]:
        """
        Inward Raster Probe: Scans core pixels to handle thin waterfall bars
        and allows dark mode charts.
        """
        try:
            h_img, w_img, _ = img_arr.shape
            l, t, r, b = map(int, box)
            
            # Find the center 50% of the box to avoid background bleed
            w_box, h_box = r - l, b - t
            l_core = l + int(w_box * 0.25)
            t_core = t + int(h_box * 0.25)
            r_core = r - int(w_box * 0.25)
            b_core = b - int(h_box * 0.25)
            
            roi = img_arr[t_core:b_core, l_core:r_core]
            if roi.size == 0: return None
            
            pixels = roi.reshape(-1, 3)
            
            # Filter only if variance is low AND luminance is HIGH (Anti-aliasing noise)
            pixel_var = np.ptp(pixels, axis=1)
            pixel_lum = np.mean(pixels, axis=1)
            
            is_noise_gray = (pixel_var < 15) & (pixel_lum > 200) # Only filter light/white-ish gray
            is_white = (pixels > 240).all(axis=1)
            is_black = (pixels < 10).all(axis=1) # Strict black filter
            
            valid_mask = ~(is_white | is_noise_gray | is_black)
            valid_pixels = pixels[valid_mask]
            
            if valid_pixels.size == 0: return None
            
            # Get most frequent valid color
            # Quantize to 10s to group similar shades
            quantized = (valid_pixels // 10) * 10
            unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
            dominant = unique_colors[np.argmax(counts)]
            
            rgb = tuple(int(x) for x in dominant)
            return '#{:02x}{:02x}{:02x}'.format(*rgb)
        except Exception:
            return None

    def _generate_dual_view_context(
        self, 
        ocr_results: List[Any], 
        page_height: int, 
        img_bytes: bytes = None,
        layout_hint: Any = None  # <--- [UPDATE 1] Accept the hint in signature
    ) -> str:
        """
        Creates a 'Holographic' Text Representation using strict geometric grouping.
        Uses TOLERANCE-based clustering (not buckets) to handle slight misalignments.
        Adds Normalized Y-Coordinates (0-1000) for the Adversarial Prompt.
        [UPDATED]: Accepts img_bytes to inject color hints.
        """
        if not ocr_results: return ""

        # Prepare image for Raster Probe if available
        img_arr = None
        if img_bytes:
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_arr = np.array(img)
            except Exception: pass

        blocks = []
        for item in ocr_results:
            poly, text, conf = item
            if not text.strip(): continue
            
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            
            # Normalize Y (0 = Top, 1000 = Bottom) for the Prompt's "Spatial Map"
            y_norm = int((min(ys) / page_height) * 1000)
            
            # [NEW] Raster Probe: Inject Color Hint if it looks like a Data Point
            color_tag = ""
            if img_arr is not None and (any(c.isdigit() for c in text) or "$" in text):
                # Convert poly to bounding box [l, t, r, b]
                l, t, r, b = min(xs), min(ys), max(xs), max(ys)
                hex_color = self._extract_dominant_colors(img_arr, [l, t, r, b])
                if hex_color:
                    color_tag = f" [{hex_color}]"

            blocks.append({
                "cx": cx, "cy": cy, "y_norm": y_norm, 
                "text": text.strip() + color_tag,  # Append hint
                "box": poly
            })

        # --- VIEW 1: ROW GROUPING (Narrative Flow) ---
        # Sort by Y-Centroid first
        rows_sorted = sorted(blocks, key=lambda b: b['cy'])
        row_groups = []
        if rows_sorted:
            current_row = [rows_sorted[0]]
            current_ref_y = rows_sorted[0]['cy']
            Y_TOLERANCE = 15  # Pixels

            for block in rows_sorted[1:]:
                if abs(block['cy'] - current_ref_y) <= Y_TOLERANCE:
                    current_row.append(block)
                else:
                    row_groups.append(current_row)
                    current_row = [block]
                    current_ref_y = block['cy']
            row_groups.append(current_row)

        # Format Rows: Sort items within row by X
        row_view_lines = []
        for row in row_groups:
            row.sort(key=lambda b: b['cx'])
            # We include the [Y=...] tag for the first item in the row to ground the LLM
            line_y = row[0]['y_norm']
            line_text = " ".join([b['text'] for b in row])
            row_view_lines.append(f"[{line_y:03d}] {line_text}")

        # --- VIEW 2: COLUMN GROUPING (Vertical Structure) ---
        
        # [UPDATE 2] The Logic Gate for Mekko/Variable Charts
        disable_column_view = False
        if layout_hint and getattr(layout_hint, 'has_charts', False):
            # If ANY chart on the page is variable width, disable strict column clustering
            # to prevent "Poisoning" the VLM with bad grid data.
            if any(getattr(c, 'is_variable_width', False) for c in layout_hint.charts):
                disable_column_view = True

        col_view_lines = []
        
        if disable_column_view:
             col_view_lines.append(">> COLUMN VIEW DISABLED: Variable-Width/Complex Layout Detected. Trust Visual Alignment. <<")
        else:
            # Sort by X-Centroid first
            cols_sorted = sorted(blocks, key=lambda b: b['cx'])
            col_groups = []
            if cols_sorted:
                current_col = [cols_sorted[0]]
                current_ref_x = cols_sorted[0]['cx']
                X_TOLERANCE = 25 # Slightly looser for columns

                for block in cols_sorted[1:]:
                    if abs(block['cx'] - current_ref_x) <= X_TOLERANCE:
                        current_col.append(block)
                    else:
                        col_groups.append(current_col)
                        current_col = [block]
                        current_ref_x = block['cx']
                col_groups.append(current_col)

            # Format Columns: Sort items within column by Y
            for idx, col in enumerate(col_groups):
                col.sort(key=lambda b: b['cy'])
                
                if col:
                    # Identify the Anchor
                    anchor_idx = len(col) - 1
                    items = []
                    for i, b in enumerate(col):
                        # Apply Tags
                        tag = "ANCHOR" if i == anchor_idx else "FLOATER"
                        items.append(f"[{b['y_norm']:03d}|{tag}] {b['text']}")
                    
                    col_content = " | ".join(items)
                    col_view_lines.append(f"Col {idx+1}: {col_content}")

        return (
            f"=== SPATIAL MAP (Coordinate System: Y=0 Top, Y=1000 Bottom) ===\n"
            f"--- VIEW 1: HORIZONTAL READING (ROWS) ---\n" + "\n".join(row_view_lines) + "\n\n"
            f"--- VIEW 2: VERTICAL SCANNING (COLUMNS) ---\n" + "\n".join(col_view_lines)
        )

    def _extract_legend_hints(self, img_bytes: bytes, ocr_results: List[Any]) -> str:
        """
        Uses SmartColorExtractor to map text labels to nearby colors.
        Scans Left, Right, and Up, using Proximity + Saturation to pick the correct key.
        """
        if not ocr_results: return ""
        
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = img.size
            candidates = []
            
            for item in ocr_results:
                box, text, _ = item
                text = text.strip()
                
                # Filter out numbers and short text (keeps labels like 'Smart-clothing')
                # ADJUSTED: < 2 to keep 'US', 'UK', 'EU'
                if not text or len(text) < 2 or self._IGNORE_LEGEND_PATTERN.match(text): continue 
                
                # Check scan area to the LEFT/RIGHT/UP of the text
                xs, ys = [p[0] for p in box], [p[1] for p in box]
                l, t, r, b = min(xs), min(ys), max(xs), max(ys)
                
                # --- DEFINE SCAN ZONES ---
                scan_depth = 40 # Pixels
                
                # 1. Left (Standard Legend format: [Box] Label)
                box_l = (max(0, l - scan_depth), t, l, b)
                # 2. Right (Suffix Legend format: Label [Box])
                box_r = (r, t, min(w, r + scan_depth), b)
                # 3. Up (Stacked/Mekko labels below bars)
                box_u = (l, max(0, t - scan_depth), r, t)

                results = []
                
                # --- RUN OMNI-SCAN ---
                # Check Left
                res_l = SmartColorExtractor.get_color_with_distance(img.crop(box_l), 'left')
                if res_l: results.append(('left', res_l))
                
                # Check Right
                res_r = SmartColorExtractor.get_color_with_distance(img.crop(box_r), 'right')
                if res_r: results.append(('right', res_r))
                
                # Check Up
                res_u = SmartColorExtractor.get_color_with_distance(img.crop(box_u), 'up')
                if res_u: results.append(('up', res_u))

                if not results: continue

                # --- THE DECISION LOGIC (Proximity Wins) ---
                # Calculate Confidence: Saturation / (Distance^1.5)
                # The exponent 1.5 heavily penalizes distance, ensuring we don't 
                # snap to a neighbor's color key.
                best_match = None
                max_confidence = 0

                for direction, data in results:
                    # Penalize distance. A gap of 5px is valid. 35px is likely noise.
                    dist_penalty = max(1, data['gap_distance'])
                    confidence = data['saturation_score'] / (dist_penalty ** 1.5) 
                    
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_match = data['rgb']
                
                if best_match:
                    candidates.append((text, best_match))
 
            if not candidates: return ""
            return "\n".join(ColorResolver.resolve_palette(candidates))
        except Exception as e:
            logger.warning(f"Legend Hint Error: {e}")
            return ""

    def _apply_snapper_logic(self, target_val: str, vlm_box: List[int], ocr_results: List[Any], img_w: int, img_h: int) -> Tuple[List[int], str]:
        """
        The Financial Snapper: Strict magnitude guards and glyph normalization.
        """
        if not target_val: return (vlm_box, "vlm_default")
        
        # Clean inputs
        clean_target = re.sub(r'[^\d\w]', '', str(target_val).upper())
        
        # Strict Vertical Gate (10% of page height)
        VERTICAL_GATE = 100 

        # Estimate VLM Centroid
        vlm_ymin, vlm_xmin, vlm_ymax, vlm_xmax = vlm_box if vlm_box else (0,0,1000,1000)
        vlm_cy = (vlm_ymin + vlm_ymax) / 2

        for item in ocr_results:
            ocr_box, ocr_text, _ = item
            clean_ocr = re.sub(r'[^\d\w]', '', ocr_text.upper())
            if not clean_ocr: continue

            # 1. MAGNITUDE GUARD: Lengths must match for numerics
            # This blocks "100" snapping to "1000"
            if any(c.isdigit() for c in clean_target) and abs(len(clean_target) - len(clean_ocr)) > 0:
                continue

            # 2. GLYPH NORMALIZATION: Fix common OCR errors
            norm_t = clean_target.translate(str.maketrans("OSI", "051"))
            norm_o = clean_ocr.translate(str.maketrans("OSI", "051"))
            
            # 3. SCORING
            if norm_t == norm_o:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, norm_t, norm_o).ratio()
            
            # 4. Check Gate
            ys = [p[1] for p in ocr_box]
            ocr_cy = (sum(ys)/len(ys) / img_h) * 1000
            
            if score > 0.85 and abs(ocr_cy - vlm_cy) < VERTICAL_GATE:
                # Return new box
                xs = [p[0] for p in ocr_box]
                return ([
                    int((min(ys)/img_h)*1000), int((min(xs)/img_w)*1000),
                    int((max(ys)/img_h)*1000), int((max(xs)/img_w)*1000)
                ], "exact_ocr_snap")
                
        return (vlm_box, "vlm_estimate")

    def _refine_grounding(self, analysis_dict: dict, ocr_results: List[Any], img_w: int, img_h: int) -> dict:
        """
        Traverses Nested Structure and applies The Snapper
        """
        # 1. Process Quantitative Metrics
        if "metrics" in analysis_dict:
            for series in analysis_dict["metrics"]:
                for point in series["data_points"]:
                    target = str(point.get("numeric_value", ""))
                    if not target or target == "None": 
                        target = point.get("label", "")
                    
                    current_box = point.get("grounding", {}).get("box_2d", [0,0,0,0])
                    new_box, source = self._apply_snapper_logic(target, current_box, ocr_results, img_w, img_h)
                    
                    point["grounding"]["box_2d"] = new_box
                    point["grounding_source"] = source

        # 2. Process Qualitative Insights
        if "insights" in analysis_dict:
            for insight in analysis_dict["insights"]:
                target = insight.get("content", "")[:30] # First 30 chars
                current_box = insight.get("grounding", {}).get("box_2d", [0,0,0,0])
                new_box, source = self._apply_snapper_logic(target, current_box, ocr_results, img_w, img_h)
                
                insight["grounding"]["box_2d"] = new_box
                insight["grounding_source"] = source

        return analysis_dict

    @traceable(name="Parse PDF File", run_type="parser")
    async def parse(self, file_path: Path) -> List[IngestedChunk]:
        if not file_path.exists(): raise FileNotFoundError(f"{file_path}")
        logger.info(f"🎨 Starting Centaur 2.0 Ingestion: {file_path.name}")
        loop = asyncio.get_running_loop()
        
        with open(file_path, "rb") as f:
            doc_hash = hashlib.sha256(f.read()).hexdigest()
        
        chunks = []
        
        # 1. DOCLING PASS (Tables)
        # We run this FIRST to get high-quality table data
        try:
            result = await loop.run_in_executor(None, self.converter.convert, file_path)
            doc = result.document
            if doc:
                for item, _ in doc.iterate_items():
                    if isinstance(item, TableItem):
                        df_string = item.export_to_dataframe(doc).to_string()
                        # Only process substantial tables
                        if len(df_string) > 100:
                            table_chunks = await self._process_complex_table(item, doc, doc_hash, file_path.name)
                            chunks.extend(table_chunks)
        except Exception as e:
            logger.error(f"Docling failed: {e}")

        # 2. VLM PASS (Visual Charts & Waterfalls)
        with fitz.open(file_path) as pdf_doc:
            for page_idx in range(len(pdf_doc)):
                img_bytes, ocr_results = await loop.run_in_executor(
                    _PROCESS_EXECUTOR, _render_and_ocr_worker, str(file_path), page_idx
                )
                if not img_bytes: continue
                
                with Image.open(io.BytesIO(img_bytes)) as pil_img: w_px, h_px = pil_img.size

                # A. LAYOUT SCAN (The "Scout")
                try:
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    layout_map = await layout_scanner.scan(img_b64)
                except Exception as e:
                    logger.warning(f"Layout scan failed for p{page_idx+1}: {e}")
                    layout_map = None

                # [UPDATE] Pass img_bytes to generate context with color hints
                # [UPDATE 2] Pass layout_map to disable column view for Mekko
                ocr_context = self._generate_dual_view_context(
                    ocr_results, 
                    h_px, 
                    img_bytes, 
                    layout_hint=layout_map # <--- INJECTING THE HINT
                )
                color_hints = self._extract_legend_hints(img_bytes, ocr_results)
                full_context = f"[VISUAL HINTS - LEGEND KEYS]\n{color_hints}\n\n{ocr_context}" if color_hints else ocr_context
                
                analysis = await vision_tool.analyze_full_page(
                    image_data=img_bytes, 
                    ocr_context=full_context,
                    layout_hint=layout_map
                )
                
                if analysis:
                    analysis_dict = analysis.model_dump()
                    refined_dict = self._refine_grounding(analysis_dict, ocr_results, w_px, h_px)
                    
                    blob_path = await BlobDriver.save_image(img_bytes, "layouts", f"{uuid4()}.png")
                    
                    chunks.append(IngestedChunk(
                        chunk_id=str(uuid4()), doc_hash=doc_hash,
                        clean_text=f"### {analysis.title}\n{analysis.summary}",
                        raw_text=analysis.summary, page_number=page_idx+1,
                        metadata={
                            "source": file_path.name, 
                            "type": "visual_page", 
                            "blob_path": blob_path, 
                            "structured_data": refined_dict,
                            "audit_log": analysis.audit_log
                        }
                    ))
        
        logger.info(f"✅ Extracted {len(chunks)} chunks from {file_path.name}")
        return chunks

    async def _process_complex_table(self, item, doc, doc_hash, filename) -> List[IngestedChunk]:
        """
        Docling Table Processor.
        Handles dense numeric grids better than VLM.
        """
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