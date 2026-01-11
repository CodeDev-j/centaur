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

    def _generate_dual_view_context(self, ocr_results: List[Any], page_height: int) -> str:
        """
        Creates a 'Holographic' Text Representation using strict geometric grouping.
        Uses TOLERANCE-based clustering (not buckets) to handle slight misalignments.
        Adds Normalized Y-Coordinates (0-1000) for the Adversarial Prompt.
        """
        if not ocr_results: return ""

        # Normalize Input & Calculate Centroids
        blocks = []
        for item in ocr_results:
            poly, text, conf = item
            if not text.strip(): continue
            
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            
            # Normalize Y (0 = Top, 1000 = Bottom) for the Prompt's "Spatial Map"
            y_norm = int((min(ys) / page_height) * 1000)
            
            blocks.append({
                "cx": cx, "cy": cy, "y_norm": y_norm, 
                "text": text.strip(), "box": poly
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
        col_view_lines = []
        for idx, col in enumerate(col_groups):
            col.sort(key=lambda b: b['cy'])
            
            if col:
                # LOGIC FIX: Identify the Anchor
                # In a standard Top-Down Y coordinate system (0=Top, 1000=Bottom),
                # the Axis Label is typically the element with the HIGHEST Y-value (lowest on page).
                anchor_idx = len(col) - 1
                
                # Formatted String Construction
                items = []
                for i, b in enumerate(col):
                    # Apply Tags (THE FIX)
                    if i == anchor_idx:
                        tag = "ANCHOR"
                    else:
                        tag = "FLOATER"
                    
                    # Construct the entry
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
        Uses ColorResolver to map text labels to nearby colors.
        Critical for 'Stacked Bar' charts in monochrome palettes.
        """
        if not ocr_results: return ""
        
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            candidates = []
            
            for item in ocr_results:
                box, text, _ = item
                text = text.strip()
                if not text or len(text) < 3 or any(c.isdigit() for c in text): continue 
                
                # Check scan area to the LEFT of the text (standard legend format: [Box] Label)
                xs, ys = [p[0] for p in box], [p[1] for p in box]
                l, t, r, b = min(xs), min(ys), max(xs), max(ys)
                
                # Scan a small area (40px) to the left
                scan_box = (max(0, l-40), t, l, b)
                
                colors = img.crop(scan_box).getcolors(maxcolors=256)
                if colors:
                    valid = [c[1] for c in colors if not (c[1][0]>240 and c[1][1]>240) and not (c[1][0]<20)]
                    if valid: candidates.append((text, max(valid, key=lambda x: x[0])))

            if not candidates: return ""
            return "\n".join(ColorResolver.resolve_palette(candidates))
        except Exception:
            return ""

    def _refine_grounding(self, facts: List[dict], ocr_results: List[Any], img_w: int, img_h: int) -> List[dict]:
        """
        THE SNAPPER (REFINED): Corrects VLM boxes using exact OCR coordinates.
        Uses 'Region Gating' to prevent cross-chart teleportation.
        """
        if not ocr_results: return facts

        for fact in facts:
            target = str(fact.get("value", "")).strip()
            if not target: continue
            
            # 1. ESTABLISH ANCHOR: Get VLM's rough estimate 
            vlm_box = fact.get("grounding", {}).get("box_2d")
            
            # Unpack specific coordinates for the Hard Gate
            vlm_ymin, vlm_xmin, vlm_ymax, vlm_xmax = vlm_box if vlm_box else (0,0,1000,1000)
            vlm_center_y = (vlm_ymin + vlm_ymax) / 2
            vlm_center_x = (vlm_xmin + vlm_xmax) / 2

            best_match = None
            best_score = 0.0

            for item in ocr_results:
                ocr_box, ocr_text, _ = item
                ocr_text = ocr_text.strip()
                if not ocr_text: continue

                # 2. STRING MATCH SCORE (0.0 - 1.0)
                seq_ratio = difflib.SequenceMatcher(None, target, ocr_text).ratio()
                
                # Calculate centroid of OCR box in 0-1000 scale
                xs, ys = [p[0] for p in ocr_box], [p[1] for p in ocr_box]
                ocr_cx = (sum(xs)/len(xs) / img_w) * 1000
                ocr_cy = (sum(ys)/len(ys) / img_h) * 1000

                # 3. REGION GATING (The Fix)
                # Hard Gate: If OCR match is > 300 units away vertically (30% of page),
                # it is likely "Teleportation" (e.g. Footer to Header). REJECT.
                if abs(ocr_cy - vlm_center_y) > 300:
                    continue
                
                # Soft Decay: Euclidean distance preferred for tie-breaking
                dist = math.sqrt((ocr_cx - vlm_center_x)**2 + (ocr_cy - vlm_center_y)**2)
                spatial_score = max(0.0, 1.0 - (dist / 300.0))

                # Combined Score
                total_score = seq_ratio * 0.7 + spatial_score * 0.3
                
                if seq_ratio > 0.85 and total_score > best_score:
                    best_score = total_score
                    best_match = ocr_box

            if best_match:
                xs, ys = [p[0] for p in best_match], [p[1] for p in best_match]
                fact["grounding"]["box_2d"] = [
                    int((min(ys)/img_h)*1000), int((min(xs)/img_w)*1000),
                    int((max(ys)/img_h)*1000), int((max(xs)/img_w)*1000)
                ]
                fact["grounding_source"] = "exact_ocr_snap"
            else:
                fact["grounding_source"] = "vlm_estimate"
        return facts

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

                # B. EXTRACTION (The "Sniper")
                ocr_context = self._generate_dual_view_context(ocr_results, h_px)
                color_hints = self._extract_legend_hints(img_bytes, ocr_results)
                full_context = f"[VISUAL HINTS - LEGEND KEYS]\n{color_hints}\n\n{ocr_context}" if color_hints else ocr_context
                
                analysis = await vision_tool.analyze_full_page(
                    image_data=img_bytes, 
                    ocr_context=full_context,
                    layout_hint=layout_map  # <--- INJECTING THE HINT
                )
                
                if analysis:
                    refined = self._refine_grounding([f.model_dump() for f in analysis.facts], ocr_results, w_px, h_px)
                    blob_path = await BlobDriver.save_image(img_bytes, "layouts", f"{uuid4()}.png")
                    
                    chunks.append(IngestedChunk(
                        chunk_id=str(uuid4()), doc_hash=doc_hash,
                        clean_text=f"### {analysis.title}\n{analysis.summary}",
                        raw_text=analysis.summary, page_number=page_idx+1,
                        metadata={
                            "source": file_path.name, 
                            "type": "visual_page", 
                            "blob_path": blob_path, 
                            "facts": refined,
                            "audit_log": analysis.audit_log
                        }
                    ))
        
        logger.info(f"✅ Extracted {len(chunks)} chunks from {file_path.name}")
        return chunks

    async def _process_complex_table(self, item, doc, doc_hash, filename) -> List[IngestedChunk]:
        """
        Docling Table Processor (Restored).
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