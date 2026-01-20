import logging
import hashlib
import math
import collections
import re
import asyncio
import io
import base64
import colorsys
import numpy as np
from uuid import uuid4
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor

# Third-party
import fitz  # PyMuPDF
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import TableItem
from langsmith import traceable

# Internal
from src.schemas.documents import IngestedChunk
from src.storage.blob_driver import BlobDriver
from src.tools.vision import vision_tool
from src.tools.layout_scanner import layout_scanner
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
    def resolve_names(cls, bindings: List[Dict[str, Any]]) -> List[str]:
        """
        Handles dictionary inputs with 'confirmed' flag.
        Input: List of {'text': str, 'rgb': tuple, 'hex': str, 'confirmed': bool}
        Output: List of strings "Legend Key: 'Label' == Hex (Semantic Name)"
        """
        # 1. Group by Base Hue
        groups = collections.defaultdict(list)
        for item in bindings:
            base = cls.get_base_hue(item['rgb'])
            groups[base].append({
                "label": item['text'], 
                "rgb": item['rgb'], 
                "hex": item['hex'], 
                "lum": cls.get_luminance(item['rgb']),
                "confirmed": item['confirmed']
            })

        results = []

        # 2. Process each group to resolve collisions
        for base_name, items in groups.items():
            if len(items) == 1:
                # No collision: Just use the Base Name
                i = items[0]
                prefix = "[CONFIRMED LEGEND]" if i['confirmed'] else "[POSSIBLE LEGEND]"
                results.append(f"{prefix} '{i['label']}' == {i['hex']} ({base_name})")
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
                    
                    prefix = "[CONFIRMED LEGEND]" if item['confirmed'] else "[POSSIBLE LEGEND]"
                    results.append(f"{prefix} '{item['label']}' == {item['hex']} ({mod} {base_name})")

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
# 📄 PDF PARSER
# ==============================================================================
class PDFParser:
    def __init__(self):
        # 1. OCR Engine
        self.ocr_engine = RapidOCR()
        
        # 2. Docling Setup (Kept ONLY for Tables)
        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = True 
        pipeline_opts.do_table_structure = True
        pipeline_opts.images_scale = 3.0 
        
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
        )

    # --- VECTOR MATH HELPERS (Preserved) ---
    def _is_valid_color(self, p: tuple) -> bool:
        r, g, b = p[0], p[1], p[2]
        if r > 240 and g > 240 and b > 240: return False # White
        if r < 20 and g < 20 and b < 20: return False    # Black
        if abs(r-g) < 15 and abs(g-b) < 15: return False # Gray
        return True

    def _rect_distance(self, r1: fitz.Rect, r2: fitz.Rect) -> float:
        x_dist = max(0, r1.x0 - r2.x1, r2.x0 - r1.x1)
        y_dist = max(0, r1.y0 - r2.y1, r2.y0 - r1.y1)
        return math.sqrt(x_dist**2 + y_dist**2)

    def _extract_text_spatial(self, img_np: np.ndarray) -> str:
        """Runs OCR on the full page image to get the Spatial Grid."""
        try:
            result, _ = self.ocr_engine(img_np)
            if not result: return ""
            spatial_text = []
            for res in result:
                box, text, _ = res
                x, y = int(box[0][0]), int(box[0][1])
                spatial_text.append(f"[{y}, {x}] {text}")
            return "\n".join(spatial_text)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    def _generate_vector_hints(self, page: fitz.Page, layout: Optional[Any] = None) -> str:
        """
        THE GOLD MINE: Uses your preserved SpatialIndex logic to find
        Legend matches on the WHOLE page and returns them as text hints.
        """
        try:
            # 1. Extract Valid Legend Keys from Layout Scout (The Semantic Filter)
            valid_keys = set()
            if layout and layout.charts:
                for chart in layout.charts:
                    for key in chart.legend_keys:
                        valid_keys.add(key.lower().strip())

            # 2. Build Index of Vector Shapes (Preserved)
            spatial_index = SpatialIndex(page.rect)
            for d in page.get_drawings(): spatial_index.insert(d)
            
            # 2. Get Vector Text (Cleaner than OCR for finding legend labels)
            text_blocks = page.get_text("dict")["blocks"]
            raw_legend_bindings = []
            
            for block in text_blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if len(text) < 2 or any(c.isdigit() for c in text): continue 
                        
                        # [NEW] Semantic Check
                        # If Layout Scout is available, check if text matches known keys
                        is_confirmed = (text.lower() in valid_keys) if valid_keys else False
                        
                        # Search for color box (Preserved Logic)
                        span_rect = fitz.Rect(span["bbox"])
                        
                        # Search area: Small buffer around the text (looking for the little color square)
                        search_rect = span_rect + (-30, -5, 30, 5) 
                        
                        nearby_shapes = spatial_index.query(search_rect)
                        candidates = []
                        
                        for shape in nearby_shapes:
                            if shape.get('fill') is None: continue
                            
                            # Check valid color
                            r, g, b = shape['fill']
                            rgb_255 = (int(r*255), int(g*255), int(b*255))
                            if not self._is_valid_color(rgb_255): continue
                            
                            s_rect = fitz.Rect(shape['rect'])
                            dist = self._rect_distance(span_rect, s_rect)
                            
                            # Tight binding only (Legend markers are visually close)
                            if dist < 25: 
                                hex_c = '#{:02x}{:02x}{:02x}'.format(*rgb_255)
                                candidates.append((dist, rgb_255, hex_c))
                        
                        if candidates:
                            candidates.sort(key=lambda x: x[0]) # Closest shape wins
                            best = candidates[0]
                            # Store dictionary for updated ColorResolver
                            raw_legend_bindings.append({
                                "text": text, 
                                "rgb": best[1], 
                                "hex": best[2], 
                                "confirmed": is_confirmed
                            })

            # 4. Resolve Shades (e.g. "Dark Red")
            if not raw_legend_bindings: return ""
            
            hints = ColorResolver.resolve_names(raw_legend_bindings)
            return "\n".join(hints)
            
        except Exception as e:
            logger.warning(f"Vector hint generation failed: {e}")
            return ""

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

    @traceable(name="Parse PDF File", run_type="parser")
    async def parse(self, file_path: Path) -> List[IngestedChunk]:
        """
        HYBRID PIPELINE:
        1. Docling: Extracts complex Tables (The Gold)
        2. VLM Loop: Extracts Visuals & Strategy (Single Pass Context)
        """
        if not file_path.exists(): raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"🎨 Starting Centaur ingestion for: {file_path.name}")
        
        chunks = []
        doc_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

        # --- STEP 1: DOCLING (Tables Only) ---
        # Run Docling on a thread to avoid blocking
        loop = asyncio.get_running_loop()
        docling_res = await loop.run_in_executor(None, self.converter.convert, file_path)
        doc = docling_res.document

        for item, _ in doc.iterate_items():
            if isinstance(item, TableItem):
                chunks.extend(await self._process_complex_table(item, doc, doc_hash, file_path.name))

        # --- STEP 2: VLM SINGLE PASS (Charts + Page Context) ---
        # We open the PDF again with Fitz to render high-res images for the Vision pipeline
        with fitz.open(file_path) as pdf_doc:
            for page_num, page in enumerate(pdf_doc, start=1):
                logger.info(f"--- Processing Page {page_num} (VLM) ---")
                
                # Render Image
                zoom = 3.0 # High Res
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                
                # Base64 for VLM
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                # A. Layout Scout (Visual Detection)
                layout = await layout_scanner.scan(img_b64)

                # B. OCR Grid (Raster Spatial Map)
                ocr_context = self._extract_text_spatial(np.array(pil_img))

                # C. Vector Hints (The "Gold" - Legend Colors)
                # [UPDATED CALL] Now passing layout
                vector_hints = self._generate_vector_hints(page, layout=layout)
                
                full_context_str = f"[OCR SPATIAL GRID]\n{ocr_context}\n\n[VECTOR LEGEND HINTS]\n{vector_hints}"

                # D. Forensic Analysis (Auditor)
                # IMPORTANT: vision_tool must be updated to support 'analyze_full_page'
                analysis = await vision_tool.analyze_full_page(
                    image_data=img_bytes,
                    ocr_context=full_context_str,
                    layout_hint=layout
                )

                if analysis:
                    # Package Page Analysis
                    content_buffer = [f"## Page {page_num}: {analysis.title}"]
                    content_buffer.append(f"**Summary:** {analysis.summary}\n")
                    
                    if analysis.metrics:
                        content_buffer.append("### Key Metrics:")
                        for m in analysis.metrics:
                            # [UPDATED] Robust String Formatter for New Schema
                            # Logic: {Currency}{Value}{Magnitude} {Measure} (Handling "None")
                            
                            points_list = []
                            for p in m.data_points:
                                cur = p.currency if p.currency != "None" else ""
                                mag = p.magnitude if p.magnitude != "None" else ""
                                meas = p.measure if p.measure != "None" else ""
                                val_str = f"{cur}{p.numeric_value}{mag} {meas}".strip()
                                points_list.append(f"{p.label}={val_str}")
                                
                            points = ", ".join(points_list)
                            content_buffer.append(f"- **{m.series_label}:** {points}")
                    
                    if analysis.insights:
                        content_buffer.append("\n### Insights:")
                        for ins in analysis.insights:
                            content_buffer.append(f"- [{ins.category}] {ins.content}")

                    full_text = "\n".join(content_buffer)
                    
                    chunk = IngestedChunk(
                        chunk_id=str(uuid4()),
                        doc_hash=doc_hash,
                        page_number=page_num,
                        clean_text=full_text,
                        raw_text=ocr_context,
                        token_count=len(full_text.split()),
                        metadata={
                            "source": file_path.name,
                            "confidence": analysis.confidence_score,
                            "has_charts": layout.has_charts,
                            "audit_log": analysis.audit_log,
                            "type": "visual_analysis"
                        }
                    )
                    chunks.append(chunk)

        logger.info(f"✅ Extracted {len(chunks)} chunks from {file_path.name}")
        return chunks

pdf_parser = PDFParser()