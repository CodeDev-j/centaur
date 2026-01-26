import asyncio
import base64
import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Any, Tuple
from uuid import uuid4

# Third-party
import fitz  # PyMuPDF
import numpy as np
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import TableItem
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langsmith import traceable
from PIL import Image, ImageEnhance
from rapidocr_onnxruntime import RapidOCR

# Internal
from src.schemas.documents import IngestedChunk
from src.storage.blob_driver import BlobDriver
from src.tools.vision import vision_tool
from src.tools.layout_scanner import layout_scanner
from src.utils.colors import ColorResolver, LegendBinding
from src.utils.spatial import SpatialIndex

logger = logging.getLogger(__name__)

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
_OCR_EXECUTOR = ThreadPoolExecutor(max_workers=2)
RENDER_ZOOM = 3.0  # ~216 DPI (Use 4.0 for ~300 DPI)
LEGEND_SEARCH_RADIUS = (-30, -5, 30, 5)  # Visual buffer around text
LEGEND_DISTANCE_THRESHOLD = 25  # Max pixels between text and color box


class PDFParser:
    def __init__(self):
        # 1. OCR Engine
        self.ocr_engine = RapidOCR()

        # 2. Docling Setup (Kept ONLY for Tables)
        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = True
        pipeline_opts.do_table_structure = True
        pipeline_opts.images_scale = RENDER_ZOOM

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
            }
        )

    # --- VECTOR MATH HELPERS ---
    def _is_valid_color(self, p: Tuple[int, int, int]) -> bool:
        r, g, b = p
        if r > 240 and g > 240 and b > 240:
            return False  # White
        if r < 20 and g < 20 and b < 20:
            return False  # Black
        if abs(r - g) < 15 and abs(g - b) < 15:
            return False  # Gray
        return True

    def _rect_distance(self, r1: fitz.Rect, r2: fitz.Rect) -> float:
        x_dist = max(0, r1.x0 - r2.x1, r2.x0 - r1.x1)
        y_dist = max(0, r1.y0 - r2.y1, r2.y0 - r1.y1)
        return (x_dist**2 + y_dist**2)**0.5

    def _extract_text_spatial(
        self,
        page: fitz.Page,
        img_np: np.ndarray,
        zoom_factor: float
    ) -> str:
        """
        HYBRID EXTRACTION STRATEGY:
        1. Extract Digital Text (Ground Truth).
        2. Run OCR (Visual Fill).
        3. Merge: Only keep OCR text that does NOT overlap with Digital Text.
        """
        spatial_lines = []

        # --- LAYER 1: DIGITAL TEXT (Ground Truth) ---
        # Returns: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        digital_words = page.get_text("words")
        digital_rects = []  # For collision detection

        # Capture all digital text first
        for w in digital_words:
            # Create unscaled rect for collision logic (PDF Coordinate Space)
            r = fitz.Rect(w[0], w[1], w[2], w[3])
            digital_rects.append(r)

            # Formatted Output (Scaled to Image Space)
            x_img = int(w[0] * zoom_factor)
            y_img = int(w[1] * zoom_factor)
            spatial_lines.append(f"[{y_img}, {x_img}] {w[4]}")

        # --- LAYER 2: OCR FILL (The Safety Net) ---
        # We run OCR to catch text embedded in images (Zombie Charts)
        try:
            ocr_results, _ = self.ocr_engine(img_np)
            if ocr_results:
                for res in ocr_results:
                    box, text, score = res
                    # RapidOCR Box: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    # Map Image Coordinates BACK to PDF Coordinates
                    x1_img, y1_img = box[0]
                    x2_img, y2_img = box[2]

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
                            # If overlap > 30% of smaller box, it's a duplicate
                            overlap_ratio = intersect.get_area() / min(
                                ocr_rect.get_area(), d_rect.get_area()
                            )
                            if overlap_ratio > 0.3:
                                is_duplicate = True
                                break

                    if not is_duplicate:
                        # It's unique! Add it (using original Image Coords)
                        spatial_lines.append(
                            f"[{int(y1_img)}, {int(x1_img)}] {text}"
                        )

        except Exception as e:
            logger.warning(f"⚠️ OCR Layer failed: {e}")

        # Final Sort: Top-down, Left-right logic for clean reading
        def parse_sort_key(line):
            try:
                coords = line.split("]")[0].strip("[").split(",")
                return int(coords[0]), int(coords[1])
            except Exception:
                return (0, 0)

        spatial_lines.sort(key=parse_sort_key)
        return "\n".join(spatial_lines)

    def _generate_vector_hints(
        self,
        page: fitz.Page,
        layout: Optional[Any] = None
    ) -> str:
        """
        Uses SpatialIndex to find Legend matches and returns text hints.
        """
        try:
            # 1. Extract Valid Legend Keys from Layout Scout (The Semantic Filter)
            valid_keys = set()
            if layout and layout.charts:
                for chart in layout.charts:
                    for key in chart.legend_keys:
                        valid_keys.add(key.lower().strip())

            # 2. Build Index of Vector Shapes
            spatial_index = SpatialIndex(page.rect)
            for d in page.get_drawings():
                spatial_index.insert(d)

            # 3. Get Vector Text (Cleaner than OCR for finding legend labels)
            text_blocks = page.get_text("dict")["blocks"]
            raw_legend_bindings: List[LegendBinding] = []

            for block in text_blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if len(text) < 2 or any(c.isdigit() for c in text):
                            continue

                        # Semantic Check
                        is_confirmed = (
                            (text.lower() in valid_keys) if valid_keys else False
                        )

                        # Search for color box
                        span_rect = fitz.Rect(span["bbox"])
                        
                        # Search area: Small buffer around the text (looking for the little color square)
                        search_rect = span_rect + LEGEND_SEARCH_RADIUS

                        nearby_shapes = spatial_index.query(search_rect)
                        candidates = []

                        for shape in nearby_shapes:
                            if shape.get('fill') is None:
                                continue

                            # Check valid color
                            r, g, b = shape['fill']
                            rgb_255 = (int(r * 255), int(g * 255), int(b * 255))
                            if not self._is_valid_color(rgb_255):
                                continue

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

            # 4. Resolve Shades (e.g. "Dark Red")
            if not raw_legend_bindings:
                return ""

            hints = ColorResolver.resolve_names(raw_legend_bindings)
            return "\n".join(hints)

        except Exception as e:
            logger.warning(f"Vector hint generation failed: {e}")
            return ""

    async def _process_complex_table(
        self,
        item,
        doc,
        doc_hash,
        filename
    ) -> List[IngestedChunk]:
        df = item.export_to_dataframe(doc)
        if df.empty:
            return []

        chunks = []
        chunk_size = 10
        total_rows = len(df)
        cols = df.columns
        header_md = f"| {' | '.join(str(c) for c in cols)} |\n|{'---|' * len(cols)}"

        for start_row in range(0, total_rows, chunk_size):
            end_row = min(start_row + chunk_size, total_rows)
            chunk_df = df.iloc[start_row:end_row]
            body_md = chunk_df.to_markdown(
                index=False, tablefmt="pipe"
            ).split('\n', 2)[-1]
            full_text = (
                f"Context (Headers):\n{header_md}\n\n"
                f"Segment ({start_row}-{end_row}):\n{body_md}"
            )

            chunk_id = str(uuid4())
            await BlobDriver.save_json(
                {"html": df.to_html()}, "tables", f"{chunk_id}.json"
            )

            chunks.append(IngestedChunk(
                chunk_id=chunk_id,
                doc_hash=doc_hash,
                clean_text=full_text,
                raw_text=full_text,
                page_number=item.prov[0].page_no if item.prov else 1,
                token_count=len(full_text.split()),
                metadata={
                    "source": filename,
                    "type": "table",
                    "rows": f"{start_row}-{end_row}"
                }
            ))
        return chunks

    @traceable(name="Parse PDF File", run_type="parser")
    async def parse(self, file_path: Path) -> List[IngestedChunk]:
        """
        HYBRID PIPELINE:
        1. Docling: Extracts complex Tables
        2. VLM Loop: Extracts Visuals & Strategy (Single Pass Context)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"🎨 Starting Centaur ingestion for: {file_path.name}")

        chunks = []
        doc_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

        # --- STEP 1: DOCLING (Tables Only) ---
        # Run Docling on a thread to avoid blocking
        loop = asyncio.get_running_loop()
        docling_res = await loop.run_in_executor(
            None, self.converter.convert, file_path
        )
        doc = docling_res.document

        for item, _ in doc.iterate_items():
            if isinstance(item, TableItem):
                chunks.extend(
                    await self._process_complex_table(
                        item, doc, doc_hash, file_path.name
                    )
                )

        # --- STEP 2: VLM SINGLE PASS (Charts + Page Context) ---
        # We open the PDF again with Fitz to render high-res images for the Vision pipeline
        with fitz.open(file_path) as pdf_doc:
            for page_num, page in enumerate(pdf_doc, start=1):
                logger.info(f"--- Processing Page {page_num} (VLM) ---")

                # Render Image
                zoom = RENDER_ZOOM
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                pil_img = Image.open(io.BytesIO(pix.tobytes("png")))

                # Base64 for VLM
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                # A. Layout Scout (Visual Detection)
                layout = await layout_scanner.scan(img_b64)

                # B. OCR Grid (Hybrid Spatial Map)
                # FORCED CALL: We always run this now to get the 'Union' of text
                ocr_context = self._extract_text_spatial(
                    page, np.array(pil_img), zoom
                )

                # C. Vector Hints (Legend Colors)
                vector_hints = self._generate_vector_hints(page, layout=layout)

                full_context_str = (
                    f"[OCR SPATIAL GRID]\n{ocr_context}\n\n"
                    f"[VECTOR LEGEND HINTS]\n{vector_hints}"
                )

                # D. Forensic Analysis (Auditor)
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
                            # Robust String Formatter
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