import logging
import hashlib
import json
import math
import collections
from pathlib import Path
from typing import List, Tuple, Optional, Any
from uuid import uuid4

# Third-party
import pandas as pd
from PIL import Image
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import DocItem, TableItem, PictureItem

# Internal
from src.schemas.documents import IngestedChunk
from src.schemas.citation import BoundingBox
from src.storage.blob_driver import BlobDriver
from src.tools.vision import vision_tool
from src.config import SystemPaths

logger = logging.getLogger(__name__)

class PDFParser:
    """
    Helix A: The Visual Stream (Pro Edition).
    Features:
    - Pixel-Perfect Grounding (0.0-1.0 Norm)
    - Sticky Header Table Chunking
    - Spatial OCR Injection & Legend Color Scanning
    - Robust Image Cropping (Coordinate Sanitization)
    """
    
    def __init__(self):
        # Configure Docling to give us the raw page images for visual analysis
        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = True
        pipeline_opts.do_table_structure = True
        pipeline_opts.generate_page_images = True # CRITICAL for Vision Tool
        pipeline_opts.generate_picture_images = True
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
            }
        )

    async def parse(self, file_path: Path) -> List[IngestedChunk]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"🎨 Starting Pro-Docling ingestion for: {file_path.name}")
        
        # 1. Run Docling
        result = self.converter.convert(file_path)
        doc = result.document
        
        # Generate Lineage Hash
        file_bytes = file_path.read_bytes()
        doc_hash = hashlib.sha256(file_bytes).hexdigest()
        
        chunks: List[IngestedChunk] = []
        
        # 2. Iterate and Route Items
        for item, level in doc.iterate_items():
            
            # Skip noise (headers/footers)
            if item.label.value in ["page_header", "page_footer"]:
                continue

            # ROUTE A: TABLES (Sticky Headers)
            if isinstance(item, TableItem):
                table_chunks = await self._process_complex_table(item, doc, doc_hash, file_path.name)
                chunks.extend(table_chunks)
                continue
                
            # ROUTE B: VISUALS (Charts/Figures)
            if isinstance(item, PictureItem) or item.label.value in ["picture", "figure", "chart"]:
                vision_chunk = await self._process_visual(item, doc, doc_hash, file_path.name)
                if vision_chunk:
                    chunks.append(vision_chunk)
                continue

            # ROUTE C: STANDARD TEXT
            if item.text.strip():
                chunk = self._create_standard_chunk(item, doc, doc_hash, file_path.name)
                if chunk:
                    chunks.append(chunk)

        logger.info(f"✅ Extracted {len(chunks)} chunks from {file_path.name}")
        return chunks

    # =========================================================================
    # LOGIC 1: COMPLEX TABLES (Sticky Headers)
    # =========================================================================
    
    async def _process_complex_table(self, item: TableItem, doc, doc_hash: str, filename: str) -> List[IngestedChunk]:
        """
        Splits large tables while repeating the headers in every chunk.
        """
        # Suppress deprecation warning and improve resolution
        df = item.export_to_dataframe(doc)
        if df.empty:
            return []
            
        chunks = []
        chunk_size = 10 # Rows per chunk
        total_rows = len(df)
        
        # 1. Identify Headers (Naive: Assume first row is header if not multi-index)
        header_md = f"| {' | '.join(str(c) for c in df.columns)} |\n|{'---|' * len(df.columns)}"
        
        # 2. Chunking Loop
        for start_row in range(0, total_rows, chunk_size):
            end_row = min(start_row + chunk_size, total_rows)
            chunk_df = df.iloc[start_row:end_row]
            
            # Combine Sticky Header + Body
            body_md = chunk_df.to_markdown(index=False, tablefmt="pipe").split('\n', 2)[-1] 
            full_text = f"Context (Table Headers):\n{header_md}\n\nSegment (Rows {start_row}-{end_row}):\n{body_md}"
            
            # 3. Create Chunk
            chunk_id = str(uuid4())
            
            # Save the layout blob (full table structure)
            await BlobDriver.save_json(
                data={"html": df.to_html()},
                folder="tables",
                filename=f"{chunk_id}.json"
            )

            chunks.append(IngestedChunk(
                chunk_id=chunk_id,
                doc_hash=doc_hash,
                clean_text=full_text,
                raw_text=full_text,
                page_number=item.prov[0].page_no if item.prov else 1,
                token_count=len(full_text.split()),
                metadata={"source": filename, "type": "table", "rows": f"{start_row}-{end_row}"}
            ))
            
        return chunks

    # =========================================================================
    # LOGIC 2: VISUALS (Spatial OCR + Color Injection)
    # =========================================================================

    async def _process_visual(self, item: DocItem, doc, doc_hash: str, filename: str) -> Optional[IngestedChunk]:
        """
        Extracts image -> Harvests Overlapping Text (OCR) -> Scans Colors -> Calls Vision Tool.
        """
        if not item.prov:
            return None
            
        prov = item.prov[0]
        page_no = prov.page_no
        
        try:
            # 1. Get the Image Crop
            # Use .pil_image to unwrap the ImageRef object safely
            page_ref = doc.pages[page_no].image
            if not page_ref or not page_ref.pil_image:
                logger.warning(f"No image data found for page {page_no}")
                return None
                
            page_img = page_ref.pil_image
                
            # --- Robust Crop Box Calculation ---
            # We sort coordinates to handle PDF vs Image origin differences (Bottom-Left vs Top-Left)
            bbox = prov.bbox
            
            # 1. Get raw coordinates
            l, t, r, b = bbox.l, bbox.t, bbox.r, bbox.b
            
            # 2. Sort them (Min = Left/Top, Max = Right/Bottom)
            # This prevents "Lower < Upper" errors if coords are inverted
            x0, x1 = min(l, r), max(l, r)
            y0, y1 = min(t, b), max(t, b)
            
            # 3. Clamp to image boundaries (Avoids "tile cannot extend outside image")
            img_w, img_h = page_img.size
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(img_w, x1)
            y1 = min(img_h, y1)
            
            # 4. Final Integer Conversion
            crop_box = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
            
            # 5. Safety Check: If crop is 0-width or 0-height, skip
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                logger.warning(f"Skipping visual on page {page_no}: Invalid crop dimensions {crop_box}")
                return None

            cropped_img = page_img.crop(crop_box)
            # ------------------------------------------
            
            # 2. Save Image for Vision Tool
            chunk_id = str(uuid4())
            img_filename = f"{chunk_id}.png"
            temp_path = SystemPaths.LAYOUTS / img_filename
            cropped_img.save(temp_path)
            
            # 3. Harvest OCR Data (Spatial Overlap Check)
            # <--- FIXED: Safely iterate ALL items and filter by page manually --->
            ocr_evidence = []
            for text_item, _ in doc.iterate_items():
                
                # Filter: Must be on the same page
                if not text_item.prov or text_item.prov[0].page_no != page_no:
                    continue
                    
                if text_item.label.value in ["text", "code", "formula", "caption"]:
                    t_bbox = text_item.prov[0].bbox
                    # Does the text center fall inside the chart?
                    t_center_x = (t_bbox.l + t_bbox.r) / 2
                    t_center_y = (t_bbox.t + t_bbox.b) / 2
                    
                    # Use our sanitized coords for check
                    if (x0 <= t_center_x <= x1) and (y0 <= t_center_y <= y1):
                        ocr_evidence.append(text_item.text.strip())

            ocr_context_str = " | ".join(ocr_evidence) if ocr_evidence else "No text detected inside chart area."

            # 4. Generate Color Context (Legend Scanning)
            color_context = self._scan_for_legend_colors(cropped_img)
            
            # 5. Build the "Pro" Context Block
            full_context = (
                f"OCR EVIDENCE (Text found inside image):\n{ocr_context_str}\n\n"
                f"COLOR EVIDENCE (Dominant colors):\n{color_context}"
            )

            # 6. Call Vision Tool
            mode = "chart" if "chart" in item.label.value or "figure" in item.label.value else "general"
            description = vision_tool.analyze(file_name=img_filename, mode=mode, context=full_context)
            
            return IngestedChunk(
                chunk_id=chunk_id,
                doc_hash=doc_hash,
                clean_text=f"[VISUAL ANALYSIS]\n{description}",
                raw_text=description,
                page_number=page_no,
                token_count=len(description.split()),
                metadata={
                    "source": filename, 
                    "type": "visual", 
                    "image_path": f"layouts/{img_filename}",
                    "mode": mode,
                    "ocr_evidence": ocr_context_str
                }
            )

        except Exception as e:
            logger.error(f"Visual processing failed: {e}")
            return None

    # =========================================================================
    # LOGIC 3: STANDARD TEXT (Pixel-Perfect Norm)
    # =========================================================================

    def _create_standard_chunk(self, item: DocItem, doc, doc_hash: str, filename: str) -> Optional[IngestedChunk]:
        if not item.prov:
            return None
            
        prov = item.prov[0]
        page = doc.pages[prov.page_no]
        width, height = page.size.width, page.size.height
        
        # Calculate normalized bbox (0.0 - 1.0)
        # Robust min/max just in case
        l, r = sorted([prov.bbox.l, prov.bbox.r])
        t, b = sorted([prov.bbox.t, prov.bbox.b])
        
        bbox = BoundingBox(
            page_number=prov.page_no,
            x=l / width,
            y=t / height,
            width=(r - l) / width,
            height=(b - t) / height
        )
        
        return IngestedChunk(
            chunk_id=str(uuid4()),
            doc_hash=doc_hash,
            clean_text=item.text.strip(),
            raw_text=item.text,
            page_number=prov.page_no,
            token_count=len(item.text.split()),
            primary_bbox=bbox,
            metadata={"source": filename, "type": "text"}
        )

    # =========================================================================
    # HELPER: COLOR MATH
    # =========================================================================
    
    def _scan_for_legend_colors(self, image: Image.Image) -> str:
        """
        Scans the image for dominant non-neutral colors to help the LLM match legends.
        """
        # Resize to speed up scan
        small_img = image.resize((100, 100))
        pixels = small_img.getdata()
        
        # Filter neutrals (White/Black/Gray)
        non_neutrals = []
        for r, g, b in pixels:
            # Simple heuristic: if saturation is low, it's neutral
            if max(r,g,b) - min(r,g,b) < 20: 
                continue # Grayish
            if r > 240 and g > 240 and b > 240: 
                continue # Whiteish
            if r < 15 and g < 15 and b < 15: 
                continue # Blackish
            non_neutrals.append((r,g,b))
            
        if not non_neutrals:
            return ""
            
        # Get top 3 colors
        most_common = collections.Counter(non_neutrals).most_common(3)
        hints = []
        for color, count in most_common:
            hex_val = '#{:02x}{:02x}{:02x}'.format(*color)
            hints.append(f"Found Dominant Color: {hex_val}")
            
        return "\n".join(hints)