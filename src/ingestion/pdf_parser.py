import logging
import hashlib
import math
import collections
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
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

# ==============================================================================
# 🧠 INTELLIGENT CLASSIFIER (The "Voting System")
# ==============================================================================
def classify_region(text_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Classifies a region as 'chart' or 'general' (Text Block) using weighted heuristics.
    Source: Gemini Deep Thinking Strategy.
    """
    if not text_items:
        # Default to chart if no text found but it was a visual crop
        return {"classification": "chart", "score": 0, "debug": "No overlapping text"}

    full_text = " ".join([item['text'] for item in text_items])
    tokens = re.findall(r'\w+', full_text.lower())
    total_tokens = len(tokens) if tokens else 1
    total_chars = len(full_text.replace(" ", ""))

    # Features
    STOPWORDS = {'the', 'and', 'of', 'to', 'in', 'is', 'for', 'with', 'on', 'as', 'by', 'at', 'from', 'be', 'are', 'due', 'approximately'}
    stopword_count = sum(1 for t in tokens if t in STOPWORDS)
    stopword_ratio = stopword_count / total_tokens

    data_chars = sum(1 for c in full_text if c.isdigit() or c in "$%()€£")
    data_density = data_chars / total_chars if total_chars > 0 else 0

    # Strong Signal: Currency format ($XX,XXX)
    has_financials = bool(re.search(r'\$\d{1,3}(?:,\d{3})*(?:\.\d+)?', full_text))

    x0_coords = [item['bbox'][0] for item in text_items]
    def get_align_ratio(coords, tol=5.0):
        if len(coords) < 2: return 0
        aligned = 0
        for i, c1 in enumerate(coords):
            if any(abs(c1 - c2) <= tol for j, c2 in enumerate(coords) if i != j):
                aligned += 1
        return aligned / len(coords)
    align_score = get_align_ratio(x0_coords)
    
    # Average Token Length (Text uses long words like "significant"; Charts use "Q1")
    avg_len = sum(len(t) for t in tokens) / total_tokens

    # --- SCORING LOGIC (TUNED) ---
    score = 0.0
    
    # 1. Linguistic Penalties (The "Text" Defenders)
    if stopword_ratio > 0.10: score -= 50   # Sentence structure detected
    elif stopword_ratio < 0.05: score += 20 # Telegraphic/Label style
    
    if avg_len > 6.0: score -= 20           # Long words -> Text Block

    # 2. Geometric Bonuses (The "Chart" Attackers)
    if align_score > 0.40: score += 30      # Grid structure
    if data_density > 0.30: score += 25     # Number heavy
    
    # 3. The Financial Tie-Breaker (Reduced from 40 to 20)
    # This ensures a "$" can't overcome a high stopword count (-50) on its own.
    if has_financials: score += 20

    classification = "chart" if score > 0 else "general"
    
    return {
        "classification": classification,
        "score": score,
        "debug": f"StopWord:{stopword_ratio:.2f}, AvgLen:{avg_len:.1f}, Align:{align_score:.2f}, HasFin:{has_financials}"
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
        
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
        )

    async def parse(self, file_path: Path) -> List[IngestedChunk]:
        if not file_path.exists(): raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"🎨 Starting Pro-Docling ingestion for: {file_path.name}")
        
        result = self.converter.convert(file_path)
        doc = result.document
        doc_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        chunks: List[IngestedChunk] = []
        
        for item, _ in doc.iterate_items():
            if item.label.value in ["page_header", "page_footer"]: continue

            if isinstance(item, TableItem):
                chunks.extend(await self._process_complex_table(item, doc, doc_hash, file_path.name))
                continue
                
            if isinstance(item, PictureItem) or item.label.value in ["picture", "figure", "chart"]:
                vision_chunk = await self._process_visual(item, doc, doc_hash, file_path.name)
                if vision_chunk: chunks.append(vision_chunk)
                continue

            if item.text.strip():
                chunk = self._create_standard_chunk(item, doc, doc_hash, file_path.name)
                if chunk: chunks.append(chunk)

        logger.info(f"✅ Extracted {len(chunks)} chunks from {file_path.name}")
        return chunks

    async def _process_complex_table(self, item: TableItem, doc, doc_hash: str, filename: str) -> List[IngestedChunk]:
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

    async def _process_visual(self, item: DocItem, doc, doc_hash: str, filename: str) -> Optional[IngestedChunk]:
        if not item.prov: return None
        prov = item.prov[0]
        page_no = prov.page_no
        
        try:
            page_ref = doc.pages[page_no].image
            if not page_ref or not page_ref.pil_image: return None
            page_img = page_ref.pil_image
            
            # Coordinate Fix
            pdf_w = doc.pages[page_no].size.width
            pdf_h = doc.pages[page_no].size.height
            img_w, img_h = page_img.size
            scale_x = img_w / pdf_w
            scale_y = img_h / pdf_h
            
            bbox = prov.bbox
            pdf_y0 = min(bbox.b, bbox.t)
            pdf_y1 = max(bbox.b, bbox.t)
            
            px_x0 = bbox.l * scale_x
            px_x1 = bbox.r * scale_x
            px_y0 = pdf_y0 * scale_y
            px_y1 = pdf_y1 * scale_y
            
            # PADDING INCREASED TO 100px
            PAD = 100
            crop_box = (
                math.floor(max(0, px_x0 - PAD)), 
                math.floor(max(0, px_y0 - PAD)), 
                math.ceil(min(img_w, px_x1 + PAD)), 
                math.ceil(min(img_h, px_y1 + PAD))
            )
            # Flip Check
            if crop_box[3] <= crop_box[1]:
                 crop_box = (
                    math.floor(max(0, px_x0 - PAD)), 
                    math.floor(max(0, img_h - px_y1 - PAD)), 
                    math.ceil(min(img_w, px_x1 + PAD)), 
                    math.ceil(min(img_h, img_h - px_y0 + PAD))
                )

            cropped_img = page_img.crop(crop_box)
            chunk_id = str(uuid4())
            img_filename = f"{chunk_id}.png"
            cropped_img.save(SystemPaths.LAYOUTS / img_filename)
            
            # Harvest OCR
            classifier_items = []
            ocr_text_list = []
            
            for text_item, _ in doc.iterate_items():
                if not text_item.prov or text_item.prov[0].page_no != page_no: continue
                if not hasattr(text_item, "text") or not text_item.text.strip(): continue

                t_bbox = text_item.prov[0].bbox
                tx_center = (t_bbox.l + t_bbox.r) / 2
                ty_center = (t_bbox.t + t_bbox.b) / 2
                
                # Logic Buffer (Same as PAD)
                if (bbox.l - 60 <= tx_center <= bbox.r + 60) and \
                   (min(bbox.t, bbox.b) - 60 <= ty_center <= max(bbox.t, bbox.b) + 60):
                    
                    txt = text_item.text.strip()
                    ocr_text_list.append(txt)
                    classifier_items.append({'text': txt, 'bbox': [t_bbox.l, t_bbox.b, t_bbox.r, t_bbox.t]})

            # Routing
            classification_result = classify_region(classifier_items)
            mode = classification_result["classification"]
            
            # Ultimate Override: If Visual Label + Financial Score > -20, force Chart
            labels = item.label.value.lower()
            is_visual_label = any(x in labels for x in ["chart", "figure", "picture"])
            
            if is_visual_label and (classification_result['score'] > -20): 
                mode = "chart"

            ocr_context = " | ".join(ocr_text_list) if ocr_text_list else "No text detected."
            color_context = self._scan_for_legend_colors(cropped_img) if mode == "chart" else "N/A"

            description = vision_tool.analyze(
                file_name=img_filename, 
                mode=mode, 
                context=f"OCR ({mode.upper()}): {ocr_context}\nSTATS: {classification_result.get('debug', 'N/A')}\nCOLORS: {color_context}"
            )
            
            return IngestedChunk(
                chunk_id=chunk_id, doc_hash=doc_hash,
                clean_text=f"[{mode.upper()} ANALYSIS]\n{description}", raw_text=description,
                page_number=page_no, token_count=len(description.split()),
                metadata={
                    "source": filename, "type": "visual", "image_path": f"layouts/{img_filename}",
                    "mode": mode, "ocr_evidence": ocr_context
                }
            )
        except Exception as e:
            logger.error(f"Visual processing failed: {e}")
            return None

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

    def _scan_for_legend_colors(self, image: Image.Image) -> str:
        w, h = image.size
        if w < 50 or h < 50: return "Image too small for color scan."
        zones = [("Left", image.crop((0, 0, min(30, w), h))), ("Right", image.crop((max(0, w-30), 0, w, h))), ("Bottom", image.crop((0, max(0, h-30), w, h)))]
        found_colors = []
        for name, zone in zones:
            if zone.width == 0 or zone.height == 0: continue
            pixels = zone.resize((50, 50)).getdata()
            non_neutrals = [p for p in pixels if (max(p)-min(p)>20) and not (p[0]>240 and p[1]>240 and p[2]>240)]
            if non_neutrals:
                most_common = collections.Counter(non_neutrals).most_common(2)
                found_colors.extend([f"{name}: {'#{:02x}{:02x}{:02x}'.format(*c[0])}" for c in most_common])
        return " | ".join(found_colors)