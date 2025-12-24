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
        return {"classification": "Unknown", "score": 0}

    # 1. Pre-Computation
    full_text = " ".join([item['text'] for item in text_items])
    tokens = re.findall(r'\w+', full_text.lower())
    total_tokens = len(tokens) if tokens else 1
    total_chars = len(full_text.replace(" ", ""))

    # 2. Features
    
    # A. Stopwords (Linguistic) - Text blocks use grammar, charts don't.
    STOPWORDS = {'the', 'and', 'of', 'to', 'in', 'is', 'for', 'with', 'on', 'as', 'by', 'at', 'from', 'be', 'are', 'due', 'approximately'}
    stopword_count = sum(1 for t in tokens if t in STOPWORDS)
    stopword_ratio = stopword_count / total_tokens

    # B. Data Density (Linguistic) - Digits & Symbols
    data_chars = sum(1 for c in full_text if c.isdigit() or c in "$%()€£")
    data_density = data_chars / total_chars if total_chars > 0 else 0

    # C. Alignment (Geometric) - Vertical Grid check
    x0_coords = [item['bbox'][0] for item in text_items]
    x1_coords = [item['bbox'][2] for item in text_items]
    
    def get_align_ratio(coords, tol=5.0):
        if len(coords) < 2: return 0
        aligned = 0
        for i, c1 in enumerate(coords):
            if any(abs(c1 - c2) <= tol for j, c2 in enumerate(coords) if i != j):
                aligned += 1
        return aligned / len(coords)

    align_score = max(get_align_ratio(x0_coords), get_align_ratio(x1_coords))

    # 3. Scoring
    score = 0.0
    
    # Rule 1: Stopwords (Strong Negative) - The "Key Highlights" Killer
    if stopword_ratio > 0.15: score -= 50
    elif stopword_ratio < 0.05: score += 20

    # Rule 2: Alignment (Strong Positive) - Charts use grids
    if align_score > 0.50: score += 30

    # Rule 3: Data Density (Moderate Positive)
    if data_density > 0.30: score += 25
    elif data_density < 0.10: score -= 10

    # Rule 4: Token Length (Nuance) - "Q1" vs "Synergies"
    avg_len = sum(len(t) for t in tokens) / total_tokens
    if avg_len < 4.5: score += 10
    
    return {
        "classification": "chart" if score > 0 else "general",
        "score": score,
        "debug": f"StopWord:{stopword_ratio:.2f}, Dens:{data_density:.2f}, Align:{align_score:.2f}"
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
                
            # Robust Crop
            bbox = prov.bbox
            x0, x1 = min(bbox.l, bbox.r), max(bbox.l, bbox.r)
            y0, y1 = min(bbox.t, bbox.b), max(bbox.t, bbox.b)
            crop_box = (math.floor(max(0, x0)), math.floor(max(0, y0)), 
                        math.ceil(min(page_img.size[0], x1)), math.ceil(min(page_img.size[1], y1)))
            
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]: return None
            cropped_img = page_img.crop(crop_box)
            
            chunk_id = str(uuid4())
            img_filename = f"{chunk_id}.png"
            cropped_img.save(SystemPaths.LAYOUTS / img_filename)
            
            # --- HARVEST OCR & CLASSIFY ---
            classifier_items = []
            ocr_text_list = []
            
            for text_item, _ in doc.iterate_items():
                if not text_item.prov or text_item.prov[0].page_no != page_no: continue
                
                # Check Overlap with 5px buffer
                t_bbox = text_item.prov[0].bbox
                tx0, tx1 = min(t_bbox.l, t_bbox.r), max(t_bbox.l, t_bbox.r)
                ty0, ty1 = min(t_bbox.t, t_bbox.b), max(t_bbox.t, t_bbox.b)
                t_center_x, t_center_y = (tx0 + tx1)/2, (ty0 + ty1)/2
                
                if (x0-5 <= t_center_x <= x1+5) and (y0-5 <= t_center_y <= y1+5):
                    txt = text_item.text.strip()
                    ocr_text_list.append(txt)
                    classifier_items.append({'text': txt, 'bbox': [tx0, ty0, tx1, ty1]})

            # Run the Voting System
            classification_result = classify_region(classifier_items)
            mode = classification_result["classification"]
            
            # Hybrid override: If Docling is certain it's a chart, and our classifier is ambivalent, trust Docling
            if "chart" in item.label.value and mode == "general":
                if classification_result['score'] > -15: mode = "chart"

            ocr_context = " | ".join(ocr_text_list) if ocr_text_list else "No text detected."
            color_context = self._scan_for_legend_colors(cropped_img) if mode == "chart" else "N/A"

            # Call Vision Tool
            description = vision_tool.analyze(
                file_name=img_filename, 
                mode=mode, 
                context=f"OCR ({mode.upper()}): {ocr_context}\nSTATS: {classification_result['debug']}\nCOLORS: {color_context}"
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