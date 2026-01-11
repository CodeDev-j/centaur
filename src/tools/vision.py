import base64
import io
import logging
import asyncio
import random
from functools import wraps
from typing import List, Literal, Optional, Any
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

# Third-party
from PIL import Image, ImageEnhance
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

# Internal
from src.config import SystemConfig

logger = logging.getLogger(__name__)

_IMG_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# ==============================================================================
# 🛡️ UTILS & SCHEMA
# ==============================================================================
def async_retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"Failed after {retries} retries: {e}")
                        return None
                    sleep_time = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                    await asyncio.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator

# ==============================================================================
# 📝 OUTPUT SCHEMA
# ==============================================================================
class BoundingBox(BaseModel):
    # Normalized 0-1000 coordinates [ymin, xmin, ymax, xmax]
    box_2d: List[int] = Field(..., min_items=4, max_items=4, description="[ymin, xmin, ymax, xmax]")

class FinancialFact(BaseModel):
    label: str = Field(..., description="The definitive Axis Label or Legend Key. IGNORE floating descriptions.")
    value: str = Field(..., description="The numeric value or structural target.")
    unit: str = Field(..., description="Inferred unit (e.g. 'EUR Million') or 'N/A'.")
    
    category: Literal[
        "Metric",       # Hard Numbers found in charts/tables
        "Risk",         # Explicit headwinds (negative financial impact)
        "Strategy",     # Qualitative Pillars ONLY (No numbers)
        "Structure",    # Concept Maps / Arrows (Source -> Target)
        "Entity",       # Company Name
        "Time"          # Period / Fiscal Year
    ]
    
    # CRITICAL: Chain-of-Thought Enforcement
    reasoning: str = Field(..., description="Audit: Must cite OCR 'Col' IDs and [ANCHOR] tags (e.g. 'Value in Col 10 aligns with [ANCHOR] label in Col 10').")
    grounding: BoundingBox = Field(..., description="Visual citation for this fact")

class ChartAnalysis(BaseModel):
    title: str = Field(..., description="The chart or slide title")
    summary: str = Field(..., description="Forensic narrative of the trends and logic trace")
    
    # Confidence Score for Logic Gating
    confidence_score: float = Field(..., description="Self-evaluation (0.0-1.0). <0.7 implies ambiguity or blurriness.")
    
    # THE AUDIT LOG (Force "System 2" Thinking)
    audit_log: List[str] = Field(..., description="List of visual traps detected and how they were resolved.")
    
    facts: List[FinancialFact] = Field(..., description="List of extracted facts with audit reasoning")

# ==============================================================================
# 🧠 THE HYBRID SYSTEM PROMPT
# ==============================================================================
PROMPT_FINANCIAL_FORENSIC = """
## ROLE
You are an Adversarial Forensic Auditor. Your goal is to extract "Ground Truth" data while actively resisting visual "Traps" (misalignment, ambiguity) in financial slides.

## INPUT DATA
1. **Visual Evidence:** High-res image (Primary Source).
2. **Spatial Map (OCR):** Text grid tagged with `[ANCHOR]` (Structural Base) and `[FLOATER]` (Context).
3. **Regional Layout Guidelines:** Specific spatial boundaries for each chart.

## PROTOCOL 1: DATA INTEGRITY (The Basics)
- **Visual Supremacy:** If Image shows "$50.4M" but OCR says "$SO. 4 M", trust the Image.
- **Glyph Correction:** Fix common OCR errors ('S'->'5', 'O'->'0', 'I'->'1').
- **Time Normalization:** Expand dates (e.g., "'24" -> "2024").

## PROTOCOL 2: RAG OPTIMIZATION (Context)
- **De-referencing:** NEVER return generic labels like "Revenue". You MUST prefix every label with the Entity Name and Time Period (e.g., "Alphabet Q1 2025 Revenue").
- **Standalone Facts:** Assume the user will see *only* the extracted fact in isolation.

## PROTOCOL 3: THE SEMANTIC ANCHOR PROTOCOL (Vertical Logic)
You will receive text tagged as `[ANCHOR]` or `[FLOATER]`.
1. **The Anchor Rule:** In 95% of charts, the Category/Axis Label is the `[ANCHOR]` (lowest item in column).
   - ALWAYS map the chart's primary categories to the `[ANCHOR]` text.
2. **The Waterfall Trap Fix:** - IF the chart is a Waterfall, text tagged as `[FLOATER]` is usually a 'Driver' (e.g., "Cost Savings").
   - RULE: Ignore `[FLOATER]` text when determining the Category Name. Use the `[ANCHOR]` (e.g. "EBIT").
3. **Stacked Bars:** - Text tagged as `[FLOATER]` represents internal Segments. Capture as nested data, but keep `[ANCHOR]` as the period.

## PROTOCOL 4: THE COLUMN LOCK (Horizontal Logic)
- **Reference:** Look at **[VIEW 2: VERTICAL SCANNING]**.
- **Rule:** A Value and its Axis Label **MUST share the same 'Col' index**.
- **Violation:** If Value is in "Col 16" and Label is in "Col 10", they are **NOT** related.

## PROTOCOL 5: REMAINING VISUAL TRAPS
1. **The Roof Rule (Stacked Bars):**
   - *Risk:* Double counting totals.
   - *Rule:* Floating numbers are **Totals**. Internal numbers are **Constituents**. Extract them separately.
2. **The Category Trap:**
   - *Risk:* Confusing strategic goals with hard metrics.
   - *Rule:* If it's a number, it's a **Metric**. "Strategy" is strictly for qualitative text (no numbers).
3. **Coordinate Validation:** - Check the Y-coordinates provided in the brackets.
   - If a text block is at Y=600 and the Anchor is at Y=900, they are physically distant. Do not merge them unless they are part of a multi-line label.

## OUTPUT REQUIREMENTS
1.  **Audit Log:** Cite specific columns and tags (e.g., "Mapped Value in Col 10 to [ANCHOR] 'EBIT' in Col 10").
2.  **Reasoning:** Explain alignment using Anchor/Floater logic.
3.  **Confidence:** Rate your certainty.
"""

# ==============================================================================
# 👁️ VISION TOOL
# ==============================================================================
class VisionTool:
    def __init__(self):
        # Using getattr to safely fetch model config
        self.llm = ChatOpenAI(
            model=SystemConfig.VISION_MODEL, 
            temperature=0.0,
            max_tokens=4000,
            request_timeout=60
        ).with_structured_output(ChartAnalysis)

        logger.info(f"👁️ Vision Tool initialized. Model: {getattr(SystemConfig, 'VISION_MODEL', 'gpt-4o')} | Mode: Hybrid Forensic")

    def _process_image_bytes(self, img_bytes: bytes, max_dim: int = 3000) -> str:
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                # 1. Handle Transparency
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode != 'RGBA': img = img.convert('RGBA')
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert('RGB')

                # 2. Boost Contrast (Helps separate light bars from white background)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.4)
                
                # 3. Boost Sharpness (Helps read small axis labels)
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.5)

                # 4. Resize if too large
                width, height = img.size
                if max(width, height) > max_dim:
                    ratio = max_dim / max(width, height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise

    @traceable(name="Analyze Page (Full)", run_type="tool")
    @async_retry_with_backoff(retries=3)
    # Note: layout_hint type is 'Any' or 'PageLayout' from layout_scanner.py
    async def analyze_full_page(self, image_data: bytes, ocr_context: str, layout_hint: Any = None) -> Optional[ChartAnalysis]:
        if not image_data: return None

        loop = asyncio.get_running_loop()
        try:
            # High-fidelity processing (3000px)
            b64_img = await loop.run_in_executor(
                _IMG_EXECUTOR, self._process_image_bytes, image_data, 3000
            )
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return None

        # --- DYNAMIC PROMPT CONSTRUCTION ---
        region_instructions = "## DYNAMIC REGIONAL GUIDELINES (Virtual ROI)\n"
        
        if layout_hint and getattr(layout_hint, 'has_charts', False):
            for chart in layout_hint.charts:
                ymin, xmin, ymax, xmax = chart.bbox
                height = ymax - ymin
                width = xmax - xmin
                
                # Content Hints from Scout (The Fix)
                labels_hint = getattr(chart, 'axis_labels', [])
                values_hint = getattr(chart, 'data_values', [])
                content_hint_str = ""
                if labels_hint or values_hint:
                    content_hint_str = f"- **Content Hints:** Look for Axis Labels: {labels_hint[:10]}... and Values: {values_hint[:10]}..."

                # Check for explicit baseline from Scout
                baseline = getattr(chart, 'axis_baseline_y', None)
                
                orientation = getattr(chart, "axis_orientation", "Bottom")
                search_zone_desc = f"Within BBox {chart.bbox}"
                context_rule = "Use Anchor tags."

                if orientation == "Bottom":
                    if baseline:
                        # Ultra-precise zone around baseline (+/- 10%)
                        # This tells the VLM *exactly* where the Anchor line is.
                        axis_min = int(baseline - (height * 0.10))
                        axis_max = int(baseline + (height * 0.10))
                        search_zone_desc = f"Y-Range [{axis_min} to {axis_max}] (around Baseline Y={baseline})"
                    else:
                        # Fallback to Bottom 25% only if baseline is missing
                        axis_min = int(ymax - (height * 0.25))
                        search_zone_desc = f"Y-Range [{axis_min} to {ymax}] (Bottom 25%)"
                    
                    context_rule = f"Text above Y={axis_min} is likely [FLOATER] Context."
                    
                elif orientation == "Left":
                    # Horizontal Bar Logic
                    axis_max_x = int(xmin + (width * 0.30))
                    search_zone_desc = f"X-Range [{xmin} to {axis_max_x}] (Left 30%)"
                    context_rule = f"Text to the right of X={axis_max_x} is Context."

                region_instructions += f"""
                ### REGION {chart.region_id} ({chart.chart_type})
                - **Boundaries:** {chart.bbox}
                - **Orientation:** {orientation}
                - **Axis Zone:** {search_zone_desc}
                - **Horizontal Bleed Guard:** Do NOT map labels found significantly outside the X-range [{xmin}-{xmax}].
                {content_hint_str}
                - **Instruction:** {context_rule}
                """
        else:
            region_instructions += "No charts detected. Treat page as standard text/table layout."

        # Combine the Forensic Prompt with the Dynamic Regional Rules
        messages = [
            SystemMessage(content=PROMPT_FINANCIAL_FORENSIC + "\n\n" + region_instructions),
            HumanMessage(content=[
                {"type": "text", "text": f"[OCR SPATIAL MAP]\n{ocr_context}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ])
        ]
        
        try:
            # Returns a Pydantic object directly
            result = await self.llm.ainvoke(messages)
            
            # Log the confidence score for debugging
            if result:
                logger.info(f"🔎 VLM Extraction Complete. Confidence: {result.confidence_score:.2f} | Facts: {len(result.facts)}")
                
            return result
        except Exception as e:
            logger.error(f"Vision Analysis Failed: {e}")
            return None

vision_tool = VisionTool()