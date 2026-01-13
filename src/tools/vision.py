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
from src.prompts import PROMPT_FINANCIAL_FORENSIC

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

# 1. ATOMIC DATA POINT (Quantitative)
class DataPoint(BaseModel):
    """Represents a single, atomic numeric value within a series."""
    label: str = Field(..., description="The X-axis label or category (e.g., '2023', 'Q1 2024').")
    numeric_value: float = Field(..., description="The cleaned numeric value. Convert '1,278' to 1278.0.")
    unit: str = Field(..., description="The unit of measurement (e.g., 'bps', 'USD Million').")
    
    # Contextual flags
    category: Literal["Metric", "Time"] = Field(
        default="Metric", 
        description="Strictly for quantitative data points."
    )
    is_forecast: bool = Field(False, description="True if this point is marked as (E), Forecast, or Projected.")
    
    # CRITICAL: Chain-of-Thought Enforcement
    reasoning: str = Field(..., description="Audit: Cite OCR 'Col' IDs, [ANCHOR] tags, and Visual/Math logic used to lock this value.")
    
    grounding: BoundingBox = Field(..., description="The bounding box of this specific number/bar.")

# 2. METRIC SERIES (The Container)
class MetricSeries(BaseModel):
    """A collection of data points representing one specific metric."""
    series_label: str = Field(..., description="The name of the metric or legend key (e.g., 'Levered Lending').")
    data_points: List[DataPoint] = Field(..., description="List of atomic data points for this series.")

# 3. QUALITATIVE INSIGHT (Qualitative)
class QualitativeInsight(BaseModel):
    """For non-numeric concepts, strategy pillars, or abstract diagrams."""
    topic: str = Field(..., description="The main header or concept (e.g., 'Ratings Procedures').")
    content: str = Field(..., description="The descriptive text or bullet point content.")
    
    category: Literal[
        "Risk",         # Explicit headwinds
        "Strategy",     # Qualitative Pillars ONLY
        "Structure",    # Concept Maps / Arrows
        "Entity",       # Company Name
        "Governance"    # Regulatory/Compliance
    ] = Field(..., description="The semantic category of this insight.")

    # CRITICAL: Chain-of-Thought Enforcement
    reasoning: str = Field(..., description="Audit: Explain why this text block was selected and how it relates to the category.")
    
    grounding: BoundingBox = Field(..., description="The bounding box of the text block.")

class ChartAnalysis(BaseModel):
    title: str = Field(..., description="The chart or slide title")
    summary: str = Field(..., description="Forensic narrative of the trends and logic trace")
    
    # Confidence Score for Logic Gating
    confidence_score: float = Field(..., description="Self-evaluation (0.0-1.0). <0.7 implies ambiguity or blurriness.")
    
    # THE AUDIT LOG (Force "System 2" Thinking)
    audit_log: List[str] = Field(..., description="List of visual traps detected, math checks performed, and logic resolutions.")

    
    # SPLIT OUTPUT
    metrics: List[MetricSeries] = Field(default=[], description="Structured time-series data for charts/tables.")
    insights: List[QualitativeInsight] = Field(default=[], description="Abstract concepts for strategy/text slides.")

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
            request_timeout=180 # <--- UPDATED: Increased to 3 minutes
        ).with_structured_output(ChartAnalysis)

        logger.info(f"👁️ Vision Tool initialized. Model: {getattr(SystemConfig, 'VISION_MODEL', 'gpt-4o')} | Mode: Hybrid Forensic")

    def _process_image_bytes(self, img_bytes: bytes, max_dim: int = 3000) -> str:
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                # 1. Handle Transparency
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
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
        if not image_data:
            return None

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
                        # Adaptive Geometry: If baseline is known, creating a very tight zone (+/- 5%)
                        # This explicitly excludes "Driver Text" floating slightly above.
                        axis_min = int(baseline - (height * 0.05)) 
                        axis_max = int(baseline + (height * 0.15)) # Bias down for labels
                        search_zone_desc = f"Y-Range [{axis_min} to {axis_max}] (Strict Baseline at Y={baseline})"
                    else:
                        # Fallback to Bottom 25% only if baseline is missing
                        axis_min = int(ymax - (height * 0.25))
                        search_zone_desc = f"Y-Range [{axis_min} to {ymax}] (Bottom 25%)"
                    
                    context_rule = f"Text above Y={axis_min} is likely [FLOATER] Context or Driver Text."
                    
                elif orientation == "Left":
                    # Horizontal Bar Logic
                    axis_max_x = int(xmin + (width * 0.30))
                    search_zone_desc = f"X-Range [{xmin} to {axis_max_x}] (Left 30%)"
                    context_rule = f"Text to the right of X={axis_max_x} is Context."

                # Explicitly warn the VLM about variable width charts using the flag from LayoutScanner
                is_variable = getattr(chart, 'is_variable_width', False)
                mekko_warning = ""
                if is_variable:
                    mekko_warning = (
                        "\n- **WARNING: VARIABLE WIDTH CHART DETECTED.**\n"
                        "  - **FORCE MODE A (Attribute Locking).**\n"
                        "  - Standard Column alignment rules are SUSPENDED.\n"
                        "  - Do NOT infer relationships based on vertical grid alignment.\n"
                        "  - Rely strictly on COLOR TAGS and LEGEND KEYS to link data."
                    )

                region_instructions += f"""
                ### REGION {chart.region_id} ({chart.chart_type}){mekko_warning}
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
                # Count total points for logging
                total_pts = sum(len(m.data_points) for m in result.metrics)
                logger.info(f"🔎 VLM Extraction Complete. Confidence: {result.confidence_score:.2f} | Metrics: {len(result.metrics)} | DataPoints: {total_pts} | Insights: {len(result.insights)}")
                
            return result
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return ChartAnalysis(
                title="Error", 
                summary="Extraction Failed", 
                audit_log=[str(e)], 
                confidence_score=0.0,
                metrics=[],
                insights=[]
            )

vision_tool = VisionTool()