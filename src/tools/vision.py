import base64
import io
import logging
import asyncio
import random
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field

# Third-party
from PIL import Image, ImageEnhance
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

# Internal
from src.config import SystemConfig
from src.prompts import PROMPT_FINANCIAL_FORENSIC
from src.tools.layout_scanner import PageLayout  # Needed for type hinting

logger = logging.getLogger(__name__)

# Dedicated thread pool for CPU-bound image operations
_IMG_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# ==============================================================================
# 📊 STRUCTURED OUTPUT SCHEMAS
# ==============================================================================
class DataPoint(BaseModel):
    label: str = Field(..., description="The X-axis label (e.g., '2024', 'Q1', 'Bridge').")
    
    # 1. The Raw Number (Strictly no math)
    numeric_value: float = Field(..., description="The numeric value exactly as seen. Example: If text is '$12.4B', extract 12.4.")
    
    # 2. Currency (Strict List - Liquid 15)
    # Critical for FX normalization downstream.
    currency: Literal[
        "USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "CHF", "INR", 
        "HKD", "SGD", "NZD", "KRW", "SEK", "BRL", 
        "None"
    ] = Field(
        default="None", 
        description="ISO 4217 Code. Infer from symbol context (e.g. '$' in Singapore -> SGD)."
    )
    
    # 3. Magnitude (Strict List - Universal Scalers)
    # Critical to prevent 'zero-counting' hallucinations.
    magnitude: Literal[
        "k", "M", "B", "T", # Financials
        "G", "P",           # Energy/Tech (Giga, Peta)
        "%", "x", "bps",    # Ratios
        "None"
    ] = Field(
        default="None", 
        description="Scaler. Map 'mn/mm'->M, 'bn'->B. Map 'GW'->Mag='G', Measure='W'."
    )
    
    # 4. Measure (Flexible String - The "Escape Hatch")
    # Allows for niche KPIs (TEU, boe, oz, subscribers) without schema errors.
    measure: str = Field(
        default="None", 
        description="Operational unit (e.g., 'sqft', 'bbl', 'users', 't'). Keep concise."
    )
    
    original_text: str = Field(default="", description="The raw text seen on the chart for audit (e.g. '$12.4B').")

class MetricSeries(BaseModel):
    series_label: str = Field(..., description="The name of the series (from Legend or Label).")
    data_points: List[DataPoint] = Field(default_factory=list)

class Insight(BaseModel):
    category: str = Field(..., description="Type: 'Trend', 'Risk', 'Anomaly', 'Strategy'.")
    topic: str = Field(..., description="Subject (e.g., 'EBITDA Growth').")
    content: str = Field(..., description="The concise insight derived from the visual.")

class ChartAnalysis(BaseModel):
    title: str = Field(default="Untitled Analysis", description="Title of the analysis.")
    summary: str = Field(..., description="High-level summary of the visual data.")
    
    # [UPDATED] Periodicity Field with 9M and W
    periodicity: Literal[
        "FY",   # Fiscal Year / Annual
        "Q",    # Quarterly
        "H",    # Half-Yearly (H1/H2)
        "9M",   # Nine Months (Common in Q3 reporting)
        "M",    # Monthly
        "W",    # Weekly
        "LTM",  # Last Twelve Months
        "YTD",  # Year to Date
        "Other",
        "Unknown"
    ] = Field(
        default="Unknown",
        description="The time basis of the data. FY=Fiscal Year, Q=Quarterly, 9M=Nine Months, LTM=Last 12 Months."
    )
    
    metrics: List[MetricSeries] = Field(default_factory=list, description="Extracted quantitative data.")
    insights: List[Insight] = Field(default_factory=list, description="Qualitative strategic takeaways.")
    audit_log: str = Field(..., description="Step-by-step reasoning of how data was extracted.")
    confidence_score: float = Field(..., description="0.0-1.0 certainty level.")

# ==============================================================================
# 👁️ VISION TOOL
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
                        return None # Return None on failure to allow graceful degradation
                    
                    # [FIX] Non-blocking sleep allows other tasks to run during backoff
                    sleep_time = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                    logger.warning(f"Vision Error: {e}. Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator

class VisionTool:
    def __init__(self):
        # Use LangChain wrapper for better tracing in LangSmith
        self.vlm = ChatOpenAI(
            model=SystemConfig.VISION_MODEL,
            temperature=0.0,
            max_tokens=3500,  # Higher token limit for dense data extraction
            request_timeout=300 
        ).with_structured_output(ChartAnalysis)
        
        logger.info(f"👁️ Vision Tool initialized with model: {SystemConfig.VISION_MODEL}")

    def _process_image_bytes(self, img_bytes: bytes, max_dim: int = 3000) -> str:
        """
        CPU-bound processing: Resize, Enhance, and Encode to JPEG.
        Executed in a thread pool to avoid blocking the async event loop.
        """
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

                # 2. Boost Contrast (Helps separate light bars/gridlines from white background)
                # Factor 1.4 = +40% Contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.4)
                
                # 3. Boost Sharpness (Helps OCR read small axis labels)
                # Factor 1.5 = +50% Sharpness
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.5)
                # ---------------------------------

                # Standard Resize Logic
                # [FIX] INCREASED MAX_DIM TO 3000
                # Previous 1500 limit was shrinking the 3x scaled images, causing fuzziness.
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

    @traceable(name="Vision Analysis", run_type="tool")
    @async_retry_with_backoff(retries=3)
    async def analyze_full_page(self, 
                              image_data: bytes, 
                              ocr_context: str, 
                              layout_hint: PageLayout) -> Optional[ChartAnalysis]:
        """
        FORENSIC ANALYSIS ENTRY POINT
        Accepts the Full Page + Spatial Grid + Scout Hints.
        """
        if not image_data:
            return None

        # 1. Offload Image Processing
        loop = asyncio.get_running_loop()
        try:
            base64_image = await loop.run_in_executor(
                _IMG_EXECUTOR, self._process_image_bytes, image_data
            )
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None

        # 2. Construct the Forensic Context Block
        # Convert Pydantic Layout object to a string summary for the Prompt
        scout_summary = "SCOUT FINDINGS:\n"
        if layout_hint and layout_hint.charts:
            for i, chart in enumerate(layout_hint.charts):
                scout_summary += (
                    f"- Region {i+1}: {chart.chart_type} | Title: '{chart.title}' | "
                    # Inject the Scout's detected legends to guide the model
                    f"Legend Keys: {chart.legend_keys} | " 
                    f"Possible Values: {chart.constituents} | "
                    f"Possible Labels: {chart.axis_labels}\n"
                )
        else:
            scout_summary += "No specific charts detected, scan for embedded data tables or text logic."

        final_prompt_context = (
            f"=== STEP 1: SPATIAL ANCHORS (OCR) ===\n{ocr_context[:5000]}\n\n"  # Truncate if massive
            f"=== STEP 2: SCOUT HINTS (VISUALS) ===\n{scout_summary}\n\n"
            f"=== INSTRUCTION ===\n"
            f"Use the Spatial Anchors to lock onto the Scout's regions. "
            f"Extract the 'Ground Truth' numbers. "
            f"Detect the Periodicity (FY, Q, 9M, LTM) from the title/headers. "
            f"If OCR and Visuals disagree, TRUST THE VISUALS (Bar Height/Position)."
        )

        # 3. Build Messages
        messages = [
            SystemMessage(content=PROMPT_FINANCIAL_FORENSIC),
            HumanMessage(
                content=[
                    {"type": "text", "text": final_prompt_context},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]
            )
        ]

        # 4. Invoke VLM
        response = await self.vlm.ainvoke(messages)
        return response

vision_tool = VisionTool()