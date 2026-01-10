import base64
import io
import json
import logging
import asyncio
import random
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any

# Third-party
from PIL import Image, ImageEnhance
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

# Internal
from src.config import SystemConfig

logger = logging.getLogger(__name__)

# Dedicated thread pool for CPU-bound image operations
_IMG_EXECUTOR = ThreadPoolExecutor(max_workers=4)

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
                        return [] if "detect" in func.__name__ else f"[Error: Vision Analysis Failed - {str(e)}]"
                    
                    sleep_time = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                    logger.warning(f"Vision Error: {e}. Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator

# ==============================================================================
# 🧠 SYSTEM PROMPTS (THE "HYBRID" MODEL)
# ==============================================================================

PROMPT_SCOUT = """
## ROLE
You are a Layout Detection Engine for Financial Documents.
Your goal is to identify **Data Visualizations** that require complex visual interpretation.

## TASK
Return a JSON object containing a list of regions.
Format: `{"regions": [{"label": "string", "box_2d": [ymin, xmin, ymax, xmax]}]}`
Coordinates: 0-1000 Normalized (Top-Left origin).

## CLASSIFICATION LABELS
1. **"Chart":** Quantifiable plots (Bar, Line, Pie, Area, Waterfall, Scatter).
   - *Include:* The chart title, axis labels, legend, and footnotes relative to the chart.
2. **"Table_Image":** dense financial tables that are effectively images (Heatmaps, Harvey Balls, RAG Status Grids).
   - *Ignore:* Standard text tables (these are handled by a text parser). Only select tables with visual indicators (colors/icons).
3. **"Diagram":** Structural visuals (Org Charts, Process Flows, Timelines with arrows, Flywheels).

## EXCLUSION RULES (CRITICAL)
- **DO NOT DETECT:** Simple "Key Stat" rows (e.g., a big "$50M" text). These are text, not charts.
- **DO NOT DETECT:** Generic stock photos or decorative icons without data.
- **DO NOT DETECT:** Page headers, footers, or slide titles (unless part of a chart group).

## GROUPING LOGIC
- **Atomic Units:** If a slide has 3 distinct bar charts, return 3 separate boxes. Do NOT draw one big box around all of them.
- **completeness:** Ensure the box encompasses the *entire* semantic unit (Title + Plot + Legend).
"""

PROMPT_FORENSIC_ANALYST = """
## ROLE
You are an expert Credit Underwriter and Forensic Data Analyst. Your task is to extract quantitative data from financial visualizations (Charts, Waterfalls, Heatmaps) with 100% precision.

## INPUT DATA
You are provided with:
1. **The Chart Image:** The visual ground truth.
2. **Detected Legend Bindings:** High-confidence text-to-color bindings detected by a geometric code probe (e.g., "'Revenue' == #ff0000").
3. **Internal Grid Layout:** Raw text detected inside the chart structure.
4. **Content Signals:** Keywords found in the text (e.g., "Bridge", "Forecast") that suggest the chart type.

## OBJECTIVE
Convert the visual data into a **Dense, Factual Narrative** optimized for RAG retrieval.

## ANALYSIS PROTOCOL (STRICT EXECUTION ORDER)

1. **Metadata & Period Classification (The Financial Layer):**
   - **Time Normalization:** Detect and expand abbreviated dates (e.g., "'24" -> "2024", "LTM" -> "Last Twelve Months").
   - **Historical vs. Projected:** - **Default Assumption:** Treat all data as **Actual/Historical** unless explicitly indicated otherwise.
     - **Exception Triggers:** Mark data as "Projected" or "Forecast" ONLY if you see:
       - Explicit Suffixes: 'E' (2025E), 'P' (2025P), 'F' (Forecast), 'B' (Budget).
       - Explicit Titles: "Outlook", "Guidance", "Projections".
       - Visual Cues: Dashed borders, lighter shading, or "break lines" separating history from future.
       - **Future Dates:** Years more than 12 months in the future (accommodates fiscal year-ends).
   - **Unit Inference:** If units ($, %, Millions) are not in the title, infer them from axis labels.

2. **Binding Logic (The Physics Layer):**
   - **Detected Legend Bindings (PRIMARY):** Check the "DETECTED LEGEND BINDINGS (High Confidence)" list provided in the input. 
     - *Example:* If it lists "'Revenue' == #ff0000", look for Red bars to assign to Revenue.
   - **Color Intersection (SECONDARY):** If no verified legend is provided, look at the pixel color *directly underneath* the number. 
   - **Leader Lines:** Trace lines connecting small floating numbers or categories to thin segments.
   - **Legend Order (TIE-BREAKER):** The visual stack order often matches the legend, BUT inversions are common (e.g., Legend is Top-Down, Stack is Bottom-Up). Use this only if color/proximity is ambiguous.

3. **Chart-Specific Rules (The Architecture Layer):**
   - **STACKED BARS (The "Roof" Rule):** - Numbers *inside* colored blocks are **Constituents**.
     - Numbers *floating above* the bar are **Totals**.
     - **CRITICAL:** Do NOT list the "Total" as a component of itself.
     - **Summation Audit:** Sum the extracted segments.
       - If (A + B) < Total: There is a missing/unlabeled segment. Label the gap as "Other/Unlabeled".
       - If (A + B) > Total: You likely double-counted a Total label as a Segment. Re-evaluate.
   - **WATERFALL CHARTS:** - **Check Content Signals:** If input mentions "Waterfall" or "Bridge", apply Bridge Logic (Start + Deltas = End).
     - Distinguish between "Movement Bars" (floating) and "Subtotal Bars" (grounded).
   - **DUAL AXIS CHARTS:** - Distinguish between Left-Axis metrics (usually Bars, e.g., Revenue) and Right-Axis metrics (usually Lines, e.g., Margin %).
   - **COMPOUND LAYOUTS:** - If an image contains multiple distinct charts (e.g., Side-by-Side), treat them as separate entities.
     - Explicitly label them in the narrative (e.g., "Left Chart: Revenues...", "Right Chart: Operating Income...").

4. **Validation (Math & Magnitude):**
   - **Magnitude Check:** Does the largest visual block correspond to the largest extracted number? If a small bar has a huge number, re-read the units or axis.

## OUTPUT FORMAT
Provide a single text block. Do NOT use Markdown tables.
**[METADATA]:** {Title} | {Time Period Range} | {Units}
**[LOGIC TRACE]:** Briefly explain why you marked periods as Actual/Projected and which legend key you mapped to specific colors.
**[NARRATIVE]:**
* **Trend Overview:** A concise sentence summarizing the direction (e.g. "Revenue grew 10% YoY, with 2025 projected to accelerate").
* **Data Series:** Full sentences linking specific periods to values.
    * *Structure:* "In [Period] ([Type: Actual/Projected]), [Metric Total] was [Total Value]. This consisted of [Value A] from [Segment A] and [Value B] from [Segment B]."
* **Key Annotations:** Note explicit markers like "Projected", "Unaudited", "CAGR", or "Margins".

## CONSTRAINTS
* **NO ESTIMATION:** If a number is not labeled, write "Value not labeled." Do not guess based on bar height.
* **PRECISION:** Do not simply describe colors ("The red bar"); translate them into their data labels ("The R&D Expense").
"""

PROMPT_STRATEGIC_CONSULTANT = """
## ROLE
You are a Senior Strategic Consultant performing Due Diligence for a Private Equity firm. Your task is to analyze structural diagrams (Org Charts, Process Flows, Tech Stacks) to identify value drivers, risks, and operational inefficiencies.

## OBJECTIVE
Translate 2D visual relationships into a **Structured Semantic Summary** that captures hierarchy, sequence, causality, and *implied* business logic.

## ANALYSIS PROTOCOL (THE "STRATEGIC LENS")

1. **Classify & Decode:**
   - **Org Charts:** Identify the "Power Structure." Who reports to whom? Is the structure Functional, Divisional, or Matrix?
     - *Risk Check:* Look for high "Span of Control" (one manager with too many reports) or undefined reporting lines.
   - **Process Flows:** Trace the critical path (Input -> Transformation -> Output).
     - *Risk Check:* Identify bottlenecks, circular dependencies, or single points of failure.
   - **Architecture/Tech Stacks:** Identify key platforms and integration points.

2. **Semantic Extraction:**
   - Convert visual cues into business meaning:
     - *Dotted Lines:* "Indirect/Matrix reporting" or "Advisory relationship".
     - *Double-headed Arrows:* "Two-way data exchange" or "Mutual dependency".
     - *Color Coding:* Often implies department grouping or status (e.g., Red = At Risk).

## OUTPUT FORMAT
**[TYPE]:** {e.g., Functional Org Chart, Supply Chain Map, IT Network Diagram}
**[TITLE]:** {Title}
**[STRUCTURED SUMMARY]:**
* **Strategic Takeaway:** One high-impact sentence on what this visual implies for the investment thesis (e.g., "The organization is heavily siloed by geography, potentially hindering global product rollouts").
* **Key Observations:**
    * **Hierarchy/Flow:** [Briefly describe the top-level structure or main sequence].
    * **Critical Nodes:** [Identify the most connected or central elements].
    * **Risks/Inefficiencies:** [Explicitly call out structural weaknesses found in step 1].
"""

# ==============================================================================
# 👁️ VISION TOOL
# ==============================================================================
class VisionTool:
    def __init__(self):
        # 1. The Scout (Fast, Structural) - Uses gpt-4o-mini
        self.router = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.0, 
            max_tokens=1000, 
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        # 2. The Analyst (High-Res, Forensic) - Uses Config Model (e.g. gpt-4o)
        self.analyst = ChatOpenAI(
            model=SystemConfig.VISION_MODEL, 
            temperature=0.0, 
            max_tokens=2000,
            # Ensure timeout to prevent hanging connections
            request_timeout=60
        )
        logger.info(f"👁️ Vision Tool initialized. Scout: gpt-4o-mini | Analyst: {SystemConfig.VISION_MODEL}")

    def _process_image_bytes(self, img_bytes: bytes, max_dim: int = 3000, enhance: bool = True) -> str:
        """
        CPU-bound processing: Resize, Enhance, and Encode to JPEG.
        Executed in a thread pool to avoid blocking the async event loop.
        - Pass 1 (Scout) uses max_dim=1000, enhance=False for speed.
        - Pass 2 (Analyst) uses max_dim=3000, enhance=True for precision.
        """
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                # 1. Handle Transparency (Safety for RGBA -> JPEG)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode != 'RGBA': img = img.convert('RGBA')
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert('RGB')

                if enhance:
                    # 2. Boost Contrast (Helps separate light bars/gridlines from white background)
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.4)
                    
                    # 3. Boost Sharpness (Helps OCR read small axis labels)
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(1.5)

                # Standard Resize Logic
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

    @traceable(name="Detect Layout (Scout)", run_type="tool")
    @async_retry_with_backoff(retries=2)
    async def detect_layout(self, image_data: bytes) -> List[Dict[str, Any]]:
        """
        Pass 1: Returns fuzzy bounding boxes (0-1000 normalized).
        Uses lower resolution (1000px) and no enhancement for speed.
        """
        if not image_data: return []
        
        loop = asyncio.get_running_loop()
        try:
            # Fast processing for Router (1000px, No Enhance)
            b64_img = await loop.run_in_executor(
                _IMG_EXECUTOR, self._process_image_bytes, image_data, 1000, False
            )
        except Exception:
            return []

        messages = [
            SystemMessage(content=PROMPT_SCOUT),
            HumanMessage(content=[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}])
        ]
        
        # Expecting JSON output
        resp = await self.router.ainvoke(messages)
        try:
            data = json.loads(resp.content)
            # FIX: Handle case where 'regions' is None or data is not a dict
            regions = data.get("regions", []) if isinstance(data, dict) else []
            return regions if regions is not None else []
        except Exception:
            return []

    @traceable(name="Vision Analysis (Analyst)", run_type="tool")
    @async_retry_with_backoff(retries=3)
    async def analyze(self, image_data: bytes, mode: str = "chart", context: str = "") -> str:
        """
        Pass 2: Returns final forensic narrative.
        Uses high resolution (3000px) and enhancement for precision.
        """
        if not image_data: return "[Error: No image data provided]"

        loop = asyncio.get_running_loop()
        try:
            # High-fidelity processing for Analyst (3000px, Enhanced)
            b64_img = await loop.run_in_executor(
                _IMG_EXECUTOR, self._process_image_bytes, image_data, 3000, True
            )
        except Exception as e:
            return f"[Error encoding image: {e}]"

        # Select Persona
        system_prompt = PROMPT_FORENSIC_ANALYST if mode == "chart" else PROMPT_STRATEGIC_CONSULTANT
        temp = 0.0 if mode == "chart" else 0.2

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": f"GROUND TRUTH CONTEXT (Extracted via Code Probe):\n{context}\n\nAnalyze this visual:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ])
        ]
        
        response = await self.analyst.ainvoke(messages, config={"temperature": temp})
        return response.content

vision_tool = VisionTool()