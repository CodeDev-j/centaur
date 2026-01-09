import base64
import io
import logging
import asyncio
import random
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# Third-party
from PIL import Image, ImageEnhance
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

# Internal
from src.config import SystemConfig

logger = logging.getLogger(__name__)

# Dedicated thread pool for CPU-bound image operations to avoid blocking the Event Loop
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
                        return f"[Error: Vision Analysis Failed - {str(e)}]"
                    
                    # [FIX] Non-blocking sleep allows other tasks to run during backoff
                    sleep_time = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                    logger.warning(f"Vision Error: {e}. Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator

# ==============================================================================
# 🧠 SYSTEM PROMPTS (THE "HYBRID" MODEL)
# ==============================================================================

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
        # Use LangChain wrapper for better tracing in LangSmith
        self.vlm = ChatOpenAI(
            model=SystemConfig.VISION_MODEL,
            temperature=0.0,
            max_tokens=1024,
            # Ensure timeout to prevent hanging connections
            request_timeout=60
        )
        logger.info(f"👁️ Vision Tool initialized with model: {SystemConfig.VISION_MODEL}")

    def _process_image_bytes(self, img_bytes: bytes, max_dim: int = 2000) -> str:
        """
        CPU-bound processing: Resize, Enhance, and Encode to JPEG.
        Executed in a thread pool to avoid blocking the async event loop.
        """
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                # 1. Handle Transparency (Safety for RGBA -> JPEG)
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
    async def analyze(self, image_data: bytes, mode: str = "general", context: str = "") -> str:
        """
        Fully async entry point. 
        Accepts bytes (Stateless) instead of file paths.
        """
        if not image_data: return "[Error: No image data provided]"

        # 1. Offload CPU work to ThreadPool
        loop = asyncio.get_running_loop()
        try:
            base64_image = await loop.run_in_executor(
                _IMG_EXECUTOR, self._process_image_bytes, image_data
            )
        except Exception as e:
            return f"[Error encoding image: {e}]"

        # 2. Select Persona
        system_prompt = PROMPT_FORENSIC_ANALYST if mode == "chart" else PROMPT_STRATEGIC_CONSULTANT
        temp = 0.0 if mode == "chart" else 0.2

        # 2. Build Contextual Prompt
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": f"Context:\n{context}\n\nAnalyze:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]
            )
        ]

        # 3. Async Network Call
        response = await self.vlm.ainvoke(messages, config={"temperature": temp})
        return response.content

vision_tool = VisionTool()