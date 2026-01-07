import base64
import io
import logging
import os
import time
import random
from functools import wraps
from pathlib import Path
from typing import Literal

# Third-party
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

# Internal
from src.config import SystemConfig, SystemPaths

logger = logging.getLogger(__name__)

# ==============================================================================
# 🛡️ RESILIENCE DECORATOR
# ==============================================================================
def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"Failed after {retries} retries: {e}")
                        return "[Error: Vision Analysis Failed]"
                    sleep = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                    logger.warning(f"Vision Error: {e}. Retrying in {sleep:.2f}s...")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

# ==============================================================================
# 🧠 SYSTEM PROMPTS (THE "SUPER-PROMPT" v2)
# ==============================================================================

PROMPT_FORENSIC_ANALYST = """
## ROLE
You are an expert Credit Underwriter and Forensic Data Analyst. Your task is to extract quantitative data from financial visualizations (Charts, Waterfalls, Heatmaps) with 100% precision.

## OBJECTIVE
Convert the visual data into a **Dense, Factual Narrative** optimized for RAG retrieval.

## ANALYSIS PROTOCOL (STRICT EXECUTION ORDER)

1. **Metadata & Period Classification (The Financial Layer):**
   - **Time Normalization:** Detect and expand abbreviated dates (e.g., "'24" -> "2024", "LTM" -> "Last Twelve Months").
   - **Historical vs. Projected:** You MUST distinguish between Actuals and Forecasts.
     - Look for Suffixes: 'E' (Estimated), 'P' (Projected), 'F' (Forecast), 'B' (Budget).
     - Look for Visuals: Dashed borders, lighter shading, or "break" lines separating history from future.
   - **Unit Inference:** If units ($, %, Millions) are not in the title, infer them from axis labels.

2. **Binding Logic (The Physics Layer):**
   - **Color Intersection (PRIMARY):** Look at the pixel color *directly underneath* the number. If a number sits on a Red background, it belongs to the "Red" category in the Legend.
   - **Leader Lines:** Trace lines connecting small floating numbers to thin segments.
   - **Legend Order (TIE-BREAKER ONLY):** - The visual stack order often matches the legend, BUT inversions are common (e.g., Legend is Top-Down, Stack is Bottom-Up).
     - Use this only if color/proximity is ambiguous.

3. **Chart-Specific Rules (The Architecture Layer):**
   - **STACKED BARS (The "Roof" Rule):** - Numbers *inside* colored blocks are **Constituents**.
     - Numbers *floating above* the bar are **Totals**.
     - **CRITICAL:** Do NOT list the "Total" as a component of itself.
   - **WATERFALL CHARTS:** - Trace the "Bridge logic": Starting Value -> [Plus/Minus Step Adjustments] -> Ending Value.
     - Distinguish between "Movement Bars" (floating) and "Subtotal Bars" (grounded).
   - **DUAL AXIS CHARTS:** - Distinguish between Left-Axis metrics (usually Bars, e.g., Revenue) and Right-Axis metrics (usually Lines, e.g., Margin %).

4. **Validation (Math & Magnitude):**
   - **Summation Check:** For Stacked/Waterfall charts, sum the extracted constituents. Does (A + B) ≈ Total? If not, check if you misidentified the Total as a Segment.
   - **Magnitude Check:** Does the largest visual block correspond to the largest extracted number?

## OUTPUT FORMAT
Provide a single text block. Do NOT use Markdown tables.
**[METADATA]:** {Title} | {Time Period Range} | {Units}
**[NARRATIVE]:**
* **Trend Overview:** A concise sentence summarizing the direction (e.g. "Revenue grew 10% YoY, with 2025 projected to accelerate").
* **Data Series:** Full sentences linking specific periods to values.
    * *Structure:* "In [Period] ([Type: Actual/Projected]), [Metric Total] was [Total Value]. This consisted of [Value A] from [Segment A] and [Value B] from [Segment B]."
* **Key Annotations:** Note explicit markers like "Projected", "Unaudited", "CAGR", or "Margins".

## CONSTRAINTS
* NO ESTIMATION: If a number is missing, write "Value not labeled."
* NO VISUAL DESCRIPTIONS: Do not say "The red bar"; say "The R&D Expense".
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
            max_tokens=1024
        )
        logger.info(f"👁️ Vision Tool initialized with model: {SystemConfig.VISION_MODEL}")

    def _resize_and_encode(self, image_path: Path, max_dim: int = 1500) -> str:
        """Resizes high-res crops to prevent token limits while maintaining legibility."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if max(width, height) > max_dim:
                    ratio = max_dim / max(width, height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise

    @traceable(name="Vision Analysis", run_type="tool")
    @retry_with_backoff(retries=3)
    def analyze(self, file_name: str, mode: str = "general", context: str = "") -> str:
        """
        Routes analysis to the correct persona based on 'mode' (chart vs general).
        """
        image_path = SystemPaths.LAYOUTS / file_name

        if not image_path.exists():
            return f"[Error: Image artifact {file_name} not found]"

        try:
            base64_image = self._resize_and_encode(image_path)
        except Exception as e:
            return f"[Error encoding image: {e}]"

        # 1. Select Persona
        if mode == "chart":
            system_prompt = PROMPT_FORENSIC_ANALYST
            # Keep temp=0 for strict data extraction
            self.vlm.temperature = 0.0
        else:
            system_prompt = PROMPT_STRATEGIC_CONSULTANT
            # Slight creativity for diagrams
            self.vlm.temperature = 0.2

        # 2. Build Contextual Prompt
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": f"Additional Context (OCR/Colors):\n{context}\n\nAnalyze this image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]
            )
        ]

        # 3. Execute via LangChain (Auto-Traced)
        response = self.vlm.invoke(messages)
        return response.content

vision_tool = VisionTool()