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
from openai import OpenAI
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
# 🧠 SYSTEM PROMPTS (RAG-OPTIMIZED)
# ==============================================================================

PROMPT_FORENSIC_ANALYST = """
## ROLE
You are an expert Credit Underwriter and Forensic Data Analyst. Your task is to extract quantitative data from financial visualizations with 100% precision.

## OBJECTIVE
Convert the visual data into a **Dense, Factual Narrative**. Your output must be optimized for RAG embeddings.

## ANALYSIS PROTOCOL (STRICT EXECUTION ORDER)
1. **Metadata Scan:** Identify Title, Time Period, and Unit of Measure (e.g., $ Millions).
2. **Binding Logic (Match Numbers to Labels):**
   - **Primary (Color Intersection):** Look at the pixel color *directly underneath* the number. If a number sits on a Red background, it belongs to the Red category from the legend.
   - **Secondary (Proximity):** If a number is floating (e.g., small values with a leader line), trace the line or proximity to the nearest colored segment.
   - **Tertiary (Magnitude Check):** The largest number *must* correspond to the largest visual segment.
3. **"Total" Identification:** Identify numbers floating *above* the stack. These are usually "Totals" and must be labeled as such.
4. **Validation:** Sum the constituent segments. Does (A + B) ≈ Total? If not, note the discrepancy.

## OUTPUT FORMAT
Provide a single text block. Do NOT use Markdown tables.
**[METADATA]:** {Title} | {Period} | {Units}
**[NARRATIVE]:**
* **Trend Overview:** A concise sentence summarizing the direction.
* **Data Series:** Full sentences linking periods to values.
    * *Example:* "In Q1 24, Total Operating Expenses were $40,000M. This comprised $23,267M in R&D (Red), $6,172M in S&M (Orange), and $3,539M in G&A (Pink)."
* **Key Annotations:** Note explicit markers like "Projected" or "Unaudited."

## CONSTRAINTS
* NO ESTIMATION: If a number is missing, write "Value not labeled."
* NO COLOR DESCRIPTIONS IN OUTPUT: Do not say "The red bar"; say "The R&D Expense".
"""

PROMPT_STRATEGIC_CONSULTANT = """
## ROLE
You are a Senior Strategic Consultant performing Due Diligence. Your task is to interpret structural diagrams, organizational hierarchies, and process flows.

## OBJECTIVE
Translate 2D visual relationships into a **Structured Semantic Summary** that captures hierarchy, sequence, and causality.

## ANALYSIS PROTOCOL
1. **Classify Structure:** (Org Charts, Process Flows, SWOT, Quadrants).
2. **Semantic Mapping:** Convert visual cues into meaning (e.g., "Dotted line implies indirect reporting").

## OUTPUT FORMAT
**[TYPE]:** {e.g., Org Chart, Process Flow}
**[TITLE]:** {Title}
**[STRUCTURED SUMMARY]:**
* **Strategic Takeaway:** One sentence on what this visual implies about the business.
* **Detailed Mapping:** Use indented bullets or numbered steps to represent the flow/hierarchy.
"""

# ==============================================================================
# 👁️ VISION TOOL
# ==============================================================================
class VisionTool:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = SystemConfig.VISION_MODEL
        logger.info(f"👁️ Vision Tool initialized with model: {self.model}")

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
            temp = 0.0 # Strict
        else:
            system_prompt = PROMPT_STRATEGIC_CONSULTANT
            temp = 0.2 # Creative

        # 2. Build Contextual Prompt
        user_content = [
            {"type": "text", "text": f"Additional Context (OCR/Colors):\n{context}\n\nAnalyze this image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]

        # 3. Execute
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temp,
            max_tokens=1000,
        )
        return response.choices[0].message.content

vision_tool = VisionTool()