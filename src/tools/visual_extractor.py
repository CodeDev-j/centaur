"""
Visual Extractor: The Forensic "Eye" of the Pipeline.

RESPONSIBILITY:
This module bridges the gap between raw pixels and structured financial data.
It orchestrates the VLM (Visual Language Model) to perform "Forensic Extraction"
governed by the strict schemas defined in `src.schemas.vision_output`.

ARCHITECTURAL FLOW:
1. Pre-Processing: Optimizes image contrast/sharpness for OCR legibility.
2. Context Injection: Fuses "Layout Analyzer" hints + "OCR" anchors (Text) 
   into a single "Cognitive Prompt".
3. Structured Inference: Forces the VLM to fill a Pydantic schema (`PageLayout`),
   enforcing the "Audit First" (Chain of Thought) protocol.
"""

# ==============================================================================
# 👁️ VISUAL EXTRACTOR - visual_extractor.py
# ==============================================================================

import asyncio
import atexit
import base64
import io
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Optional

# Third-party
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from PIL import Image, ImageEnhance

# Internal
from src.config import SystemConfig
from src.prompts import PROMPT_VISUAL_EXTRACTOR

# SCHEMA ALIASING:
# We distinguish between the "Hint" (from Layout Analyzer) and the "Result" (from Visual Extractor).
from src.schemas.layout_output import PageLayout as LayoutHint 
from src.schemas.vision_output import PageLayout as VisionResult

logger = logging.getLogger(__name__)

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
VISION_TEMPERATURE = 0.0
VISION_MAX_TOKENS = 5000
VISION_TIMEOUT = 300      # seconds
RETRY_COUNT = 3
RETRY_BACKOFF = 1         # seconds

# ==============================================================================
# ⚙️ RESOURCE MANAGEMENT
# ==============================================================================
# Determine worker count based on CPU cores, capped at 4 for image ops
_MAX_WORKERS = min(4, (os.cpu_count() or 1))
_IMG_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS)

def _shutdown_executor():
    """Ensure thread pool is closed on program exit."""
    _IMG_EXECUTOR.shutdown(wait=False)

atexit.register(_shutdown_executor)


# ==============================================================================
# 🛡️ RESILIENCE UTILS
# ==============================================================================
def async_retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(
                            f"❌ Visual Extraction Failed after {retries} retries. "
                            f"Error: {e}"
                        )
                        return None  # Return None to allow graceful degradation

                    # Jittered backoff to prevent thundering herd
                    sleep_time = (
                        (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                    )
                    logger.warning(
                        f"⚠️ VLM Glitch: {e}. Retrying in {sleep_time:.2f}s..."
                    )
                    await asyncio.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator


# ==============================================================================
# 👁️ VISUAL EXTRACTOR TOOL
# ==============================================================================
class VisualExtractor:
    def __init__(self):
        # 1. Initialize the Base LLM via Factory
        base_llm = SystemConfig.get_llm(
            model_name=SystemConfig.VISION_MODEL,
            temperature=VISION_TEMPERATURE,
            max_tokens=VISION_MAX_TOKENS,
            timeout=VISION_TIMEOUT
        )

        # 2. Bind the "Cognitive Contract" (Structured Output)
        # We force the model to adhere to the `VisionResult` (PageLayout) schema.
        self.vlm = base_llm.with_structured_output(VisionResult)

        logger.info(
            f"👁️ Visual Extractor initialized. "
            f"Model: {SystemConfig.VISION_MODEL} | Mode: {SystemConfig.DEPLOYMENT_MODE}"
        )

    def _process_image_bytes(
        self,
        img_bytes: bytes,
        max_dim: int = 3000
    ) -> str:
        """
        CPU-bound processing: Resize, Enhance, and Encode to JPEG.
        Executed in a separate thread to avoid blocking the Async Event Loop.
        """
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                # 1. Handle Transparency
                if img.mode in ('RGBA', 'LA') or \
                   (img.mode == 'P' and 'transparency' in img.info):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert('RGB')

                # 2. Boost Contrast (Helps separate light bars/gridlines)
                # Factor 1.4 = +40% Contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.4)

                # 3. Boost Sharpness (Helps OCR read small axis labels)
                # Factor 1.5 = +50% Sharpness
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.5)
                # ---------------------------------

                # Resize Logic
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

    @traceable(name="Visual Extraction", run_type="tool")
    @async_retry_with_backoff(
        retries=RETRY_COUNT,
        backoff_in_seconds=RETRY_BACKOFF
    )
    async def analyze_full_page(
        self,
        image_data: bytes,
        ocr_context: str,
        layout_hint: Optional[LayoutHint]  # The "Analyzer" Hint
    ) -> Optional[VisionResult]:         # The "Forensic" Result
        """
        FORENSIC ANALYSIS ENTRY POINT.
        
        Orchestrates the "Cognitive Sandwich":
        1. Context: Injects OCR Grid + Layout Hints.
        2. Visuals: Injects the Enhanced Image.
        3. Logic: Injects the System Prompt with strict Protocols.
        """
        if not image_data:
            logger.warning("⚠️ No image data provided to VisualExtractor.")
            return None

        # 1. Offload Image Processing
        loop = asyncio.get_running_loop()
        try:
            base64_image = await loop.run_in_executor(
                _IMG_EXECUTOR, self._process_image_bytes, image_data
            )
        except Exception as e:
            logger.error(f"❌ Error encoding image: {e}")
            return None

        # 2. Construct the "Scout Dossier" (Context Injection)
        # We transform the LayoutHint object into a dense text summary for the VLM.
        scout_summary = "SCOUT FINDINGS (VISUAL HINTS):\n"
        if layout_hint and layout_hint.charts:
            for i, chart in enumerate(layout_hint.charts):
                scout_summary += (
                    f"- Region {i+1} (Box: {chart.bbox}): "
                    f"{chart.chart_type} | "
                    f"Title: '{chart.title}' | "
                    f"Legend: {chart.legend_keys} | "
                    f"X-Axis: {chart.x_axis_labels} | "
                    f"LHS-Axis: {chart.y_axis_labels} | "
                    f"RHS-Axis: {chart.rhs_y_axis_labels} | "
                    f"Values: {chart.constituents} {chart.aggregates}\n"
                )
        else:
            scout_summary += (
                "No specific charts detected, scan for embedded data tables "
                "or text logic."
            )

        # 3. Build the Final Cognitive Prompt
        final_prompt_context = (
            f"=== STEP 1: SPATIAL ANCHORS (OCR) ===\n"
            f"{ocr_context[:6000]}\n\n"  # Safety cap for tokens
            
            f"=== STEP 2: SCOUT DOSSIER (LAYOUT HINTS) ===\n"
            f"{scout_summary}\n\n"
            
            f"=== INSTRUCTION ===\n"
            f"1. ANALYZE: Use the Spatial Anchors to lock onto the Scout's regions.\n"
            f"2. AUDIT: Fill the `audit_log` FIRST. Explain your reasoning.\n"
            f"3. EXTRACT: Extract the 'Ground Truth' numbers (No mental math).\n"
            f"4. PERIODICITY: Detect the Page Default (e.g. 'Q'). If a specific column "
            f"(like 'YTD') differs, you MUST override it at the DataPoint level.\n"
            f"5. RESOLVE: If OCR and Visuals disagree, TRUST THE VISUALS "
            f"(Bar Height/Position)."
        )

        # 4. Build Messages
        messages = [
            SystemMessage(content=PROMPT_VISUAL_EXTRACTOR),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": final_prompt_context
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                ]
            )
        ]

        # 5. Invoke VLM with Structured Output
        response = await self.vlm.ainvoke(messages)
        
        return response

# Singleton Instance
visual_extractor = VisualExtractor()