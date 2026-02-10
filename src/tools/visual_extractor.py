"""
Visual Extractor Tool: The Forensic "Eye" (Perception Engine)
=============================================================

1. THE MISSION (Overall Approach)
---------------------------------
This module bridges the gap between raw pixels and structured financial data.
It orchestrates the VLM (Visual Language Model) to perform "Forensic Extraction"
governed by the strict schemas defined in `src.schemas.vision_output`.

The architectural flow is designed as a **"Cognitive Sandwich"**:
- **Context (The Foundation):** Injects precise Layout Hints ("Scout Dossier") 
  and full OCR text to ground the model in spatial reality.
- **Visuals (The Input):** Injects high-fidelity, contrast-enhanced images 
  optimized for reading dense financial grids.
- **Logic (The Constraint):** Enforces a strict "Audit First" protocol via 
  System Prompts to prevent hallucination.

2. THE MECHANISM (Implementation)
---------------------------------
To ensure robust extraction, the tool employs four key technical strategies:

A. **Pre-Processing Pipeline:** Uses CPU-bound threads to optimize image 
   contrast and sharpness, making faint gridlines and axis labels legible.

B. **Dynamic Prompt Construction:** Fuses "Spatial Anchors" (OCR) with 
   "Layout Hints" to guide the VLM's attention to specific bounding boxes.

C. **Resilience Layers:** Implements async retries with jittered backoff and 
   strict timeouts to prevent pipeline hangs during high-load.

D. **Observability:** Logs extraction confidence scores and metric counts, 
   providing immediate visibility into data quality via LangSmith.
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
from typing import Optional, Dict, Any

# Third-party
from langchain_core.messages import HumanMessage, SystemMessage
# [FIX] Import the Context Manager to control the RunTree
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import traceable
from PIL import Image, ImageEnhance

# Internal
from src.config import SystemConfig
from src.prompts import PROMPT_VISUAL_EXTRACTOR
from src.schemas.layout_output import PageLayout as LayoutHint 
from src.schemas.vision_output import VisionPageResult

logger = logging.getLogger(__name__)

# ==============================================================================
# ⚙️ CONFIGURATION (TUNABLES)
# ==============================================================================

# 1. MODEL PARAMETERS
VISION_TEMPERATURE = 0.0
VISION_MAX_TOKENS = 16000     # Bumped for GPT-4.1-mini (1M Context)
LLM_REQUEST_TIMEOUT = 300     # Seconds for the API call itself

# 2. IMAGE PRE-PROCESSING
CONTRAST_FACTOR = 1.4         # +40% contrast helps separate light gridlines
SHARPNESS_FACTOR = 1.5        # +50% sharpness improves small axis label OCR
MAX_IMAGE_DIM = 3000          # Pixels - balance between detail and token cost
JPEG_QUALITY = 85             # Compression quality (0-100)

# 3. RESILIENCE & CONCURRENCY
RETRY_COUNT = 3
RETRY_BACKOFF = 1             # Seconds
PAGE_EXTRACTION_TIMEOUT = 180 # Hard cap (seconds) for end-to-end page processing
WORKER_CAP = 4                # Max threads for image processing

# ==============================================================================
# ⚙️ RESOURCE MANAGEMENT
# ==============================================================================
# Determine worker count based on CPU cores, capped for stability
_MAX_WORKERS = min(WORKER_CAP, (os.cpu_count() or 1))
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
            retry_attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if retry_attempt == retries:
                        logger.error(
                            f"❌ Visual Extraction Failed after {retries} retries. "
                            f"Error: {e}"
                        )
                        return None  # Return None to allow graceful degradation

                    # Jittered backoff to prevent thundering herd
                    sleep_time = (
                        (backoff_in_seconds * 2 ** retry_attempt) + 
                        random.uniform(0, 1)
                    )
                    logger.warning(
                        f"⚠️ VLM Glitch: {e}. Retrying in {sleep_time:.2f}s..."
                    )
                    await asyncio.sleep(sleep_time)
                    retry_attempt += 1
        return wrapper
    return decorator

# [CRITICAL FIX] Universal Payload Sanitizer
def mask_heavy_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggressively masks ANY heavy payload in the inputs, regardless of key name.
    This prevents 'ConnectionError' in LangSmith when uploading massive traces.
    """
    clean_inputs = inputs.copy()
    for key, value in clean_inputs.items():
        # Mask Raw Bytes (The Image)
        if isinstance(value, bytes):
            clean_inputs[key] = f"<MASKED_BYTES: {len(value)} bytes>"
        # Mask Massive Strings (Base64) - Threshold: 2KB
        elif isinstance(value, str) and len(value) > 2000:
            if "ocr_context" in key:
                continue # Keep OCR text, it's valuable and usually <100KB
            clean_inputs[key] = f"<MASKED_STRING: {len(value)} chars>"
    return clean_inputs


# ==============================================================================
# 👁️ VISUAL EXTRACTOR TOOL
# ==============================================================================
class VisualExtractor:
    def __init__(self):
        # 1. Initialize the Base LLM via Factory
        # [VERIFIED] Supports gpt-4.1-mini (April 2025 release)
        base_llm = SystemConfig.get_llm(
            model_name=SystemConfig.VISION_MODEL,
            temperature=VISION_TEMPERATURE,
            max_tokens=VISION_MAX_TOKENS,
            # We enforce timeout at the tool level now, but keep this as a fallback
            timeout=LLM_REQUEST_TIMEOUT 
        )

        # 2. Bind the "Cognitive Contract" (Structured Output)
        # We force the model to adhere to the `VisionPageResult` schema.
        self.vlm = base_llm.with_structured_output(VisionPageResult)

        logger.info(
            f"👁️ Visual Extractor initialized. "
            f"Model: {SystemConfig.VISION_MODEL} | Mode: {SystemConfig.DEPLOYMENT_MODE}"
        )

    def _process_image_bytes(
        self,
        img_bytes: bytes,
        max_dim: int = MAX_IMAGE_DIM
    ) -> str:
        """
        CPU-bound processing: Resize, Enhance, and Encode to JPEG.
        Executed in a separate thread to avoid blocking the Async Event Loop.
        """
        try:
            # Context Manager ensures buffer is closed immediately after use
            with io.BytesIO(img_bytes) as input_buf, Image.open(input_buf) as img:
                # 1. Handle Transparency (PNG -> RGB)
                if img.mode in ('RGBA', 'LA') or \
                   (img.mode == 'P' and 'transparency' in img.info):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert('RGB')

                # 2. Boost Contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(CONTRAST_FACTOR)

                # 3. Boost Sharpness
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(SHARPNESS_FACTOR)
                # ---------------------------------

                # Resize Logic
                width, height = img.size
                if max(width, height) > max_dim:
                    ratio = max_dim / max(width, height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Encode to Base64
                with io.BytesIO() as output_buf:
                    img.save(output_buf, format="JPEG", quality=JPEG_QUALITY)
                    return base64.b64encode(output_buf.getvalue()).decode("utf-8")
                    
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise

    # [FIX] Enhanced input masking for Parent Trace
    @traceable(
        name="Visual Extraction", 
        run_type="tool",
        process_inputs=mask_heavy_inputs
    )
    @async_retry_with_backoff(
        retries=RETRY_COUNT,
        backoff_in_seconds=RETRY_BACKOFF
    )
    async def analyze_full_page(
        self,
        image_data: bytes,
        ocr_context: str,
        layout_hint: Optional[LayoutHint]  # The "Analyzer" Hint
    ) -> Optional[VisionPageResult]:       # The "Forensic" Result
        """
        FORENSIC ANALYSIS ENTRY POINT.
        Wraps the implementation with a strict timeout to prevent pipeline hangs.
        """
        try:
            return await asyncio.wait_for(
                self._analyze_full_page_impl(image_data, ocr_context, layout_hint),
                timeout=PAGE_EXTRACTION_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(
                f"⏱️ Page extraction timed out after {PAGE_EXTRACTION_TIMEOUT}s. "
                f"Skipping page to preserve pipeline health."
            )
            return None

    async def _analyze_full_page_impl(
        self,
        image_data: bytes,
        ocr_context: str,
        layout_hint: Optional[LayoutHint]
    ) -> Optional[VisionPageResult]:
        """
        Implementation logic:
        1. Context: Injects FULL OCR Grid + Structured Layout Hints.
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

        # 2. Construct the "Scout Dossier" (Structured Context)
        scout_summary = "=== STEP 2: DYNAMIC REGIONAL GUIDELINES ===\n"
        
        # Direct Pydantic Access (Cognitive Contract)
        # We trust that layout_hint matches the strict schema.
        charts = layout_hint.charts if layout_hint else []
        
        if charts:
            for i, chart in enumerate(charts):
                # Clean Enum Access
                vtype = (
                    chart.visual_type.value 
                    if hasattr(chart.visual_type, 'value') 
                    else str(chart.visual_type)
                )

                scout_summary += (
                    f"\nREGION_{i+1}:\n"
                    f"  BBox: {chart.bbox}\n"
                    f"  Type: {vtype}\n"
                    f"  Title: '{chart.title}'\n"
                )
                
                # Optional Enhancements (Check for existence or empty list)
                if chart.legend_keys:
                    scout_summary += f"  Legends: {', '.join(str(l) for l in chart.legend_keys[:5])}\n"
                
                # Assume standard chart attributes; gracefully handle if specific axes are missing
                if hasattr(chart, 'x_axis_labels') and chart.x_axis_labels:
                    scout_summary += f"  X-Axis: {', '.join(str(l) for l in chart.x_axis_labels[:5])}\n"
                
                if hasattr(chart, 'y_axis_labels') and chart.y_axis_labels:
                    scout_summary += f"  Y-Axis: {', '.join(str(l) for l in chart.y_axis_labels[:5])}\n"

        else:
            scout_summary += (
                "No specific charts detected via layout analysis. "
                "Scan full page for embedded data tables or text logic.\n"
            )

        # 3. Build the Final Cognitive Prompt
        # [OBSERVABILITY] Warn if OCR context is truly massive (>100k chars)
        if len(ocr_context) > 100_000:
            logger.warning(
                f"⚠️ Large OCR context: {len(ocr_context):,} chars. "
                f"Page may have extremely dense text or complex layout."
            )

        # [RESTORED] The "Cognitive Contract" Instructions
        final_prompt_context = (
            f"=== STEP 1: SPATIAL ANCHORS (FULL OCR) ===\n"
            f"{ocr_context}\n\n"  
            
            f"{scout_summary}\n"
            
            f"=== INSTRUCTION ===\n"
            f"1. PERCEPTION ONLY: Capture data exactly as seen. Do NOT normalize (e.g. keep 'Rev', don't write 'Revenue').\n"
            f"2. AUDIT FIRST: Fill the `audit_log` before extracting any metrics. Analyze Units, Legends, and Periodicity here.\n"
            f"3. PERIODICITY: Identify the Page Default. If a specific series/point differs (e.g. 'LTM'), override it at the local level.\n"
            f"4. MEMO ROWS: Look BELOW charts for table rows (e.g. 'Memo: CapEx'). Extract these as new Series.\n"
            f"5. CONFLICT RESOLUTION: If OCR text disagrees with visual bar height, TRUST THE VISUAL BAR."
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
        # [CRITICAL FIX] Explicitly disable the RunTree Context.
        # This prevents LangChain from auto-creating a Child Span with the 10MB payload.
        # This supersedes config={"callbacks": []} which was found ineffective for auto-tracing.
        with tracing_v2_enabled(False):
            response = await self.vlm.ainvoke(messages)
        
        # 6. Observability Logging
        if response:
            metric_count = len(response.metrics) if response.metrics else 0
            insight_count = len(response.insights) if response.insights else 0
            logger.info(
                f"✅ Visual Extraction Complete | "
                f"Confidence: {response.confidence_score:.2f} | "
                f"Metrics: {metric_count} series | "
                f"Insights: {insight_count} items"
            )
            
            if response.confidence_score < 0.5:
                logger.warning(
                    f"⚠️ Low confidence extraction ({response.confidence_score:.2f}). "
                    f"Page may have poor visual quality or complex layout."
                )
        else:
            logger.error("❌ Visual extraction returned None")

        return response

# Singleton Instance
visual_extractor = VisualExtractor()