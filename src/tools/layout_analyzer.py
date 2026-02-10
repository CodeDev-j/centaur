"""
Layout Analyzer Tool: The "Scout" (Structural Perception)
=========================================================

1. THE MISSION
--------------
To provide a high-level structural map of the page *before* forensic extraction begins.
It identifies Regions of Interest (RoIs) like charts, tables, and text blocks,
allowing the Visual Extractor to focus its attention.

2. THE MECHANISM
----------------
- **Input:** A single Base64 encoded image of the page.
- **Output:** A strict `PageLayout` object containing bounding boxes and classifications.
- **Resilience:** Implements strict payload masking to prevent LangSmith timeouts.
"""

import asyncio
import logging
import random
from functools import wraps
from typing import Dict, Any

# Third-party
from langchain_core.messages import HumanMessage, SystemMessage
# [FIX] Import the Context Manager to control the RunTree
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import traceable

# Internal
from src.config import SystemConfig
from src.prompts import PROMPT_LAYOUT_ANALYZER 
from src.schemas.layout_output import PageLayout

logger = logging.getLogger(__name__)

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
ANALYZER_TEMPERATURE = 0.0  # Deterministic output for coordinates
ANALYZER_MAX_TOKENS = 2000
ANALYZER_TIMEOUT = 180      # seconds
RETRY_COUNT = 3
RETRY_BACKOFF = 1           # seconds

# ==============================================================================
# 🛡️ RESILIENCE UTILS (Async Specific)
# ==============================================================================

# [FIX 1] Added Payload Sanitizer for Layout Analyzer
def mask_layout_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Surgical Masking for LangSmith.
    Prevents logging the raw 'img_b64' string to save bandwidth (avoiding timeouts).
    """
    if "img_b64" in inputs:
        clean_inputs = inputs.copy()
        val = inputs["img_b64"]
        if isinstance(val, str) and len(val) > 500:
            clean_inputs["img_b64"] = f"<MASKED_B64: {len(val)} chars>"
        return clean_inputs
    return inputs

def async_retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    """
    Exponential backoff decorator specific to ASYNC methods.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"❌ Layout Analysis Failed after {retries} retries: {e}")
                        # Return safe fallback. Do not crash the pipeline.
                        return PageLayout(
                            has_charts=False, 
                            confidence_score=0.0, 
                            charts=[]
                        )
                    
                    # Jittered backoff to prevent thundering herds
                    sleep_time = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                    logger.warning(
                        f"⚠️ Layout Error: {e}. Retrying in {sleep_time:.2f}s... "
                        f"(Attempt {x+1}/{retries})"
                    )
                    await asyncio.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator


# ==============================================================================
# 🧠 LAYOUT ANALYZER TOOL
# ==============================================================================
class LayoutAnalyzer:
    """
    Wraps the VLM to perform Object Detection on document pages.
    """
    
    def __init__(self):
        """
        Initializes the dedicated Layout Model.
        
        Architectural Note:
        We use `SystemConfig.get_llm` to ensure seamless switching between 
        Local OpenAI and Azure OpenAI based on the environment configuration.
        """
        base_llm = SystemConfig.get_llm(
            model_name=SystemConfig.LAYOUT_MODEL,
            temperature=ANALYZER_TEMPERATURE,
            max_tokens=ANALYZER_MAX_TOKENS,
            timeout=ANALYZER_TIMEOUT
        )
        
        # Bind Structured Output
        self.llm = base_llm.with_structured_output(PageLayout)
        
    @traceable(
        name="Layout Analyzer", 
        run_type="tool", 
        process_inputs=mask_layout_inputs
    )
    @async_retry_with_backoff(retries=RETRY_COUNT, backoff_in_seconds=RETRY_BACKOFF)
    async def scan(self, img_b64: str) -> PageLayout:
        """
        Scans the page for chart regions using the VLM.
        
        Args:
            img_b64 (str): Base64 encoded JPEG of the page.
            
        Returns:
            PageLayout: A Pydantic object containing a list of `ChartRegion` 
                        bounding boxes and metadata.
        """
        msg = [
            SystemMessage(content=PROMPT_LAYOUT_ANALYZER),
            HumanMessage(content=[
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                }
            ])
        ]
        
        # [CRITICAL FIX] Explicitly disable the RunTree Context.
        # This prevents LangChain from auto-creating a Child Span with the 10MB payload.
        # This supersedes config={"callbacks": []} which was found ineffective for auto-tracing.
        with tracing_v2_enabled(False):
            result = await self.llm.ainvoke(msg)
            
        return result


# Singleton Instance
layout_analyzer = LayoutAnalyzer()