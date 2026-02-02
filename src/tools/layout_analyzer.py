"""
Layout Analyzer Tool (The "Glance" Phase).

GOAL: This module acts as the "Air Traffic Controller" for the pipeline. 
It scans a rendered page image to identify regions of interest (Charts, 
Tables, Infographics) and returns their coordinates on a 0-1000 grid.

ARCHITECTURAL NOTE:
- Routing Logic: If this tool detects NO charts, the pipeline skips the 
  Visual Extraction phase, saving cost and time.
- Structural Enforcement: Uses OpenAI's `with_structured_output` to force 
  the VLM to return a valid `PageLayout` Pydantic object, ensuring downstream 
  type safety.
- Fail-Safe: Includes a custom async retry decorator that returns a "Safe Empty" 
  layout on total failure, allowing the text-only pipeline to proceed.
"""

import asyncio
import logging
import random
from functools import wraps

# Third-party
from langchain_core.messages import HumanMessage, SystemMessage
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
def async_retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    """
    Exponential backoff decorator specific to ASYNC methods.
    
    Distinct from `src.utils.resilience` because it handles Awaitables and 
    provides a domain-specific fallback (Empty PageLayout) to prevent 
    pipeline crashes on single-page failures.
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
                        # CRITICAL: Return safe fallback. Do not crash the pipeline.
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
        
    @traceable(name="Layout Analyzer", run_type="tool")
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
        
        # Invoke the VLM with Structured Output enforcement.
        result = await self.llm.ainvoke(msg)
        return result


# Singleton Instance
layout_analyzer = LayoutAnalyzer()