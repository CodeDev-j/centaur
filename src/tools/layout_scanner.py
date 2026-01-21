import asyncio
import base64
import logging
import random
from functools import wraps

# Third-party
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

# Internal
from src.config import SystemConfig
from src.prompts import PROMPT_LAYOUT_SCANNER
from src.schemas.layout import PageLayout

logger = logging.getLogger(__name__)

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
SCANNER_TEMPERATURE = 0.0
SCANNER_MAX_TOKENS = 2000
SCANNER_TIMEOUT = 180  # seconds
RETRY_COUNT = 3
RETRY_BACKOFF = 1  # seconds

# ==============================================================================
# 🛡️ RESILIENCE UTILS
# ==============================================================================
def async_retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    """
    Exponential backoff decorator specific to async methods.
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
                        logger.error(f"Layout Scan Failed after {retries} retries: {e}")
                        # Return safe fallback on total failure
                        return PageLayout(
                            has_charts=False, 
                            confidence_score=0.0, 
                            charts=[]
                        )
                    
                    sleep_time = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                    logger.warning(f"Layout Error: {e}. Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator

# ==============================================================================
# 🧠 LAYOUT SCANNER TOOL
# ==============================================================================
class LayoutScanner:
    def __init__(self):
        # Using the Layout Model (cheaper/faster) or Vision Model (smarter) as per config
        self.llm = ChatOpenAI(
            model=SystemConfig.LAYOUT_MODEL, 
            temperature=SCANNER_TEMPERATURE,
            max_tokens=SCANNER_MAX_TOKENS, 
            request_timeout=SCANNER_TIMEOUT
        ).with_structured_output(PageLayout)
        
    @traceable(name="Layout Scout", run_type="tool")
    @async_retry_with_backoff(retries=RETRY_COUNT, backoff_in_seconds=RETRY_BACKOFF)
    async def scan(self, img_b64: str) -> PageLayout:
        """
        Scans the page for chart regions using the VLM.
        Single Pass: Detects regions AND extracts high-level content hints.
        """
        msg = [
            SystemMessage(content=PROMPT_LAYOUT_SCANNER),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ])
        ]
        
        # Retry logic is handled by the decorator; direct invocation here
        result = await self.llm.ainvoke(msg)
        return result

layout_scanner = LayoutScanner()