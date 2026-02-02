"""
Resilience & Sanitization Logic (The "Safety" Phase).

GOAL: This module ensures the pipeline survives real-world chaos. It handles 
two primary failure modes:
1. API Instability: Handles timeouts and rate limits via exponential backoff.
2. LLM Verbosity: Extracts valid JSON from "chatty" responses (e.g., when 
   models add markdown blocks or preambles like "Here is your JSON...").

ARCHITECTURAL NOTE:
- Decorator Pattern: 'retry_with_backoff' is designed to wrap any external 
  API call (Azure, Database, etc.) without polluting business logic.
- JSON Repair: 'clean_and_parse_json' implements a 3-stage heuristic to 
  surgically extract data from broken or formatted strings.
"""

import json
import logging
import random
import re
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

def retry_with_backoff(retries: int = 3, backoff_in_seconds: float = 1) -> Callable:
    """
    Exponential backoff decorator for LLM/API calls.
    
    Usage:
        @retry_with_backoff(retries=3, backoff_in_seconds=2)
        def call_openai(): ...
        
    Strategy:
        Delays are calculated as: (backoff * 2^attempt) + jitter
        Jitter (random.uniform) is crucial to prevent "thundering herd" 
        problems when multiple threads retry simultaneously.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries:
                        logger.error(f"❌ Failed after {retries} retries: {e}")
                        raise e
                    
                    sleep_time = (backoff_in_seconds * 2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"⚠️ Error: {e}. Retrying in {sleep_time:.2f}s... "
                        f"(Attempt {attempt + 1}/{retries})"
                    )
                    time.sleep(sleep_time)
                    attempt += 1
        return wrapper
    return decorator


def clean_and_parse_json(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extracts and sanitizes JSON from 'chatty' LLM responses.
    
    The Problem:
        LLMs often return: "Here is the data: ```json { ... } ```" 
        Standard `json.loads` fails on this.
        
    Strategy:
        1. Markdown Stripping: Regex removes code fences (```).
        2. Substring Extraction: Locates the first '{' and last '}'.
        3. Syntax Repair: Removes trailing commas (common LLM error).
    
    Returns:
        Dict: Parsed JSON object.
        None: If parsing fails after all recovery attempts.
    """
    if not response_text:
        return None

    cleaned_text = response_text.strip()

    # 1. Remove Markdown Code Blocks (```json ... ```)
    if "```" in cleaned_text:
        pattern = r"```(?:json)?\s*(.*?)s*```"
        match = re.search(pattern, cleaned_text, re.DOTALL)
        if match:
            cleaned_text = match.group(1)
        else:
            # Fallback: simple split if regex fails
            parts = cleaned_text.split("```")
            if len(parts) > 1:
                cleaned_text = parts[1]

    # 2. Extract strictly from first '{' to last '}'
    # This removes preambles ("Here is your file:") and postscripts.
    start_idx = cleaned_text.find('{')
    end_idx = cleaned_text.rfind('}') + 1
    
    if start_idx != -1 and end_idx != 0:
        cleaned_text = cleaned_text[start_idx:end_idx]

    # 3. Parse with Emergency Repair
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        # Common Fix: Remove trailing commas before closing braces
        # Pattern matches: , followed by whitespace, followed by } or ]
        cleaned_text = re.sub(r',\s*([}\]])', r'\1', cleaned_text)
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON Repair Failed. Raw text sample: {response_text[:50]}... Error: {e}"
            )
            return None