import time
import random
import json
import re
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """
    Exponential backoff decorator for LLM/API calls.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"❌ Failed after {retries} retries: {e}")
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                    logger.warning(f"⚠️ Error: {e}. Retrying in {sleep:.2f}s...")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

def clean_and_parse_json(response_text: str):
    """
    Robustly extracts JSON from 'chatty' LLM responses.
    Handles Markdown blocks, preambles, and trailing commas.
    """
    if not response_text:
        return None

    # 1. Remove Markdown Code Blocks
    if "```" in response_text:
        pattern = r"```(?:json)?\s*(.*?)s*```"
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            response_text = match.group(1)
        else:
            parts = response_text.split("```")
            if len(parts) > 1:
                response_text = parts[1]

    # 2. Extract strictly from first '{' to last '}'
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}') + 1
    
    if start_idx != -1 and end_idx != 0:
        response_text = response_text[start_idx:end_idx]

    # 3. Parse with Emergency Repair
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Trailing comma fix (common in LLM JSON)
        response_text = re.sub(r',\s*}', '}', response_text)
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Repair Failed: {e}")
            return None