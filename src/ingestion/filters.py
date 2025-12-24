import re
import logging
from typing import List

logger = logging.getLogger(__name__)

class ContentFilter:
    """
    The 'Garbage Filter'.
    Sanitizes text to remove non-semantic noise (headers, footers, artifacts).
    """

    # Regex patterns for common document noise
    PATTERNS = [
        r"Page \d+ of \d+",       # "Page 1 of 50"
        r"^\d+$",                 # Standalone page numbers
        r"(?i)confidential",      # "CONFIDENTIAL" watermarks
        r"(?i)draft",             # "DRAFT" watermarks
        r"_{5,}",                 # Long underscores (signatures/forms)
        r"\s{2,}",                # Excessive whitespace
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p) for p in self.PATTERNS]

    def clean(self, text: str) -> str:
        """
        Applies all filters to the input text.
        """
        if not text:
            return ""

        cleaned_text = text

        # 1. Regex Removal
        for pattern in self.compiled_patterns:
            cleaned_text = pattern.sub(" ", cleaned_text)

        # 2. Whitespace Normalization
        # Collapse multiple spaces/newlines into single space
        cleaned_text = " ".join(cleaned_text.split())

        return cleaned_text.strip()

    def is_valid_chunk(self, text: str, min_chars: int = 50) -> bool:
        """
        Decides if a chunk has enough semantic value to be worth embedding.
        """
        if not text:
            return False
            
        # Filter out chunks that are too short (usually OCR noise)
        if len(text) < min_chars:
            return False
            
        return True