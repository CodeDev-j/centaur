"""
Document-Level Metadata Extraction.

3-tier strategy:
  Tier 1 (Deterministic): page_count, currency, language, confidentiality
  Tier 2 (VLM-assisted):  company_name, document_type, sector, geography,
                           as_of_date, period_label, extraction_confidence

Runs after full document parsing. Costs ~$0.01 per document (one GPT-4.1-mini call).
"""

import logging
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import SystemConfig
from src.schemas.deal_stream import UnifiedDocument, HeaderItem, FinancialTableItem

logger = logging.getLogger(__name__)


# ── VLM Schema for document classification ────────────────────────────

_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {
            "type": ["string", "null"],
            "description": (
                "Primary company/entity name this document is about. "
                "NOT the advisor/bank name. null if multi-company or unclear."
            ),
        },
        "document_type": {
            "type": "string",
            "enum": [
                "earnings_slides", "annual_report", "investor_presentation",
                "10k", "10q", "8k", "proxy_statement",
                "cim", "lender_presentation", "financial_model", "term_sheet",
                "credit_agreement", "fairness_opinion", "teaser",
                "management_presentation", "quality_of_earnings",
                "industry_report", "sector_analysis", "comp_sheet",
                "market_update", "expert_transcript",
                "economic_report", "central_bank", "market_outlook", "policy_brief",
                "legal_opinion", "regulatory_filing", "internal_memo", "other",
            ],
        },
        "sector": {
            "type": ["string", "null"],
            "description": (
                "GICS sector: Technology, Healthcare, Financials, "
                "Consumer Discretionary, Consumer Staples, Industrials, "
                "Energy, Materials, Real Estate, Utilities, Communication Services. "
                "null if unclear or macro document."
            ),
        },
        "geography": {
            "type": ["string", "null"],
            "description": (
                "Primary market: 'US', 'Europe', 'Asia', 'Global', or specific country. "
                "null if unclear."
            ),
        },
        "as_of_date": {
            "type": ["string", "null"],
            "description": (
                "Financial period end date in YYYY-MM-DD format. "
                "For FY 2024 → '2024-12-31'. For Q3 2025 → '2025-09-30'. "
                "null if not determinable."
            ),
        },
        "period_label": {
            "type": ["string", "null"],
            "description": (
                "Human-readable period: 'FY 2024', 'Q3 2025', 'LTM Sep-24', "
                "'H1 2025'. null if not applicable."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "Your confidence in the overall classification (0.0-1.0).",
        },
    },
    "required": ["document_type", "confidence"],
}


async def extract_document_metadata(
    doc: UnifiedDocument,
    file_path: Path,
    page_count: int,
) -> Dict[str, Any]:
    """
    Extract document-level metadata using 3-tier strategy.
    Returns a flat dict ready for upsert_document_meta().
    """
    meta: Dict[str, Any] = {}

    # ── Tier 1: Deterministic (zero LLM cost) ─────────────────────────
    meta["page_count"] = page_count
    meta["currency"] = _extract_consensus_currency(doc)
    meta["language"] = "en"  # Default; future: OCR language detection
    meta["confidentiality"] = _detect_confidentiality(doc)

    # ── Tier 2: VLM-assisted classification ────────────────────────────
    vlm_result = await _classify_with_vlm(doc, file_path)
    if vlm_result:
        meta["company_name"] = vlm_result.get("company_name")
        meta["document_type"] = vlm_result.get("document_type", "other")
        meta["sector"] = vlm_result.get("sector")
        meta["geography"] = vlm_result.get("geography")
        meta["period_label"] = vlm_result.get("period_label")
        meta["extraction_confidence"] = vlm_result.get("confidence", 0.0)

        # Parse as_of_date from VLM string
        as_of_str = vlm_result.get("as_of_date")
        if as_of_str:
            try:
                meta["as_of_date"] = date.fromisoformat(as_of_str)
            except ValueError:
                meta["as_of_date"] = None
    else:
        meta["document_type"] = "other"
        meta["extraction_confidence"] = 0.0

    return meta


def _extract_consensus_currency(doc: UnifiedDocument) -> Optional[str]:
    """Majority vote across FinancialTableItem currencies."""
    currencies = []
    for item in doc.items:
        if isinstance(item, FinancialTableItem):
            c = getattr(item, "currency", None) or getattr(item.data, "currency", None)
            if c:
                currencies.append(c.upper())
    if not currencies:
        return None
    counter = Counter(currencies)
    return counter.most_common(1)[0][0]


def _detect_confidentiality(doc: UnifiedDocument) -> Optional[str]:
    """Scan first 3 pages' headers for confidentiality markers."""
    for item in doc.items:
        if isinstance(item, HeaderItem) and item.source.page_number <= 3:
            text = (item.text or "").upper()
            if "CONFIDENTIAL" in text:
                return "Confidential"
            if "PUBLIC" in text:
                return "Public"
    return None


async def _classify_with_vlm(
    doc: UnifiedDocument,
    file_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Send first 2 page images to GPT-4.1-mini for document classification.
    Returns parsed JSON dict or None on failure.
    """
    try:
        import fitz  # PyMuPDF
        import base64

        pdf = fitz.open(str(file_path))
        images = []
        for page_idx in range(min(2, len(pdf))):
            pix = pdf[page_idx].get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode()
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
            })
        pdf.close()

        # Build context from first 10 items (titles, headers, narratives)
        text_context = []
        for item in doc.items[:10]:
            if hasattr(item, "text") and item.text:
                text_context.append(
                    f"[{item.type}, p{item.source.page_number}] {item.text[:200]}"
                )

        llm = SystemConfig.get_llm(
            model_name="gpt-4.1-mini",
            temperature=0.0,
            max_tokens=500,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document classifier for a high-finance intelligence platform. "
                    "Classify the document shown in the images. "
                    "IMPORTANT: The company_name is the SUBJECT of the document, "
                    "NOT the investment bank or advisor that prepared it. "
                    "For example, if Goldman Sachs prepared a CIM about Acme Corp, "
                    "the company_name is 'Acme Corp', not 'Goldman Sachs'."
                ),
            },
            {
                "role": "user",
                "content": [
                    *images,
                    {
                        "type": "text",
                        "text": (
                            "Classify this document. Here are the first extracted items "
                            "for additional context:\n\n"
                            + "\n".join(text_context)
                        ),
                    },
                ],
            },
        ]

        structured_llm = llm.with_structured_output(_CLASSIFICATION_SCHEMA)
        result = await structured_llm.ainvoke(messages)

        if isinstance(result, dict):
            return result
        return result.model_dump() if hasattr(result, "model_dump") else None

    except Exception as e:
        logger.warning(f"VLM document classification failed (non-critical): {e}")
        return None
