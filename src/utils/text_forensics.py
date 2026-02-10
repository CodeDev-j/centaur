"""
src/utils/text_forensics.py

Forensic Text Analysis Utilities.
Responsible for deducing structured financial context (Accounting Basis,
Periodicity, Currency) from unstructured text headers and footnotes.
"""

import re
from typing import Dict, Optional


def extract_table_metadata(
    headers: str, footnotes: str = ""
) -> Dict[str, Optional[str]]:
    """
    [FORENSIC LOGIC] Regex-based detection of financial metadata.

    Strategy: "Compound Vocabulary"
    We detect the interaction of terms (e.g., "Pro Forma" + "Adjusted") to
    classify the nature of the data.

    Args:
        headers (str): Concatenated column headers from the table.
        footnotes (str): Concatenated footnotes/captions relevant to the table.

    Returns:
        Dict with keys: 'accounting_basis', 'periodicity', 'currency'.
    """
    meta = {
        "accounting_basis": "Unknown",
        "periodicity": None,
        "currency": None
    }

    full_text = (headers + " " + footnotes).lower()

    # ==========================================================================
    # 1. Accounting Basis Logic
    # ==========================================================================
    # We use \b word boundaries to prevent partial matches (e.g. "asset" -> "as")

    has_pf = bool(re.search(r'\b(pro ?forma|pf|projected)\b', full_text))
    has_adj = bool(re.search(r'\b(adjusted|adj\.?|non-gaap|underlying|ebitda)\b', full_text))
    is_gaap = bool(re.search(r'\b(gaap|ifrs|ias|frs|statutory|audited|reported)\b', full_text))
    is_run_rate = bool(re.search(r'\b(run[- ]?rate)\b', full_text))

    if is_run_rate:
        meta["accounting_basis"] = "Run-Rate"
    elif has_pf and has_adj:
        meta["accounting_basis"] = "Pro Forma Adjusted"
    elif has_pf:
        meta["accounting_basis"] = "Pro Forma"
    elif has_adj:
        meta["accounting_basis"] = "Adjusted"
    elif is_gaap:
        # Distinguish IFRS vs GAAP if possible, otherwise generic
        if "ifrs" in full_text or "ias" in full_text:
            meta["accounting_basis"] = "IFRS"
        else:
            meta["accounting_basis"] = "GAAP/Statutory"

    # ==========================================================================
    # 2. Periodicity Detection
    # ==========================================================================
    if re.search(r'\b(ltm|last 12 months)\b', full_text):
        meta["periodicity"] = "LTM"
    elif re.search(r'\b(ytd|year to date)\b', full_text):
        meta["periodicity"] = "YTD"
    elif re.search(r'\b(hy|h1|h2|half[- ]?year)\b', full_text):
        meta["periodicity"] = "H"
    elif re.search(r'\b(quarter|q[1-4])\b', full_text):
        meta["periodicity"] = "Q"
    elif re.search(r'\b(fy|fiscal year)\b', full_text):
        meta["periodicity"] = "FY"

    # ==========================================================================
    # 3. Currency Inference
    # ==========================================================================
    if re.search(r'(\$|USD)', full_text, re.IGNORECASE):
        meta["currency"] = "USD"
    elif re.search(r'(€|EUR)', full_text, re.IGNORECASE):
        meta["currency"] = "EUR"
    elif re.search(r'(£|GBP)', full_text, re.IGNORECASE):
        meta["currency"] = "GBP"

    return meta