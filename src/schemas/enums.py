"""
src/schemas/enums.py
The Rosetta Stone: Shared Domain Definitions.
"""
from typing import Literal, TypeAlias

# ==============================================================================
# 1. TIME (Periodicity)
# ==============================================================================
PeriodicityType: TypeAlias = Literal[
    "FY",    # Fiscal Year
    "Q",     # Quarterly
    "H",     # Half-Yearly (H1/H2) - [Standardized from 'HY']
    "9M",    # Nine Months
    "M",     # Monthly
    "W",     # Weekly
    "LTM",   # Last Twelve Months
    "YTD",   # Year to Date
    "Mixed", # Multiple periods
    "Other", 
    "Unknown"
]

# ==============================================================================
# 2. MONEY (Currency & Magnitude)
# ==============================================================================
# ISO 4217 Liquid 15 + None
CurrencyType: TypeAlias = Literal[
    "USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "CHF", "INR",
    "HKD", "SGD", "NZD", "KRW", "SEK", "BRL", "None"
]

# Power of 10 Scalars only (No 'G' for Gigawatts)
MagnitudeType: TypeAlias = Literal[
    "k", "M", "B", "T", "None"
]

# ==============================================================================
# 3. SEMANTICS (Category & Sentiment)
# ==============================================================================
CategoryType: TypeAlias = Literal[
    "Financial", 
    "Operational", 
    "Strategic", 
    "External", 
    "Deal_Math"
]

SentimentType: TypeAlias = Literal[
    "Positive", "Negative", "Neutral", "Unknown"
]