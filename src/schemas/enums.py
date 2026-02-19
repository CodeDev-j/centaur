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
    "Financial",      # Historical ledger facts (P&L, BS, CF, bridge components)
    "Operational",    # Historical non-ledger KPIs (headcount, production, utilisation)
    "Market",         # Industry structure, TAM, competitive dynamics, macro commentary
    "Strategic",      # Decisions/corporate events by management OR shareholders/owners (past or future)
    "Transactional",  # Deal terms, financing, capital structure, covenants, returns
]

SentimentType: TypeAlias = Literal[
    "Positive", "Negative", "Neutral", "Unknown"
]

# ==============================================================================
# 4. CHART SEMANTICS
# ==============================================================================
SeriesNatureType: TypeAlias = Literal[
    "level",  # Absolute stock value (e.g. EBIT = €14,224M; bar rooted to x-axis)
    "delta",  # Period-over-period change or bridge driver (e.g. Δ FX = +€41M; floating bar)
]

StatedDirectionType: TypeAlias = Literal[
    "positive_contributor",  # Chart explicitly marks with '+', up-arrow, or equivalent
    "negative_contributor",  # Chart explicitly marks with '–', down-arrow, or equivalent
]