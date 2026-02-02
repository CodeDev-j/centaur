"""
Schema definitions for Vision Tool outputs (The "Read" Phase).

GOAL: This file defines the output of the Vision Layer (`vision.py`).
It serves as the "Cognitive Contract" that forces the VLM to structure 
unstructured visual data into a strict financial taxonomy.

ARCHITECTURAL NOTE:
- The "Brain" of the Pipeline: The field descriptions below function as 
  System Instructions, guiding the model's reasoning logic.
- Instruction Following: We use triple-quoted descriptions to act as 
  strict constraints (e.g., "The Agency Rule").
- Chain of Thought: The `audit_log` is placed FIRST to force reasoning 
  before extraction (CoT).
"""

from typing import List, Literal, Optional, TypeAlias
from pydantic import BaseModel, Field

# ==============================================================================
# 🧩 SHARED TYPE DEFINITIONS
# ==============================================================================

# Financial Reporting Periods
PeriodicityType: TypeAlias = Literal[
    "FY",     # Fiscal Year / Annual
    "Q",      # Quarterly
    "H",      # Half-Yearly (H1/H2)
    "9M",     # Nine Months (Common in Q3 reporting)
    "M",      # Monthly
    "W",      # Weekly
    "LTM",    # Last Twelve Months
    "YTD",    # Year to Date
    "Mixed",  # Explicit support for mixed timeframes (e.g. Q1, Q2, YTD)
    "Other",  # Catch-all for non-standard periods (e.g. "2-Month", "Holiday")
    "Unknown"
]

# ISO 4217 Currency Codes (Top 15 Liquid Currencies)
CurrencyType: TypeAlias = Literal[
    "USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "CHF", "INR",
    "HKD", "SGD", "NZD", "KRW", "SEK", "BRL", "None"
]

# Scalars (Pure Math Multipliers). STRICTLY EXCLUDES Units (%, bps).
MagnitudeType: TypeAlias = Literal[
    "k",    # Thousands (10^3)
    "M",    # Millions (10^6)
    "B",    # Billions (10^9)
    "T",    # Trillions (10^12)
    "None"  # Absolute value (10^0)
]

# The "MECE" (Mutually Exclusive) Category List
# Replaces fuzzy topics with strict "Nature of Fact" definitions.
CategoryType: TypeAlias = Literal[
    "Financial",   # The Ledger (Past Results)
    "Operational", # Business Metrics (Current State)
    "Strategic",   # The Plan (Future Actions/Claims)
    "External"     # The Environment (Macro/Regs)
]

# ==============================================================================
# 📊 VISION OUTPUT SCHEMAS
# ==============================================================================

class DataPoint(BaseModel):
    """
    Represents a single atomic number extracted from a chart or table.
    """
    label: str = Field(
        ...,
        description="""
        The X-axis label (e.g., '2024', 'Q1'). 
        CRITICAL: Scan the far right of the chart for Summary Columns.
        Labels like 'Avg', 'Total', 'CAGR', or date ranges (e.g., '08-18')
        are VALID and must be extracted as the final data point.
        """
    )

    numeric_value: float = Field(
        ...,
        description="""
        The base numeric value exactly as seen. 
        Example: If text is '$12.4B', extract 12.4. 
        Do NOT perform mental math (e.g., do not convert 12.4B to 12400000000).
        """
    )

    currency: CurrencyType = Field(
        default="None",
        description="""
        ISO 4217 Code.
        Infer from symbol context (e.g. '$' in Singapore -> SGD).
        If no currency symbol is present, use 'None'.
        """
    )

    magnitude: MagnitudeType = Field(
        default="None",
        description="""
        The Power-of-10 Multiplier. 
        Map 'mn/mm' -> M, 'bn' -> B.
        CRITICAL: Check the Chart Title, Subtitle, AND Axis Labels for Global Units.
        Example: If title says 'Revenue (£m)', you MUST set magnitude='M' for all points.
        STRICTLY EXCLUDES: %, bps, x (Put these in 'measure').
        """
    )

    measure: Optional[str] = Field(
        default=None, 
        description="""
        The Unit or Ratio.
        Examples: 'Percent', 'BasisPoints', 'Barrels', 'Users', 'x' (Multiple).
        CRITICAL EXCLUSIONS:
        1. Do NOT put Currency here (use 'currency' field).
        2. Do NOT put Multipliers here (e.g., 'Millions' goes in 'magnitude').
        3. Do NOT put Time here (e.g., 'Years' is usually implied by Periodicity).
        """
    )

    # [HIERARCHY PRIORITY 1: ATOMIC OVERRIDE]
    periodicity: Optional[PeriodicityType] = Field(
        default=None,
        description="""
        ATOMIC OVERRIDE: If the 'label' text explicitly contains 'YTD', 
        'FY', 'LTM', or '9M', you MUST set this field to match. 
        This overrides the Series and Page settings. 
        If label is standard (e.g. 'Q1'), leave as None.
        """
    )

    original_text: str = Field(
        default="",
        description="""
        The raw text seen on the chart for audit purposes (e.g. '$12.4B').
        Used for anti-hallucination verification.
        """
    )


class MetricSeries(BaseModel):
    """
    A collection of data points representing a specific metric (e.g. 'Revenue').
    """
    series_label: str = Field(
        ...,
        description="""
        The name of the series. For Chart Lines/Bars, use the Legend. 
        For Table/Memo rows below the chart, use the Row Header 
        exactly (e.g. 'Memo: JV Wholesales').
        """
    )
    
    definition: str = Field(
        default="",
        description="""
        Critical context from footnotes/legends.
        Check for footnote markers (e.g., '(1)', '*') attached to the series name.
        If found, scan the chart footer and extract the exact definition text.
        Example: 'Underlying EBITDA excluding one-offs'.
        """
    )

    # [HIERARCHY PRIORITY 2: SEMANTIC OVERRIDE]
    periodicity: Optional[PeriodicityType] = Field(
        default=None,
        description="""
        SEMANTIC OVERRIDE: Check the Chart Title or Header. 
        If it says 'LTM' or 'Monthly', set this field accordingly 
        to override the Page Default.
        """
    )

    # [UI LINKING]
    source_region_id: Optional[int] = Field(
        default=None,
        description="""
        The ID of the visual region (box) this series was extracted from. 
        Used for pixel-perfect highlighting on hover in the UI.
        """
    )

    data_points: List[DataPoint] = Field(default_factory=list)


class Insight(BaseModel):
    """
    Qualitative takeaways derived from the visual data.
    IMPORTANT: This class contains the 'Brain' of the classification logic.
    """
    
    category: CategoryType = Field(
        ...,
        description="""
        Classify using these STRICT PRIORITY RULES (The "Nature of Fact"):

        1. 'Financial' (THE LEDGER RULE): 
           Historical facts from the P&L, Balance Sheet, or Cash Flow.
           If it involves specific $ or % figures from the past, it is Financial.
           
        2. 'Operational' (THE ENGINE ROOM):
           Business metrics that drive the financials. 
           Examples: Headcount, Production Volumes, User Counts, Efficiency Ratios.

        3. 'Strategic' (THE AGENCY RULE):
           Forward-looking plans or decisions made by management.
           Examples: M&A, R&D Investment, Product Launches, Guidance/Outlook.
           
        4. 'External' (THE ENVIRONMENT):
           Factors outside the company's control.
           Examples: Macroeconomics, Competitor Moves, Regulation, FX Rates.
           
        *EXCLUDED*: 'Sentiment' (Do not capture subjective spin or fluff).
        """
    )

    topic: str = Field(
        ..., 
        description="""
        Subject (e.g., 'EBITDA Growth' or '2008 Event').
        """
    )

    content: str = Field(
        ...,
        description="""
        A detailed, information-dense extraction of the insight. 
        Do not summarize "high-level". Preserve specific nouns, dates, and causal links 
        to ensure high-quality retrieval in Vector Search.
        THE ANNOTATION RULE: Scan the Plot Area for 'Floating Text' or Callouts 
        (e.g. 'Global Financial Crisis'). These are CRITICAL insights.
        """
    )
    
    # [UI LINKING]
    source_region_id: Optional[int] = Field(
        default=None,
        description="""
        The ID of the visual region (box) this insight was derived from.
        """
    )


class PageLayout(BaseModel):
    """
    Root Container for the Forensic Analysis.
    """
    # [CHAIN OF THOUGHT: MUST BE FIRST]
    audit_log: str = Field(
        ...,
        description="""
        The 'Cognitive Scratchpad'. 
        BEFORE extracting metrics, you MUST explain the chart structure here:
        1. Identify the Unit of Measure and Magnitude (Global vs Local).
        2. Identify the Chart Type (Stacked, Waterfall, Dual-Axis).
        3. Explain how you mapped Colors to Legends.
        This forces reasoning before action.
        """
    )

    title: str = Field(
        default="Untitled",
        description="""
        The technical title of the analysis/chart.
        """
    )
    
    summary: str = Field(
        ...,
        description="""
        A detailed, information-dense summary of the visual data.
        Optimized for Semantic Retrieval (Embeddings). 
        Include specific trends, outliers, and entity names. 
        Avoid vague corporate speak.
        """
    )

    # [HIERARCHY PRIORITY 3: PAGE DEFAULT]
    periodicity: PeriodicityType = Field(
        default="Unknown",
        description="""
        The DOMINANT time basis of the page. Individual series can 
        override this. FY=Fiscal Year, Q=Quarterly, 9M=Nine Months, 
        LTM=Last 12 Months.
        """
    )

    metrics: List[MetricSeries] = Field(
        default_factory=list,
        description="""
        Extracted quantitative data.
        """
    )
    
    insights: List[Insight] = Field(
        default_factory=list,
        description="""
        Qualitative strategic takeaways.
        """
    )

    confidence_score: float = Field(
        default=0.0,
        description="""
        0.0 to 1.0 rating of extraction certainty based on visual clarity.
        """
    )