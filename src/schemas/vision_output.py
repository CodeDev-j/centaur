"""
Vision Output Schema: The "Cognitive Contract" (Perception Layer)
================================================================

1. THE MISSION (Overall Approach)
---------------------------------
This file defines the strict Pydantic taxonomy for the Visual Extractor tool. 
It acts as the critical API boundary between the VLM’s unstructured perception 
and the pipeline’s structured data stream.

The core philosophy is **"Separation of Concerns"**:
- **The VLM is the Eye (Perception):** Its sole job is to "see" and "capture" 
  raw visual data exactly as it appears on the page.
- **Python is the Brain (Logic):** Complex transformations, normalization 
  (e.g., mapping "Rev" -> "Revenue"), and calculations are explicitly FORBIDDEN 
  at this layer. These are deferred to deterministic Python services downstream 
  to minimize hallucination risk and maximize auditability.

2. THE MECHANISM (Implementation)
---------------------------------
To enforce this philosophy, the schema employs four key technical strategies:

A. **System Instructions as Code:** The `Field` descriptions function as mini-prompts, guiding the model's 
   attention to specific visual elements (Legends, Axes, Footnotes).

B. **Forced Chain of Thought (CoT):** The `audit_log` field is intentionally placed first. This forces the model 
   to verbally analyze the chart structure (Units, Colors, Anomalies) *before* it is allowed to commit to extracting a single number.

C. **Resilience via Validators:** We use `BeforeValidator` hooks to implement "Soft Failures." If the VLM 
   see "N/A" or "—", the validator gracefully converts this to `None` 
   instead of crashing the entire pipeline.

D. **Structural Hierarchy:** The schema enforces a strict hierarchy (Page -> Series -> DataPoint) 
   that mirrors the physical layout of financial documents.
"""

from typing import Annotated, List, Optional, Any
from pydantic import BaseModel, Field, BeforeValidator, model_validator, ConfigDict
from src.schemas.enums import (
    PeriodicityType,
    CurrencyType,
    MagnitudeType,
    CategoryType,
    SeriesNatureType,
    StatedDirectionType,
)

# ==============================================================================
# 🛡️ RESILIENCE VALIDATORS (Soft Failures)
# ==============================================================================

def sanitize_financial_float(v: Any) -> Optional[float]:
    """
    The 'Soft Fail' Validator.
    Converts messy financial strings into floats or None. Never crashes.
    
    Inputs: "N/A", "NM", "-", "($50.2)", "1,200", "15x"
    Outputs: None, None, None, -50.2, 1200.0, 15.0
    """
    if v is None:
        return None
    if isinstance(v, (float, int)):
        return float(v)
    if isinstance(v, str):
        clean = v.strip().lower()
        # Handle Private Equity slang for Null
        if clean in ["n/a", "nm", "-", "none", "—", "null", "not meaningful", ""]:
            return None
        
        # Remove common artifacts
        clean = clean.replace("$", "").replace(",", "").replace("x", "").replace("%", "")
        
        # Handle Accounting Negatives: ($50) -> -50
        if "(" in clean and ")" in clean:
            clean = "-" + clean.replace("(", "").replace(")", "")
            
        try:
            return float(clean)
        except ValueError:
            return None # Soft Fail to Null, don't crash pipeline
    return None

def normalize_currency(v: Any) -> str:
    """Soft Match for Currency Enum."""
    if isinstance(v, str):
        clean = v.strip().upper()
        # Handle "NONE" explicitly to match Literal["None"]
        if clean in ["NONE", "NULL", "N/A", ""]:
            return "None"
            
        # Common symbol mapping
        if clean == "$":
            return "USD"
        if clean == "€":
            return "EUR"
        if clean == "£":
            return "GBP"
        return clean
    return "None"

# ==============================================================================
# 📊 VISION OUTPUT SCHEMAS
# ==============================================================================

class DataPoint(BaseModel):
    """
    Represents a single atomic number extracted from a chart or table.
    Enriched with Anti-Hallucination hooks.
    """
    # Soft Handshake: Ignore extra noise from LLM
    model_config = ConfigDict(extra='ignore')

    # 1. Verbatim Extraction (The Truth)
    label: str = Field(
        ...,
        description="""
        The X-axis label (e.g., '2024', 'Q1'). 
        CRITICAL: Scan the far right of the chart for Summary Columns.
        Labels like 'Avg', 'Total', 'CAGR', or date ranges (e.g., '08-18')
        are VALID and must be extracted as the final data point.
        """
    )
    
    # 2. Soft-Fail Value (The "Safe" Number)
    numeric_value: Annotated[Optional[float], BeforeValidator(sanitize_financial_float)] = Field(
        ...,
        description="""
        The base numeric value exactly as seen. 
        Example: If text is '$12.4B', extract 12.4. 
        Do NOT perform mental math (e.g., do not convert 12.4B to 12400000000).
        Null if 'N/A' or illegible.
        """
    )

    # 3. Context & Units (The "Deal Math")
    # Applied BeforeValidator for Enum Soft Matching (e.g. '$' -> 'USD')
    currency: Annotated[CurrencyType, BeforeValidator(normalize_currency)] = Field(
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

    # 4. Hierarchy Priority 1: Atomic Override
    periodicity: Optional[PeriodicityType] = Field(
        default=None,
        description="""
        ATOMIC OVERRIDE: If the 'label' text explicitly contains 'YTD', 
        'FY', 'LTM', or '9M', you MUST set this field to match. 
        This overrides the Series and Page settings. 
        """
    )

    source_scenario: Optional[str] = Field(
        None, 
        description="""
        Context qualifier: 'Base Case', 'Downside', 'Management Case'.
        """
    )

    # 5. Anti-Hallucination (Evidence Anchoring)
    original_text: str = Field(
        default="",
        description="""
        The raw text exactly as seen on the chart for audit purposes (e.g. '$12.4B').
        Used for anti-hallucination verification.
        """
    )
    
    @model_validator(mode='after')
    def enforce_periodicity_fallback(self) -> 'DataPoint':
        """
        Auto-detect periodicity from label if not explicitly set.
        Implements Priority 1 of the hierarchy (Deterministic Override).
        """
        if self.periodicity is None:
            label_lower = self.label.lower()
            if 'ltm' in label_lower:
                self.periodicity = "LTM"
            elif 'ytd' in label_lower:
                self.periodicity = "YTD"
            elif 'fy' in label_lower or 'annual' in label_lower:
                self.periodicity = "FY"
            elif any(q in label_lower for q in ['q1', 'q2', 'q3', 'q4']):
                self.periodicity = "Q"
            elif '9m' in label_lower:
                self.periodicity = "9M"
            elif any(h in label_lower for h in ['h1', 'h2', 'half']):
                self.periodicity = "H"
        return self


class MetricSeries(BaseModel):
    """
    A collection of data points representing a specific metric (e.g. 'Revenue').
    """
    # Soft Handshake: Ignore extra noise from LLM
    model_config = ConfigDict(extra='ignore')

    series_label: str = Field(
        ...,
        description="""
        The name of the series exactly as it appears.
        
        SOURCES (in priority order):
        1. Legend/Key: For standard chart lines/bars, use the legend text.
        2. Table Rows: For rows BELOW the chart (memo items), use the row header
           exactly as written (e.g. 'Memo: JV Wholesales', 'Unit Volumes').
        3. Y-Axis Label: If no legend exists, use the Y-axis label.
        
        INSTRUCTION: Capture the text VERBATIM. 
        If it says 'Wholesales*', write 'Wholesales*'. Do not strip the marker.
        """
    )
    
    definition: str = Field(
        default="",
        description="""
        Context or definitions found in footnotes/notes.
        
        SEARCH PROTOCOL (Check BOTH):
        1. **Visual Link:** Look at the 'series_label' you just extracted. Does it have a marker (*, 1, †)? 
           If yes, scan for the matching footnote.
        2. **Semantic Link:** Scan all footnotes/notes text on the page, especially the bottom parts.   
           Does a note explicitly name this metric? 
           (e.g. Label='EBITDA', Note='Adjusted EBITDA excludes...')
           If yes, extract that text even if there is no marker on the label.
        """
    )
    
    accounting_basis: Optional[str] = Field(
        default=None,
        description="""
        Financial methodology qualifier for THIS series specifically.
        Search TWO sources in strict order:

        1. THE LABEL (Primary): Read the series_label for qualifying terms
           (e.g., 'EBIT Adjusted' -> 'Adjusted', 'PF LTM EBITDA' -> 'Pro Forma').

        2. FOOTNOTES (Secondary — specificity-gated): A footnote may only inform
           this field if it is specifically tied to THIS series, via either:
           - A matching footnote marker on the label (e.g., '*', '†', '1'), OR
           - The footnote explicitly names this series (e.g., 'PF Adj. EBITDA
             includes pro forma impact of ABC acquisition' -> confirms
             basis='Pro Forma Adjusted' for the 'PF Adj. EBITDA' series).
           FORBIDDEN: Do NOT apply a footnote to this series if it names a
           DIFFERENT series on the same chart.

        If qualifying terms are found in EITHER source, you MUST populate this
        field. Leave null ONLY if no qualifying term is traceable to THIS series.

        Common terms and their canonical forms:
        - 'Adjusted', 'Adj.', 'Non-GAAP', 'Underlying' -> 'Adjusted'
        - 'Pro Forma', 'PF' -> 'Pro Forma'
        - 'Run-Rate', 'RR', 'Annualized' -> 'Run-Rate'
        - 'GAAP', 'IFRS', 'Reported' -> 'GAAP'
        - Compound (e.g., 'PF Adjusted', 'Adj. Run-Rate') -> use the compound form.

        SCOPE BOUNDARY: This field is for financial statement methodology ONLY.
        Do NOT use it for market research or forecast data — use 'data_provenance'.
        """
    )

    # [MARKET & RESEARCH DATA PROVENANCE]
    data_provenance: Optional[str] = Field(
        default=None,
        description="""
        Captures the estimation methodology or source type for NON-financial-statement
        data (e.g., market research, CDD data, third-party benchmarks).
        Leave null for audited company financial statements.

        Populate from TWO dimensions — combine if both are present:

        1. DATA ORIGIN (who produced it):
           - 'Management Estimate' — company-provided, unaudited
           - 'Third-Party Research' — from a named research firm (Gartner, Bain, IDC, etc.)
           - 'Consensus' — aggregated analyst estimates

        2. TEMPORAL NATURE (historical vs forward-looking):
           - 'Historical' — reported actuals (e.g., '2022A', '2023A')
           - 'Estimated' — approximate or survey-derived (e.g., market size '~$50B')
           - 'Forecast' — projected future values (e.g., '2025E', '2027F', 'CAGR')

        DETECTION SIGNALS (check in order):
        1. Axis label suffixes: 'A' or 'Actual' -> 'Historical'; 'E', 'F', 'P',
           or 'Est.' suffix -> 'Forecast'.
        2. Chart title or subtitle: 'Market Forecast', 'Bottom-Up TAM' -> 'Forecast'.
        3. Source footnote: 'Source: Gartner' -> 'Third-Party Research'.
        4. Mixed series (some historical, some forecast): use 'Historical and Forecast'.

        Examples:
        - TAM chart sourced from Bain with 2025E labels -> 'Third-Party Research; Forecast'
        - Management-prepared revenue bridge with actuals -> leave null (financial statement)
        - Consensus EBITDA estimates chart -> 'Consensus; Forecast'
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

    series_nature: SeriesNatureType = Field(
        default="level",
        description="""
        The semantic nature of this series' values.
        - 'level' : An absolute stock value at a point in time.
                    Examples: EBIT = €14,224M; Revenue = $500M.
                    Waterfall: bars rooted to the X-axis ("Stock" bars).
        - 'delta' : A signed period-over-period change or bridge driver.
                    Examples: Δ FX = +€41M; YoY revenue growth = -3%.
                    Waterfall: floating bars ("Flow" bars).
                    Also applies to variance charts, budget-vs-actual bars,
                    and any series whose values represent changes rather than levels.
        CRITICAL: Always set 'delta' when values represent changes. Downstream
        systems use this to avoid misreading a signed change (e.g. Δ Selling
        expenses = +€41M) as an absolute figure (Selling expenses = €41M).
        """
    )

    data_points: List[DataPoint] = Field(default_factory=list)


class Insight(BaseModel):
    """
    Qualitative takeaways derived from the visual data.
    IMPORTANT: This class contains the 'Brain' of the classification logic.
    """
    # Soft Handshake: Ignore extra noise from LLM
    model_config = ConfigDict(extra='ignore')
    
    category: CategoryType = Field(
        ...,
        description="""
        Classify by the TYPE of assertion, NOT the topic. The same topic
        (FX, Capex, Legal proceedings) can belong to different categories.

        1. 'Financial' — "What did the ledger record?"
           Quantitative facts from the P&L, Balance Sheet, or Cash Flow — historical
           or projected financial statement line items.
           BRIDGE RULE: Any insight explaining WHY a metric changed (variance/waterfall
           drivers) is Financial regardless of the underlying topic.
           Examples: "EBIT fell due to volume/pricing." | "Legal add-back +€94M in 2023."

        2. 'Operational' — "How did the business run?"
           Non-ledger business metrics describing operating performance.
           Examples: Headcount, Production Volumes, Utilisation Rates, Customer Counts.

        3. 'Market' — "What does the industry or environment look like?"
           Market structure, TAM/SAM/SOM, competitive dynamics, sector trends, and
           macro commentary. Distinct from the company's own reported performance.
           Examples: Market size, CAGR, competitive positioning, regulatory context.

        4. 'Strategic' — "What decision or corporate event occurred or is planned?"
           Any assertion about a decision or corporate event by those with agency —
           management OR shareholders/owners — regardless of whether it is past,
           present, or future. This preserves historical strategic context across
           a multi-year deal lifecycle.
           - Management: M&A (completed or planned), product launches, R&D direction,
             restructuring decisions, Capex programmes.
           - Shareholders/Owners: PE sponsor directives, exit events or planning,
             dividend policy, ownership restructuring, co-investor or JV strategy.
           DISTINCTION: The restructuring charge on the P&L is Financial. The
           decision to restructure (past or future) is Strategic. Historical M&A
           context ("the company acquired XYZ in 2021") is Strategic; the resulting
           revenue contribution is Financial.

        5. 'Transactional' — "How was the deal structured and financed?"
           Anything describing the transaction itself rather than the business:
           purchase price, entry/exit multiples, synergies, debt tranche terms
           (pricing, maturity, amortisation), leverage metrics at close, covenant
           thresholds, and returns analysis (IRR, MoM, yield-to-maturity).
           EDGE CASE: Actual leverage ratio (4.2x Net Debt/EBITDA) = Financial.
           The covenant it is tested against (e.g., 5.5x maximum) = Transactional.

        *EXCLUDED*: Do not capture subjective spin, qualitative fluff, or
        assertions that cannot be traced to a specific observable fact.
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
    
    # [HYBRID DATA UPGRADE]
    supporting_metrics: List[DataPoint] = Field(
        default_factory=list,
        description="""
        ONLY populate if the insight text ITSELF contains an explicit number
        (e.g., text says 'driven by +15%' → extract 15.0).
        FORBIDDEN: Do NOT borrow values from the metrics/chart bars to 
        'prove' a qualitative statement. If uncertain, leave empty [].
        """
    )
    
    stated_direction: Optional[StatedDirectionType] = Field(
        default=None,
        description="""
        Populate ONLY when the chart explicitly marks this annotation with a
        directional symbol (e.g. a '+' bullet, '–' bullet, up/down arrow).
        Do NOT infer from content — if no symbol is present, leave null.
        Example: Mercedes-Benz bridge uses blue '+' and red '–' bullets;
        capture those signals here.
        """
    )
    
    # [UI LINKING]
    source_region_id: Optional[int] = Field(
        default=None,
        description="""
        The ID of the visual region (box) this insight was derived from.
        """
    )
    
    @model_validator(mode='after')
    def validate_supporting_metrics(self) -> 'Insight':
        """Filter out supporting metrics with no numeric value."""
        if self.supporting_metrics:
            self.supporting_metrics = [
                dp for dp in self.supporting_metrics 
                if dp.numeric_value is not None
            ]
        return self


class VisionPageResult(BaseModel):
    """
    Root Container for the Forensic Analysis.
    """ 
    # 1. Soft Handshake: extra='ignore' to handle parser noise
    # 2. Schema Drift: populate_by_name=True for OpenAI compatibility
    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True
    )

    # [CHAIN OF THOUGHT: MUST BE FIRST]
    audit_log: str = Field(
        ...,
        description="""
        The 'Cognitive Scratchpad'. 
        BEFORE extracting metrics, you MUST analyze the chart structure in this exact order:
        
        1. UNITS: Identify the Unit of Measure and Magnitude (Global vs Local).
        2. PERIODICITY: Identify the time frequency (FY, LTM, Quarterly).
        3. CHART TYPE: Identify the structure (Stacked, Waterfall, Pie, Valuation).
        4. LEGENDS: Explain how you mapped Colors to Legends.
        5. CONTEXT: Note any Financial Context flags (Adjusted, Pro Forma).
           If Adjustment bars are present, explicitly state their neighbours:
           "Left=Reported, Right=Adjusted → positive bar is add-back" OR
           "Left=Adjusted, Right=Reported → positive bar is deduction."
        6. ANOMALIES: Flag missing data, discontinuities, or annotations.
        7. MATH CHECK (CRITICAL):
           - Perform a summation/logic check on the visible numbers.
           - Waterfall: Start + Sum(Deltas) ≈ End.
           - Stacked / Pie: Sum(Parts) ≈ Total (or 100%).
           - Valuation Field: Low Value < Mean/Median < High Value.
           - Table: Check Row/Column Totals if explicit.
           - If the math fails, RE-READ the labels to find the error.
        8. CONFIDENCE & ANNOTATIONS:
           - Assign confidence_score (0.0-1.0) to the root `confidence_score` field.
             Base it on visual clarity, OCR legibility, and math check outcome.
             High confidence = clear labels + math checks pass. Do NOT leave at default 0.
           - List any floating text callouts or qualitative annotations you see
             (e.g., "Operational efficiencies outweigh one-time payments"). Each
             becomes a separate entry in `insights`. If none exist, state "No annotations."
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