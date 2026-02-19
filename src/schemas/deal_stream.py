"""
src/schemas/deal_stream.py

THE UNIFIED DEAL THEORY (The Stream Layer)
==========================================
This schema merges high-performance engineering (Pydantic V2 Streams) with 
high-value Deal Logic. It treats a document as a linear stream of 
Typed Events, but preserves the Logical Graph via cluster IDs.

ARCHITECTURE:
- Connectivity: Uses 'layout_cluster_id' to prevent Context Shredding.
- Composition: Introduces 'ChartTableItem' for hybrid artifacts (Football Fields).
- Fidelity: 'FinancialTableItem' now supports row hierarchy (Indentation).
- Simplicity: Stripped raw BBox geometry from the stream to save tokens.
"""

from typing import Annotated, List, Literal, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from src.schemas.layout_output import VisualArchetype
from src.schemas.enums import (
    PeriodicityType,
    CurrencyType,
    SentimentType,
    CategoryType
)

from src.schemas.vision_output import MetricSeries

# ==============================================================================
# 1. INFRASTRUCTURE PRIMITIVES
# ==============================================================================

class SourceRef(BaseModel):
    """
    Lightweight lineage. Points strictly to the origin location.
    Normalized bbox (0-1) carried for citation overlays in the viewer.
    """
    # Soft Handshake: Ignore extra noise to prevent blocking errors
    model_config = ConfigDict(extra='ignore')

    file_hash: str = Field(
        ...,
        description="""
        SHA-256 hash of the original source file.
        """
    )
    page_number: int = Field(
        ...,
        description="""
        1-based page number where this item was found.
        """
    )
    layout_id: Optional[str] = Field(
        None,
        description="""
        Link to the layout region ID (ChartRegion.region_id).
        """
    )

    # Normalized bounding box (0.0-1.0) for citation overlays.
    # Converted from ChartRegion's 0-1000 grid at creation time.
    bbox_x: Optional[float] = Field(None, description="Left X (0-1)")
    bbox_y: Optional[float] = Field(None, description="Top Y (0-1)")
    bbox_width: Optional[float] = Field(None, description="Width (0-1)")
    bbox_height: Optional[float] = Field(None, description="Height (0-1)")

class BaseDocItem(BaseModel):
    # Soft Handshake: Ignore extra noise from Parser/LLM
    model_config = ConfigDict(extra='ignore')

    id: str = Field(
        ..., 
        description="""
        Unique UUID for this specific item in the stream.
        """
    )
    
    # Prevent Context Shredding
    parent_id: Optional[str] = Field(
        None, 
        description="""
        UUID of the containing section or slide (if hierarchy is detected).
        """
    )
    
    layout_cluster_id: Optional[str] = Field(
        None, 
        description="""
        UUID linking components that belong together logically but are 
        separated spatially (e.g. A Chart and its specific Footnote).
        """
    )
    
    source: SourceRef = Field(
        ...,
        description="""
        Provenance data linking this item back to the raw file.
        """
    )

# ==============================================================================
# 2. SEMANTIC STREAM ITEMS
# ==============================================================================

class HeaderItem(BaseDocItem):
    """
    Structural marker for document sections.
    """
    type: Literal["header"] = "header"
    text: str = Field(
        ...,
        description="""
        The content of the header (e.g., 'Investment Highlights').
        """
    )
    level: int = Field(
        ...,
        description="""
        Hierarchy level (1=Title, 2=Section, 3=Subsection).
        """
    )

class NarrativeItem(BaseDocItem):
    """
    Replaces generic 'TextItem'.
    Captures the qualitative arguments of the deal.
    """
    type: Literal["narrative"] = "narrative"

    # Fine-grained citation bboxes (per-sentence, per-line)
    value_bboxes: Optional[Dict[str, List[List[float]]]] = Field(
        default=None,
        description="""Per-value bounding boxes for fine-grained citation highlighting.
        Keys: sentence text. Values: list of [bbox_x, bbox_y, bbox_w, bbox_h] (0-1 normalized).
        Multiple bboxes per key handle multi-line wrapping."""
    )

    text_content: str = Field(
        ...,
        description="""
        The raw text content of the paragraph or bullet point.
        """
    )
    
    # Deal Semantics
    is_strategic_claim: bool = Field(
        False, 
        description="""
        Does this text make a forward-looking value assertion?
        (e.g. 'Revenue growth driven by APAC expansion').
        """
    )
    
    sentiment: SentimentType = Field(
        "Unknown",
        description="""
        The impact of this statement on the investment thesis.
        """
    )
    
    # Insight Category (propagated from VLM Insight.category)
    category: Optional[CategoryType] = Field(
        None,
        description="""
        The 5-way insight category (Financial, Operational, Market, Strategic,
        Transactional). Propagated from VLM Insight.category when this
        NarrativeItem was derived from a visual insight. None for Docling text.
        """
    )

    # The "Hybrid Claim": Linking text to numbers
    referenced_metrics: Optional[List[dict]] = Field(
        None,
        description="""
        Simplified list of metrics specifically mentioned in this text block.
        Used to link the 'Argument' to the 'Proof'.
        """
    )

class FinancialTableRow(BaseModel):
    """
    Preserves indentation hierarchy (the 'Bridge').
    Allows reconstruction of the accounting tree (Net Income -> EBITDA).
    """
    # Soft Handshake: Ignore extra noise
    model_config = ConfigDict(extra='ignore')

    cells: List[str] = Field(..., description="The raw cell values for this row.")
    
    indentation_depth: int = Field(
        0, 
        description="Visual indentation level (0=Header, 1=Item, 2=Sub-item)."
    )
    
    row_type: Literal["header", "data", "total"] = Field(
        "data", 
        description="Semantic classification of the row."
    )

class FinancialTableItem(BaseDocItem):
    """
    Replaces generic 'TableItem'.
    Captures the rigorous accounting reality of the grid.
    """
    type: Literal["financial_table"] = "financial_table"

    # Fine-grained citation bboxes (per-cell)
    value_bboxes: Optional[Dict[str, List[List[float]]]] = Field(
        default=None,
        description="""Per-cell bounding boxes for fine-grained citation highlighting.
        Keys: cell text. Values: list of [bbox_x, bbox_y, bbox_w, bbox_h] (0-1 normalized).
        Multiple bboxes per key handle duplicate values across columns."""
    )

    # Representation
    html_repr: str = Field(
        ...,
        description="""
        Full HTML representation for frontend rendering.
        """
    )
    
    # Hierarchical rows instead of flat lists
    rows: List[FinancialTableRow] = Field(
        default_factory=list,
        description="""
        Structured data grid preserving indentation and row hierarchy.
        """
    )
    
    # [FLATTENED METADATA]
    # Note: accounting_basis remains 'str' because it uses the dynamic "Compound Vocabulary"
    # (e.g. "Pro Forma Adjusted") which is too complex for a strict Enum.
    accounting_basis: str = Field(
        default="Unknown",
        description="""
        GAAP, IFRS, or Non-GAAP/Adjusted status derived from table headers.
        """
    )

    # [STRICT] Use Enum to catch Parser drift (e.g. ensure 'H' not 'HY')
    periodicity: Optional[PeriodicityType] = Field(
        default=None,
        description="""
        The dominant timeframe of the table columns (e.g. 'FY', 'Q', 'LTM').
        """
    )
    
    # [STRICT] Use Enum for ISO 4217 compliance
    currency: Optional[CurrencyType] = Field(
        None,
        description="""
        The dominant currency detected in the table headers.
        """
    )

class VisualItem(BaseDocItem):
    """
    Container for Charts, Maps, and Complex Deal Art.
    Wraps the output of the Vision Tool.
    """
    type: Literal["visual"] = "visual"

    # Fine-grained citation bboxes (per-value, per-series-label)
    value_bboxes: Optional[Dict[str, List[List[float]]]] = Field(
        default=None,
        description="""Per-value bounding boxes for fine-grained citation highlighting.
        Keys: normalized float strings for numbers, series labels for legends.
        Values: list of [bbox_x, bbox_y, bbox_w, bbox_h] (0-1 normalized).
        Multiple bboxes per key handle duplicate values."""
    )

    archetype: VisualArchetype = Field(
        ...,
        description="""
        The semantic classification (Market Map, Waterfall, Org Chart).
        """
    )
    
    # Extracted content
    title: str = Field(
        ...,
        description="""
        The technical title of the chart/visual.
        """
    )
    
    summary: str = Field(
        ...,
        description="""
        High-density summary of the insights derived from this visual.
        """
    )
    
    # [FIX] Updated to accept MetricSeries (Grouped Data)
    metrics: List[MetricSeries] = Field(
        default_factory=list,
        description="""
        The structured quantitative data extracted from the visual, grouped by Series.
        """
    )

class ChartTableItem(BaseDocItem):
    """
    The "Football Field" Solver.
    A hybrid artifact that is visually a chart but structurally a table.
    """
    type: Literal["chart_table"] = "chart_table"

    # Fine-grained citation bboxes (per-value, per-series-label)
    value_bboxes: Optional[Dict[str, List[List[float]]]] = Field(
        default=None,
        description="""Per-value bounding boxes for fine-grained citation highlighting.
        Keys: normalized float strings for numbers, series labels for legends.
        Values: list of [bbox_x, bbox_y, bbox_w, bbox_h] (0-1 normalized)."""
    )

    archetype: VisualArchetype = Field(
        ...,
        description="""
        Usually 'WATERFALL' or 'VALUATION_FIELD' (Floating Bars).
        """
    )
    
    title: str = Field(..., description="Chart Title.")
    
    # Dual Identity: Contains both grid data and visual metrics
    table_rows: List[FinancialTableRow] = Field(
        default_factory=list,
        description="The tabular data underpinning the chart."
    )
    
    # [FIX] Updated to accept MetricSeries
    visual_metrics: List[MetricSeries] = Field(
        default_factory=list,
        description="The explicit data labels rendered on the chart, grouped by Series."
    )

# ==============================================================================
# 3. THE DISCRIMINATED UNION
# ==============================================================================

DocItem = Annotated[
    Union[HeaderItem, NarrativeItem, FinancialTableItem, VisualItem, ChartTableItem],
    Field(discriminator="type")
]

# ==============================================================================
# 4. THE ROUTING SIGNALS (Pure Data, No Logic)
# ==============================================================================

class RoutingSignals(BaseModel):
    """
    Computed properties populated by the Service Layer, NOT the Schema.
    """
    # Soft Handshake: Ignore extra noise
    model_config = ConfigDict(extra='ignore')

    detected_artifact_type: Literal["CIM", "LenderPres", "FinancialModel", "Legal", "Unknown"] = Field(
        "Unknown",
        description="""
        The classified document type based on its content composition.
        """
    )
    
    # Specific Flags
    has_valuation_models: bool = Field(
        False,
        description="""
        True if Valuation Football Fields or Sensitivity Tables are detected.
        """
    )
    
    has_pro_forma_adjustments: bool = Field(
        False,
        description="""
        True if 'Pro Forma' or 'Adjusted' financial tables are detected.
        """
    )
    
    has_legal_definitions: bool = Field(
        False,
        description="""
        True if significant legal definitions sections are detected.
        """
    )
    
    deal_relevance_score: float = Field(
        0.0, 
        ge=0.0, 
        le=1.0,
        description="""
        Heuristic score (0-1) indicating the density of deal-relevant data.
        """
    )

class UnifiedDocument(BaseModel):
    """
    The Master Container.
    Represents the full "Parsed Reality" of the file.
    """
    # Soft Handshake: Ignore extra noise
    model_config = ConfigDict(extra='ignore')

    doc_id: str = Field(
        ...,
        description="""
        Unique ID for the document processing job.
        """
    )
    
    filename: str = Field(
        ...,
        description="""
        Original filename.
        """
    )
    
    # The Valid Stream
    items: List[DocItem] = Field(
        default_factory=list,
        description="""
        The linear stream of successfully validated content items.
        """
    )
    
    # The Hospital (Quarantine for bad data)
    quarantined_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="""
        Items that failed strict validation. Stored here for audit/debugging
        instead of crashing the pipeline.
        """
    )
    
    # The Brain
    signals: RoutingSignals = Field(
        default_factory=RoutingSignals,
        description="""
        Computed routing signals and classification tags.
        """
    )