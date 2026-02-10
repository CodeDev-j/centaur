"""
Schema definitions for Layout Analysis (The "Glance" Phase).

GOAL: This file defines the output of the Object Detection layer (`layout_analyzer.py`).
It serves as the "Map" that tells the Vision Tool (`visual_extractor.py`) WHERE to look 
and WHAT to look for (Archetype).

ARCHITECTURAL NOTE:
- Coordinate System: Uses a 0-1000 Integer Grid (not 0.0-1.0 floats).
  Why? Integer grids are faster for Intersection-over-Union (IoU) calculations
  during the detection phase.
- Deal-Ready: Adopts 'VisualArchetype' to distinguish between 'Cartesian' (Charts) 
  and 'Semantic' (Market Maps, Waterfalls, Org Charts).
"""

from typing import List, Literal, Optional, Any, Annotated
from enum import Enum
from pydantic import BaseModel, Field, ValidationInfo, field_validator, ConfigDict, BeforeValidator

# ==============================================================================
# 🧩 ENUMS & SHARED TYPES
# ==============================================================================

class VisualArchetype(str, Enum):
    # Standard Financials
    CARTESIAN = "Cartesian"     # Bar, Line, Scatter (Data-driven)
    TABLE = "Table"             # Financial Statements
    WATERFALL = "Waterfall"     # EBITDA Bridges (Critical!)
    VALUATION_FIELD = "ValuationField" # Floating Bar / Football Field (Audit Requirement)
    
    # Deal Specific (The "Infographics")
    MARKET_MAP = "MarketMap"    # Logos on X/Y (Competitive Landscape)
    HIERARCHY = "Hierarchy"     # Org Charts, Legal Entity Trees
    PROCESS_FLOW = "ProcessFlow" # Supply Chain, Synergy Plans
    OTHER = "Other"

def normalize_archetype(v: Any) -> str:
    """Soft Handshake for Enum Casing"""
    if isinstance(v, str):
        # Map common VLM hallucinations to Schema
        clean = v.strip().lower()
        if "bar" in clean or "line" in clean:
            return "Cartesian"
        if "water" in clean:
            return "Waterfall"
        if "football" in clean:
            return "ValuationField"
        # Attempt case-insensitive match
        for member in VisualArchetype:
            if member.value.lower() == clean:
                return member.value
    return v

# Annotated Type for Schema Injection
VisualArchetypeField = Annotated[
    VisualArchetype, 
    BeforeValidator(normalize_archetype)
]

class VisualEntity(BaseModel):
    """
    Represents an entity inside a non-standard visual (e.g. Logo on a Market Map).
    Replaces generic text extraction for semantic objects.
    """
    # Soft Handshake: Ignore extra noise from LLM
    model_config = ConfigDict(extra='ignore')

    label: str = Field(
        ..., 
        description="""
        Text label or entity name (e.g. 'Competitor X', 'CTO').
        """
    )
    
    category_quadrant: Optional[str] = Field(
        None, 
        description="""
        e.g. 'Leaders', 'Visionaries', 'High-Cost', 'Top-Right'.
        """
    )
    
    coordinates: Optional[List[float]] = Field(
        None, 
        description="""
        Relative X/Y position [0.0-1.0] within the bounding box.
        """
    )


# ==============================================================================
# 📊 LAYOUT SCHEMAS
# ==============================================================================

class ChartRegion(BaseModel):
    # 1. Soft Handshake: extra='ignore' to handle parser noise
    # 2. Schema Drift: populate_by_name=True for OpenAI compatibility
    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True
    )

    region_id: int = Field(
        ..., 
        description="""
        Index 1, 2... Unique identifier for the chart on this page.
        """
    )
    
    bbox: List[int] = Field(
        ...,
        description="""
        [ymin, xmin, ymax, xmax] bounding box using a 0-1000 integer scale.
        Example: [100, 100, 500, 900] represents the top half of the page.
        """
    )
    
    # Semantic Archetypes replace generic strings
    # Use Annotated field for soft matching
    visual_type: VisualArchetypeField = Field(
        default=VisualArchetype.CARTESIAN,
        description="""
        The semantic classification of the visual structure.
        """
    )

    # --- METADATA ---
    title: str = Field(
        default="",
        description="""
        The exact text string identified as the chart title based on 
        spatial proximity (e.g., the text line immediately above the bounding box).
        Do not generate or summarize; extract verbatim.
        """
    )
    
    footnotes: List[str] = Field(
        default_factory=list,
        description="""
        Source notes, citations, or methodology descriptions found 
        near the chart boundaries.
        """
    )

    # --- STRUCTURE ---
    is_variable_width: bool = Field(
        default=False,
        description="""
        True for complex charts like Marimekko, Mosaic, or Non-Uniform Time 
        charts where column widths encode data.
        """
    )

    # Explicit Orientation
    axis_orientation: Literal["Bottom", "Left", "Top", "Right"] = Field(
        default="Bottom",
        description="""
        Where are the primary category labels? 
        'Bottom' for standard vertical bars. 'Left' for horizontal bars.
        """
    )

    # Baseline Detection
    axis_baseline_y: Optional[int] = Field(
        None,
        description="""
        The Y-coordinate (0-1000) where the primary category labels 
        are aligned. Critical for identifying the "zero line" in Waterfalls.
        """
    )

    # --- CONTENT EXTRACTION (Standard) ---
    x_axis_labels: List[str] = Field(
        default_factory=list,
        description="""
        Categories or Time periods detected on the horizontal axis 
        (e.g. '2021', 'Q1', 'North America').
        """
    )

    y_axis_labels: List[str] = Field(
        default_factory=list,
        description="""
        Numeric values detected on the Primary Left-Hand Side (LHS) 
        vertical axis (e.g. '$0', '$50M').
        """
    )

    rhs_y_axis_labels: List[str] = Field(
        default_factory=list,
        description="""
        Numeric values detected on the Secondary Right-Hand Side (RHS) 
        vertical axis (e.g. '0%', '15%').
        """
    )

    # CONTENT EXTRACTION (Aggregates)
    aggregates: List[str] = Field(
        default_factory=list,
        description="""
        High-level sums or totals explicitly displayed on the chart 
        (e.g., Stack tops, Waterfall endpoints, Table 'Grand Total').
        """
    )

    constituents: List[str] = Field(
        default_factory=list,
        description="""
        The granular data points that make up the chart body 
        (e.g., Bar segments, Waterfall steps, Line points).
        """
    )

    legend_keys: List[str] = Field(
        default_factory=list,
        description="""
        Distinct series names identified in the chart legend.
        """
    )

    # --- DEAL SPECIFIC (Non-Standard) ---
    detected_entities: List[VisualEntity] = Field(
        default_factory=list,
        description="""
        Logos/People detected in Market Maps or Org Charts.
        """
    )

    @field_validator('axis_baseline_y')
    def validate_baseline(cls, v, info: ValidationInfo):
        if 'bbox' in info.data and v is not None:
            ymin, _, ymax, _ = info.data['bbox']
            # Allow a small buffer (50 units) for labels just outside the box
            if not (ymin <= v <= ymax + 50):
                return ymax
        return v


class PageLayout(BaseModel):
    # 1. Soft Handshake: extra='ignore' to handle parser noise
    # 2. Schema Drift: populate_by_name=True for OpenAI compatibility
    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True
    )

    has_charts: bool = Field(
        ...,
        description="""
        True if at least one data visualization is detected on the page.
        """
    )
    
    confidence_score: float = Field(
        ...,
        description="""
        Self-evaluation (0.0-1.0). Low score if the page is blurry, 
        watermarked, or the layout is ambiguous.
        """
    )
    
    charts: List[ChartRegion] = Field(
        default_factory=list,
        description="""
        List of detected chart regions found on the page.
        """
    )