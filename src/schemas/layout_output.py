"""
Schema definitions for Layout Analysis (The "Glance" Phase).

GOAL: This file defines the output of the Object Detection layer (`layout_scanner.py`).
It serves as the "Map" that tells the Vision Tool (`vision.py`) WHERE to look 
before it starts reading.

ARCHITECTURAL NOTE:
- Coordinate System: Uses a 0-1000 Integer Grid (not 0.0-1.0 floats).
  Why? Integer grids are faster for Intersection-over-Union (IoU) calculations
  during the detection phase.
- downstream Consumption: These 'ChartRegion' objects are passed to the
  Vision Tool, which uses the 'bbox' to crop the high-res image for analysis.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


# ==============================================================================
# 📊 LAYOUT SCHEMAS
# ==============================================================================

class ChartRegion(BaseModel):
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
    
    chart_type: str = Field(
        ...,
        description="""
        The detected class. e.g. 'Bar Chart', 'Waterfall', 'Pie', 'Table'.
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

    is_infographic: bool = Field(
        default=False,
        description="""
        True for non-standard visualizations (radial maps, process flows) 
        where X/Y axes may not apply strictly.
        """
    )

    # Explicit Orientation
    axis_orientation: Literal["Bottom", "Left", "Top", "Right"] = Field(
        ...,
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

    # --- AXIS DATA (Explicit Separation) ---
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

    # CONTENT EXTRACTION
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

    @field_validator('axis_baseline_y')
    def validate_baseline(cls, v, info: ValidationInfo):
        if 'bbox' in info.data and v is not None:
            ymin, _, ymax, _ = info.data['bbox']
            # Allow a small buffer (50 units) for labels just outside the box
            if not (ymin <= v <= ymax + 50):
                return ymax
        return v


class PageLayout(BaseModel):
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