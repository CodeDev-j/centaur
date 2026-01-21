from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, ValidationInfo

# ==============================================================================
# 📊 LAYOUT SCHEMAS
# ==============================================================================
class ChartRegion(BaseModel):
    region_id: int = Field(..., description="Index 1, 2...")
    bbox: List[int] = Field(
        ..., 
        description="[ymin, xmin, ymax, xmax] bounding box (0-1000 scale)"
    )
    chart_type: str = Field(
        ..., 
        description="e.g. 'Bar Chart', 'Waterfall', 'Pie', 'Table'"
    )
    
    # --- METADATA ---
    title: str = Field(
        default="", 
        description="The explicit chart title (e.g., 'Adjusted EBITDA')."
    )
    footnotes: List[str] = Field(
        default_factory=list, 
        description="Source notes or citations found near the chart."
    )
    
    # --- STRUCTURE ---
    is_variable_width: bool = Field(
        default=False, 
        description="True for Marimekko, Mosaic, or Non-Uniform Time charts where column widths vary."
    )
    
    is_infographic: bool = Field(
        default=False, 
        description="True for radial maps, process flows, or non-data visualizations."
    )
    
    # Explicit Orientation
    axis_orientation: Literal["Bottom", "Left", "Top", "Right"] = Field(
        ..., 
        description="Where are the category labels? 'Bottom' for standard bars, 'Left' for horizontal bars."
    )
    
    # Baseline Detection
    axis_baseline_y: Optional[int] = Field(
        None,
        description="The Y-coordinate (0-1000) where the primary category labels are aligned. Critical for Waterfalls."
    )
    
    # CONTENT EXTRACTION
    axis_labels: List[str] = Field(
        default_factory=list, 
        description="The categorical labels on the primary axis."
    )
    
    aggregates: List[str] = Field(
        default_factory=list, 
        description="High-level sums or totals explicitly displayed (e.g., Stack tops, Waterfall endpoints, Table 'Grand Total')."
    )
    
    constituents: List[str] = Field(
        default_factory=list, 
        description="The granular data points that make up the chart (e.g., Bar segments, Waterfall steps, Line points)."
    )
    
    legend_keys: List[str] = Field(
        default_factory=list, 
        description="Legend items if present."
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
    has_charts: bool = Field(..., description="True if data visualization exists")
    confidence_score: float = Field(
        ..., 
        description="Self-evaluation (0.0-1.0). Low score if the page is blurry or layout is ambiguous."
    )
    charts: List[ChartRegion] = Field(
        default_factory=list, 
        description="List of detected chart regions"
    )