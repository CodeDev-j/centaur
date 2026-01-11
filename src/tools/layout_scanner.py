from typing import List, Tuple, Literal, Optional
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
import logging

from src.config import SystemConfig

logger = logging.getLogger(__name__)

# --- SCHEMA ---
class ChartRegion(BaseModel):
    region_id: int = Field(..., description="Index 1, 2...")
    bbox: List[int] = Field(..., description="[ymin, xmin, ymax, xmax] bounding box (0-1000 scale)")
    chart_type: str = Field(..., description="e.g. 'Bar Chart', 'Waterfall', 'Pie', 'Table'")
    
    # Explicit Orientation
    axis_orientation: Literal["Bottom", "Left", "Top", "Right"] = Field(
        ..., 
        description="Where are the category labels? 'Bottom' for standard bars, 'Left' for horizontal bars."
    )
    
    # Baseline Detection (Deep Research Update)
    axis_baseline_y: Optional[int] = Field(
        None,
        description="The Y-coordinate (0-1000) where the primary category labels are aligned. Critical for Waterfalls."
    )
    
    # CONTENT EXTRACTION
    axis_labels: List[str] = Field(default_factory=list, description="The categorical labels on the primary axis (e.g. 'Q1', 'EBIT').")
    data_values: List[str] = Field(default_factory=list, description="Visible numeric values found in the chart (e.g. '958', '12.6%').")
    legend_keys: List[str] = Field(default_factory=list, description="Legend items if present (e.g. 'Revenue', 'Cost').")

    @field_validator('axis_baseline_y')
    def validate_baseline(cls, v, info: ValidationInfo):
        """
        Sanity Check: Ensure the baseline is actually within the vertical bounds of the chart.
        If the model hallucinates a baseline outside the box, clamp it to the bottom edge.
        """
        if 'bbox' in info.data and v is not None:
            ymin, _, ymax, _ = info.data['bbox']
            # Allow a small buffer (50 units) for labels just outside the box
            if not (ymin <= v <= ymax + 50):
                return ymax
        return v

    def get_axis_zone(self, buffer_percent: float = 0.20) -> Tuple[int, int]:
        """
        Refined Zone Logic: Uses baseline if available, else falls back to bottom percent.
        """
        ymin, xmin, ymax, xmax = self.bbox
        height = ymax - ymin

        if self.axis_baseline_y:
            # Create a focused zone +/- 5% height around the baseline
            margin = int(height * 0.05)
            # Create narrow zone around baseline, biased down
            return (
                max(ymin, self.axis_baseline_y - margin), 
                min(1000, self.axis_baseline_y + margin + margin) # Bias down
            )
        else:
            # Fallback to original "Bottom 25%" logic if detection fails
            zone_start = int(ymax - (height * buffer_percent))
            return (zone_start, ymax)

class PageLayout(BaseModel):
    has_charts: bool = Field(..., description="True if data visualization exists")
    confidence_score: float = Field(..., description="Self-evaluation (0.0-1.0). Low score if the page is blurry or layout is ambiguous.")
    charts: List[ChartRegion] = Field(default_factory=list, description="List of detected chart regions")

# --- PROMPT ---
PROMPT_LAYOUT_SCANNER = """
Role: Document Layout Scout.
Task: Detect and bound data visualizations for downstream extraction.

1. **Detection:** Are there any charts, graphs, or data tables?
   - If NO: Return `has_charts=false`.
   - If YES: Return `has_charts=true` and list each one as a `ChartRegion`.

2. **Broad Bounding Boxes (GREEDY STRATEGY):** - **CRITICAL:** Your BBox MUST include the **Chart Title**, the **Graphic**, the **Legend** (if any), and **ALL Axis Labels**.
   - **Footer Check:** In financial slides, X-Axis labels often sit at the very bottom (Y=800-950). EXTEND your BBox vertically to capture them.

3. **Structure & Content Analysis (CRITICAL):**
   - **Orientation:** Identify if labels are at the 'Bottom' (Standard) or 'Left' (Horizontal).
   - **Baseline (`axis_baseline_y`):** DETECT the exact Y-coordinate line where the text labels sit.
     - For a Waterfall chart, this is the line containing "EBIT 2023", "Adjustments", etc.
     - This is your semantic anchor.
   - **Extraction:**
     - **Labels:** Extract the "Immutable Axis Labels" found on the Baseline.
     - **Values:** Extract visible data labels (e.g. "958", "12.6%") found on bars or lines.
     - **Legend:** Extract legend keys **ONLY if clearly present**. Do not hallucinate a legend for Waterfall charts that lack one.

4. **Multi-Chart Handling:**
   - If a page has a "Revenue" chart on top and "EBIT" chart on bottom, create TWO distinct regions.

5. **Self-Correction:**
   - Assign a `confidence_score` (0.0 - 1.0). If the chart boundaries are fuzzy or the axis is hard to find, lower the score.
"""

class LayoutScanner:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=SystemConfig.LAYOUT_MODEL, 
            temperature=0.0,
            max_tokens=1000
        ).with_structured_output(PageLayout)

    @traceable(name="Layout Scout", run_type="tool")
    async def scan(self, img_b64: str) -> PageLayout:
        msg = [
            SystemMessage(content=PROMPT_LAYOUT_SCANNER),
            HumanMessage(content=[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}])
        ]
        try:
            result = await self.llm.ainvoke(msg)
            # LOGGING THE SCORE
            if result.has_charts:
                logger.info(f"📸 Layout Scout: Found {len(result.charts)} regions. Confidence: {result.confidence_score:.2f}")
            return result
        except Exception as e:
            logger.warning(f"Layout scan failed: {e}")
            return PageLayout(has_charts=False, confidence_score=0.0, charts=[])

layout_scanner = LayoutScanner()