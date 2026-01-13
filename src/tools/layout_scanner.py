import re
import asyncio
from typing import List, Tuple, Literal, Optional
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from rapidocr_onnxruntime import RapidOCR
import numpy as np
import io
from PIL import Image
import base64
import logging

from src.config import SystemConfig
from src.prompts import PROMPT_LAYOUT_SCANNER, PROMPT_CONTENT_EXTRACTOR

logger = logging.getLogger(__name__)

# --- SCHEMAS ---

class ChartContent(BaseModel):
    """
    Focused schema for the secondary extraction pass.
    Strictly isolated to content fields to prevent hallucination of bbox/layout.
    """
    axis_labels: List[str] = Field(default_factory=list, description="The categorical labels on the primary axis.")
    data_values: List[str] = Field(default_factory=list, description="All visible numeric values.")
    legend_keys: List[str] = Field(default_factory=list, description="Legend items if present.")

class ChartRegion(BaseModel):
    region_id: int = Field(..., description="Index 1, 2...")
    bbox: List[int] = Field(..., description="[ymin, xmin, ymax, xmax] bounding box (0-1000 scale)")
    chart_type: str = Field(..., description="e.g. 'Bar Chart', 'Waterfall', 'Pie', 'Table'")
    
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
            
            # Bias downwards because labels are usually below the line
            # Returning tuple (ymin, ymax) for the zone
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

class LayoutScanner:
    
        # Captures: 
    # - 4-digit years (1990-2059): \b(19|20)\d{2}\b
    # - Fiscal Years (FY24, FY2024): FY\d{2,4}
    # - Estimates (2024E): \d{4}E
    # - Quarters (Q1, 1Q24): Q[1-4]| \d{1}Q\d{2}
    # - Relative: LTM, YTD
    # - Months (3-digit): Jan, Feb, Mar...
    _FINANCIAL_AXIS_PATTERN = re.compile(
        r"\b(?:(19|20)\d{2}E?|FY\d{2,4}|Q[1-4]|LTM|YTD|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", 
        re.IGNORECASE
    )

    def __init__(self):
        # Primary LLM for Layout Detection
        self.llm = ChatOpenAI(
            model=SystemConfig.LAYOUT_MODEL, 
            temperature=0.0,
            max_tokens=1000,
            request_timeout=180
        ).with_structured_output(PageLayout)
        
        # Secondary LLM for Isolated Content Extraction
        self.extractor_llm = ChatOpenAI(
            model=SystemConfig.LAYOUT_MODEL,
            temperature=0.0,
            max_tokens=2000,
            request_timeout=180
        ).with_structured_output(ChartContent)
        
        self.ocr = RapidOCR()

    # --- HORIZONTAL SLICER (New Method) ---
    def _detect_side_by_side(self, img_bytes: bytes, region: ChartRegion) -> List[ChartRegion]:
        """
        [FIX 1] Detects Side-by-Side charts by clustering X-coordinates of repeated axis labels.
        """
        try:
            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                w, h = pil_img.size
                ymin, xmin, ymax, xmax = region.bbox
                l, t, r, b = int(xmin/1000*w), int(ymin/1000*h), int(xmax/1000*w), int(ymax/1000*h)
                crop = pil_img.crop((l, t, r, b))
                crop_bytes = io.BytesIO()
                crop.save(crop_bytes, format='PNG')
                
                ocr_result, _ = self.ocr(crop_bytes.getvalue())
                if not ocr_result: return [region]
            
            # Collect centroids using the class constant
            label_centroids = []
            for item in ocr_result:
                box, text, _ = item
                if self._FINANCIAL_AXIS_PATTERN.search(text):
                    xs = [p[0] for p in box]
                    cx = sum(xs) / len(xs)
                    label_centroids.append(cx)

            if not label_centroids: return [region]

            # X-Axis Clustering (Look for gaps > 20% of width)
            label_centroids.sort()
            clusters = []
            current_cluster = [label_centroids[0]]
            
            gap_threshold = (r - l) * 0.15 # 15% width gap implies separation

            for x in label_centroids[1:]:
                if x - current_cluster[-1] < gap_threshold:
                    current_cluster.append(x)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [x]
            clusters.append(np.mean(current_cluster))

            if len(clusters) > 1:
                logger.info(f"✂️ Horizontal Slicer: Detected {len(clusters)} side-by-side charts. Splitting...")
                new_regions = []
                
                # Calculate split lines (midpoints between cluster centers)
                split_points = []
                bbox_width_norm = xmax - xmin
                crop_w = r - l # [CORRECT] Denominator is crop width
                
                for i in range(len(clusters) - 1):
                    mid_x_px = (clusters[i] + clusters[i+1]) / 2
                    page_split_x = int(xmin + (mid_x_px / crop_w) * bbox_width_norm)
                    split_points.append(page_split_x)

                current_xmin = xmin
                for split_x in split_points:
                    new_r = region.model_copy()
                    new_r.region_id = int(f"{region.region_id}{9}") # Distinct suffix
                    new_r.bbox = [ymin, current_xmin, ymax, split_x]
                    new_regions.append(new_r)
                    current_xmin = split_x
                
                last_r = region.model_copy()
                last_r.region_id = int(f"{region.region_id}{8}")
                last_r.bbox = [ymin, current_xmin, ymax, xmax]
                new_regions.append(last_r)
                return new_regions

            return [region]

        except Exception as e:
            logger.warning(f"Horizontal split failed: {e}")
            return [region]

    # --- VERTICAL SLICER (Existing Method) ---
    def _detect_repeated_axis(self, img_bytes: bytes, region: ChartRegion) -> List[ChartRegion]:
        """
        ALGORITHMIC POST-PROCESSOR:
        Detects multiple vertically stacked charts by looking for repeating X-axis patterns.
        Improved Slicer: Filters out 'weak' axis rows (like titles with 1 date).
        """
        try:
            # Crop image to the detected region
            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                w, h = pil_img.size
                ymin, xmin, ymax, xmax = region.bbox
                # Convert 0-1000 norm to pixels
                l, t, r, b = int(xmin/1000*w), int(ymin/1000*h), int(xmax/1000*w), int(ymax/1000*h)
                crop = pil_img.crop((l, t, r, b))
                crop_bytes = io.BytesIO()
                crop.save(crop_bytes, format='PNG')
                
                # Run RapidOCR on the crop
                ocr_result, _ = self.ocr(crop_bytes.getvalue())
                if not ocr_result: return [region]

            axis_y_coords = []
            
            for item in ocr_result:
                box, text, _ = item
                if self._FINANCIAL_AXIS_PATTERN.search(text):
                    ys = [p[1] for p in box]
                    cy = sum(ys) / len(ys)
                    axis_y_coords.append(cy)

            if not axis_y_coords:
                return [region]

            # Histogram Analysis: Group Y-coords into "Rows"
            axis_y_coords.sort()
            clusters = []
            if axis_y_coords:
                current_cluster = [axis_y_coords[0]]
                for y in axis_y_coords[1:]:
                    if y - current_cluster[-1] < 20: # 20px tolerance
                        current_cluster.append(y)
                    else:
                        clusters.append(current_cluster) # Append full cluster for density check
                        current_cluster = [y]
                clusters.append(current_cluster)

            # Filter weak clusters (Density Check) and get means
            valid_clusters = [np.mean(c) for c in clusters if len(c) >= 3]

            # Split Logic
            if len(valid_clusters) > 1:
                logger.info(f"✂️ Algorithmic Slicer: Detected {len(valid_clusters)} stacked charts in Region {region.region_id}. Splitting...")
                new_regions = []
                
                # Calculate split points (midpoints between clusters)
                # Convert crop-relative pixels back to 0-1000 page norm
                split_points = []
                
                # Normalize relative to BBox height (not Page height)
                bbox_height_norm = ymax - ymin
                
                for i in range(len(valid_clusters) - 1):
                    mid_y_px = (valid_clusters[i] + valid_clusters[i+1]) / 2
                    
                    # Correct Formula: ymin + (pixel_percent * bbox_height)
                    relative_offset = (mid_y_px / h) * bbox_height_norm
                    mid_y_norm = int(ymin + relative_offset)
                    
                    split_points.append(mid_y_norm)

                current_ymin = ymin
                # Create regions for all but the last chart
                for i, split_y in enumerate(split_points):
                    new_r = region.model_copy()
                    new_r.region_id = int(f"{region.region_id}{i+1}") # e.g. 11, 12
                    new_r.bbox = [current_ymin, xmin, split_y, xmax]
                    new_r.axis_baseline_y = split_y - 20 # Approximation
                    new_regions.append(new_r)
                    current_ymin = split_y
                
                # Final segment
                last_r = region.model_copy()
                last_r.region_id = int(f"{region.region_id}{len(valid_clusters)}")
                last_r.bbox = [current_ymin, xmin, ymax, xmax]
                new_regions.append(last_r)
                return new_regions

            return [region]

        except Exception as e:
            logger.warning(f"Vertical split failed: {e}")
            return [region]

    async def _extract_chart_content(self, img_bytes: bytes, region: ChartRegion, legend_hints: List[str] = []) -> ChartRegion:
        """
        Isolated extraction pass: Crops the image to the chart region and extracts data.
        """
        try:
            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                w, h = pil_img.size
                ymin, xmin, ymax, xmax = region.bbox
                
                # --- ADAPTIVE BUFFERING ---
                # Expand vertically to capture "Orbiting Context" (Titles above, Labels below)
                # 10% Top Buffer, 15% Bottom Buffer (for multi-line axis labels)
                height = ymax - ymin
                buffer_top = int(height * 0.10)
                buffer_bottom = int(height * 0.15)
                
                # Expand horizontally slightly (5%) to catch cut-off numbers
                width = xmax - xmin
                buffer_side = int(width * 0.05)

                # Apply and Clamp to 0-1000 norm
                new_ymin = max(0, ymin - buffer_top)
                new_ymax = min(1000, ymax + buffer_bottom)
                new_xmin = max(0, xmin - buffer_side)
                new_xmax = min(1000, xmax + buffer_side)

                # Convert to pixels
                l = int(new_xmin/1000 * w)
                t = int(new_ymin/1000 * h)
                r = int(new_xmax/1000 * w)
                b = int(new_ymax/1000 * h)

                crop = pil_img.crop((l, t, r, b))
                
                # Encode crop to base64 for VLM
                crop_bytes = io.BytesIO()
                crop.save(crop_bytes, format='PNG')
                crop_b64 = base64.b64encode(crop_bytes.getvalue()).decode("utf-8")

            # Inject hints into prompt
            formatted_prompt = PROMPT_CONTENT_EXTRACTOR + f"\n\nCONTEXT HINTS:\nThe parent document suggests these Legend Keys might apply here: {legend_hints}\nIf the crop is missing a legend, USE THESE KEYS to label the series."

            msg = [
                SystemMessage(content=formatted_prompt),
                HumanMessage(content=[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_b64}"}}])
            ]
            
            # Invoke VLM on specific crop
            result = await self.extractor_llm.ainvoke(msg)
            
            # Update the region with extracted data
            region.axis_labels = result.axis_labels
            region.data_values = result.data_values
            region.legend_keys = result.legend_keys
            
            return region
            
        except Exception as e:
            logger.error(f"Extraction failed for region {region.region_id}: {e}")
            return region

    @traceable(name="Layout Scout", run_type="tool")
    async def scan(self, img_b64: str) -> PageLayout:
        # Decode image for Python processing
        img_bytes = base64.b64decode(img_b64)
        
        # 1. PASS 1: Layout Detection (Global)
        msg = [
            SystemMessage(content=PROMPT_LAYOUT_SCANNER),
            HumanMessage(content=[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}])
        ]
        try:
            result = await self.llm.ainvoke(msg)
            # LOGGING THE SCORE
            if result.has_charts:
                # Harvest Global Legends from the "Scout" (Page Level)
                global_legends = []
                for chart in result.charts:
                    if chart.legend_keys:
                        global_legends.extend(chart.legend_keys)
                global_legends = list(set(global_legends))

                # 2. Algorithmic Refinement (Slicing)
                refined_charts = []
                for chart in result.charts:
                    if "Bar" in chart.chart_type or "Chart" in chart.chart_type:
                        # [FIX 1] Sequential Slicing: Vertical First, then Horizontal
                        v_splits = self._detect_repeated_axis(img_bytes, chart)
                        
                        for v_region in v_splits:
                            h_splits = self._detect_side_by_side(img_bytes, v_region)
                            refined_charts.extend(h_splits)
                    else:
                        refined_charts.append(chart)
                
                # 3. PASS 2: Parallel Isolated Content Extraction (Local)
                # <--- CHANGED: Uses asyncio.gather for parallel execution --->
                logger.info(f"🔄 Starting parallel extraction for {len(refined_charts)} regions...")
                
                extraction_tasks = [
                    self._extract_chart_content(img_bytes, chart, legend_hints=global_legends) 
                    for chart in refined_charts
                ]
                
                # Concurrency Guard: Handle exceptions gracefully
                final_charts = await asyncio.gather(*extraction_tasks, return_exceptions=True)
                
                # Filter out failed tasks to prevent crashing the whole page
                valid_charts = []
                for res in final_charts:
                    if isinstance(res, Exception):
                        logger.error(f"Chart extraction failed for a region: {res}")
                    else:
                        valid_charts.append(res)

                result.charts = valid_charts
                logger.info(f"✅ Layout Scout Complete: Processed {len(result.charts)} regions.")
            
            return result
        except Exception as e:
            logger.warning(f"Layout scan failed: {e}")
            return PageLayout(has_charts=False, confidence_score=0.0, charts=[])

layout_scanner = LayoutScanner()