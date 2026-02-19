"""
Visual Extractor Tool: The Forensic "Eye" (Perception Engine)
=============================================================

1. THE MISSION (Overall Approach)
---------------------------------
This module bridges the gap between raw pixels and structured financial data.
It orchestrates the VLM (Visual Language Model) to perform "Forensic Extraction"
governed by the strict schemas defined in `src.schemas.vision_output`.

The architectural flow is designed as a **"Cognitive Sandwich"**:
- **Context (The Foundation):** Injects precise Layout Hints ("Scout Dossier") 
  and full OCR text to ground the model in spatial reality.
- **Visuals (The Input):** Injects high-fidelity, contrast-enhanced images 
  optimized for reading dense financial grids.
- **Logic (The Constraint):** Enforces a strict "Audit First" protocol via 
  System Prompts to prevent hallucination.

2. THE MECHANISM (Implementation)
---------------------------------
To ensure robust extraction, the tool employs four key technical strategies:

A. **Pre-Processing Pipeline:** Uses CPU-bound threads to optimize image 
   contrast and sharpness, making faint gridlines and axis labels legible.

B. **Dynamic Prompt Construction:** Fuses "Spatial Anchors" (OCR) with 
   "Layout Hints" to guide the VLM's attention to specific bounding boxes.

C. **Resilience Layers:** Implements async retries with jittered backoff and 
   strict timeouts to prevent pipeline hangs during high-load.

D. **Observability:** Logs extraction confidence scores and metric counts, 
   providing immediate visibility into data quality via LangSmith.
"""

# ==============================================================================
# 👁️ VISUAL EXTRACTOR - visual_extractor.py
# ==============================================================================

import asyncio
import atexit
import base64
import io
import logging
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Optional, Dict, Any, List, Tuple

# Third-party
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import traceable
from PIL import Image, ImageEnhance

# Internal
from src.config import SystemConfig
from src.prompts import PROMPT_VISUAL_EXTRACTOR
from src.schemas.layout_output import PageLayout as LayoutHint 
from src.schemas.vision_output import VisionPageResult

logger = logging.getLogger(__name__)

# ==============================================================================
# ⚙️ CONFIGURATION (TUNABLES)
# ==============================================================================

# 1. MODEL PARAMETERS
VISION_TEMPERATURE = 0.0
VISION_MAX_TOKENS = 16000     # Bumped for GPT-4.1-mini (1M Context)
LLM_REQUEST_TIMEOUT = 300     # Seconds for the API call itself

# 2. IMAGE PRE-PROCESSING
CONTRAST_FACTOR = 1.4         # +40% contrast helps separate light gridlines
SHARPNESS_FACTOR = 1.5        # +50% sharpness improves small axis label OCR
MAX_IMAGE_DIM = 3000          # Pixels - balance between detail and token cost
JPEG_QUALITY = 85             # Compression quality (0-100)

# 3. RESILIENCE & CONCURRENCY
RETRY_COUNT = 3
RETRY_BACKOFF = 1             # Seconds
PAGE_EXTRACTION_TIMEOUT = 180 # Hard cap (seconds) for end-to-end page processing
WORKER_CAP = 4                # Max threads for image processing

# ==============================================================================
# ⚙️ RESOURCE MANAGEMENT
# ==============================================================================
# Determine worker count based on CPU cores, capped for stability
_MAX_WORKERS = min(WORKER_CAP, (os.cpu_count() or 1))
_IMG_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS)


def _shutdown_executor():
    """Ensure thread pool is closed on program exit."""
    _IMG_EXECUTOR.shutdown(wait=False)


atexit.register(_shutdown_executor)


# ==============================================================================
# 🛡️ RESILIENCE UTILS
# ==============================================================================
def async_retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if retry_attempt == retries:
                        logger.error(
                            f"❌ Visual Extraction Failed after {retries} retries. "
                            f"Error: {e}"
                        )
                        return None  # Return None to allow graceful degradation

                    # Jittered backoff to prevent thundering herd
                    sleep_time = (
                        (backoff_in_seconds * 2 ** retry_attempt) + 
                        random.uniform(0, 1)
                    )
                    logger.warning(
                        f"⚠️ VLM Glitch: {e}. Retrying in {sleep_time:.2f}s..."
                    )
                    await asyncio.sleep(sleep_time)
                    retry_attempt += 1
        return wrapper
    return decorator

# Universal Payload Sanitizer
def mask_heavy_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggressively masks ANY heavy payload in the inputs, regardless of key name.
    This prevents 'ConnectionError' in LangSmith when uploading massive traces.
    """
    clean_inputs = inputs.copy()
    for key, value in clean_inputs.items():
        # Mask Raw Bytes (The Image)
        if isinstance(value, bytes):
            clean_inputs[key] = f"<MASKED_BYTES: {len(value)} bytes>"
        # Mask Massive Strings (Base64) - Threshold: 2KB
        elif isinstance(value, str) and len(value) > 2000:
            if "ocr_context" in key:
                continue # Keep OCR text, it's valuable and usually <100KB
            clean_inputs[key] = f"<MASKED_STRING: {len(value)} chars>"
    return clean_inputs


# ==============================================================================
# 📐 COLUMN MANIFEST BUILDER
# ==============================================================================
_OCR_LINE_RE = re.compile(r'\[(\d+),\s*(\d+)\]\s*(.*)')
_NUMERIC_RE = re.compile(r'^[+-]?[\d,]+\.?\d*%?$')


def _parse_manifest_number(text: str) -> float:
    """Parse a manifest value like '-5,151' or '14,224' to float. Returns NaN on failure."""
    cleaned = text.replace(',', '').replace('%', '')
    try:
        return float(cleaned)
    except ValueError:
        return float('nan')


def _remove_partial_sums(
    manifest: Dict[str, List[str]], 
    zone_order: List[str]
) -> Dict[str, List[str]]:
    """
    Remove values that are contiguous partial sums of other zone values.

    In waterfall charts, running subtotals or group annotations (e.g. "Gross
    Profit -5,151") sometimes appear in the OCR. These are mathematically
    derivable from the bar data labels: they equal the sum of 2+ contiguous
    zone values. This function detects and removes them.

    Algorithm: for each zone with >1 value, try each candidate as the
    "primary". Build the full ordered value list, compute all contiguous
    partial sums of 2+ zones, and check if the other candidate matches any.
    """
    

    # Identify single-value zones (primaries are known)
    multi_zones = {label for label in zone_order if len(manifest.get(label, [])) > 1}
    if not multi_zones:
        return manifest

    # For single-value zones, parse the primary value
    single_primaries = {}  # {label: float}
    for label in zone_order:
        vals = manifest.get(label, [])
        if len(vals) == 1:
            single_primaries[label] = _parse_manifest_number(vals[0])

    cleaned = dict(manifest)

    for mz_label in multi_zones:
        candidates = manifest[mz_label]
        survivors = []

        for i, candidate_text in enumerate(candidates):
            candidate_val = _parse_manifest_number(candidate_text)
            if math.isnan(candidate_val):
                survivors.append(candidate_text)
                continue

            # Use another candidate as the "primary" for this zone
            other_primary = None
            for j, other_text in enumerate(candidates):
                if j != i:
                    v = _parse_manifest_number(other_text)
                    if not math.isnan(v):
                        other_primary = v
                        break

            if other_primary is None:
                survivors.append(candidate_text)
                continue

            # Build ordered value list with other_primary for this zone
            ordered_vals = []
            for label in zone_order:
                if label == mz_label:
                    ordered_vals.append(other_primary)
                elif label in single_primaries:
                    ordered_vals.append(single_primaries[label])

            # Compute all contiguous partial sums of 2+ values
            is_partial_sum = False
            n = len(ordered_vals)
            for start in range(n):
                running = 0.0
                for end in range(start, n):
                    running += ordered_vals[end]
                    if end > start and abs(running - candidate_val) < 1.0:
                        is_partial_sum = True
                        break
                if is_partial_sum:
                    break

            if not is_partial_sum:
                survivors.append(candidate_text)

        cleaned[mz_label] = survivors if survivors else candidates  # fallback: keep all if everything got removed

    return {k: v for k, v in cleaned.items() if v}


def _synthesize_stacked_total(response, title: str, num_regions: int = 1) -> None:
    """
    If the VLM extracted 2+ level series with matching labels but no total,
    compute the total deterministically and insert it at position 0.

    This is a safety net for VLM non-compliance on the stacked bar total rule.
    The VLM consistently mentions totals in audit_log/summary but omits the
    MetricSeries. Rather than fighting the VLM, we compute it from the segments
    it already extracted correctly.

    When num_regions > 1, series likely belong to different side-by-side charts
    (e.g. Revenue bars next to Operating Income bars) — summing them would be
    financially nonsensical, so we skip synthesis entirely.
    """
    from src.schemas.vision_output import MetricSeries, DataPoint

    if not response or not response.metrics:
        return

    # Multiple chart regions → series belong to different charts, not stacked segments
    if num_regions > 1:
        return

    # Already has a total? Skip.
    if any(s.series_label.lower().startswith("total") for s in response.metrics):
        return

    # Collect level series with currency-denominated points (skip CAGR/percent)
    level_series = [
        s for s in response.metrics
        if s.series_nature == "level"
        and s.data_points
        and s.data_points[0].currency != "None"
    ]

    if len(level_series) < 2:
        return

    # All level series must share the same labels
    label_sets = [tuple(dp.label for dp in s.data_points) for s in level_series]
    if len(set(label_sets)) != 1:
        return

    # Sum per label
    labels = label_sets[0]
    ref = level_series[0].data_points[0]
    total_points = []
    for i, label in enumerate(labels):
        total = sum(
            s.data_points[i].numeric_value
            for s in level_series
            if s.data_points[i].numeric_value is not None
        )
        total_points.append(DataPoint(
            label=label,
            numeric_value=round(total, 2),
            currency=ref.currency,
            magnitude=ref.magnitude,
            original_text=f"Σ={round(total, 1)}",
        ))

    subject = title.replace(":", "").strip() if title else "Market"
    response.metrics.insert(0, MetricSeries(
        series_label=f"Total {subject}",
        series_nature="level",
        data_points=total_points,
    ))
    logger.info(f"📊 Synthesized total: 'Total {subject}' ({len(total_points)} pts)")


# --- Comparative-claim correction vocabulary ---
_COMP_LOWER = [
    'lower', 'below', 'less', 'smaller', 'fewer',
    'beneath', 'under', 'trailing', 'lagging', 'behind',
]
_COMP_HIGHER = [
    'higher', 'above', 'greater', 'larger',
    'exceeding', 'exceeds', 'surpassing', 'outpacing', 'ahead',
]
_COMP_SWAP = {
    # lower-family → natural antonym
    'lower': 'higher', 'below': 'above', 'less': 'greater',
    'smaller': 'larger', 'fewer': 'more', 'beneath': 'above',
    'under': 'over', 'trailing': 'leading', 'lagging': 'leading',
    'behind': 'ahead',
    # higher-family → natural antonym
    'higher': 'lower', 'above': 'below', 'greater': 'less',
    'larger': 'smaller', 'exceeding': 'trailing', 'exceeds': 'trails',
    'surpassing': 'trailing', 'outpacing': 'lagging', 'ahead': 'behind',
}


def _validate_summary_consistency(response) -> None:
    """
    Post-extraction: cross-check comparative claims in summary against metric averages.
    Detects "A is lower than B" when data shows avg(A) > avg(B), and corrects the claim.

    Strategy: for each pair of series (A, B), search the summary for
    "{A} ... {comparative} than {B}" and verify the direction against actual averages.
    Only corrects comparative words that directly precede the second series name,
    avoiding false matches on intermediate comparisons.
    """
    if not response or not response.metrics or not response.summary:
        return

    # Build {label: average} for each series with numeric data
    series_avgs: Dict[str, float] = {}
    for s in response.metrics:
        vals = [dp.numeric_value for dp in s.data_points if dp.numeric_value is not None]
        if vals:
            series_avgs[s.series_label] = sum(vals) / len(vals)

    if len(series_avgs) < 2:
        return

    summary = response.summary
    corrections = 0

    for label_a, avg_a in series_avgs.items():
        for label_b, avg_b in series_avgs.items():
            if label_a == label_b or abs(avg_a - avg_b) < 0.01:
                continue

            # Check both "lower" and "higher" word families
            for word_list, claim_dir in [(_COMP_LOWER, 'lower'), (_COMP_HIGHER, 'higher')]:
                for word in word_list:
                    # Pattern: "{label_a} ... {word} [than] {label_b}"
                    pat = re.compile(
                        re.escape(label_a) + r'.{1,150}?\b(' + re.escape(word)
                        + r')\s+(?:than\s+)?' + re.escape(label_b),
                        re.IGNORECASE | re.DOTALL
                    )
                    m = pat.search(summary)
                    if not m:
                        continue

                    is_wrong = (
                        (claim_dir == 'lower' and avg_a > avg_b)
                        or (claim_dir == 'higher' and avg_a < avg_b)
                    )

                    if is_wrong:
                        old_word = m.group(1)
                        new_word = _COMP_SWAP.get(old_word.lower(), 'higher' if claim_dir == 'lower' else 'lower')
                        if old_word[0].isupper():
                            new_word = new_word.capitalize()
                        summary = summary[:m.start(1)] + new_word + summary[m.end(1):]
                        corrections += 1
                        logger.warning(
                            f"📊 Summary fix: '{label_a}' (avg {avg_a:.0f}) is NOT "
                            f"{old_word} than '{label_b}' (avg {avg_b:.0f}) — corrected"
                        )

    if corrections:
        response.summary = summary
        logger.info(f"📊 Summary validator: {corrections} claim(s) corrected")


def _validate_insight_categories(response) -> None:
    """
    Post-extraction: reclassify 'Financial' insights to 'Market' when they reference
    third-party/market series rather than company financial data.

    Works per-insight, not per-page, so mixed pages (e.g. company revenue vs
    market benchmark) are handled correctly: insights referencing only market
    series are reclassified, those referencing company series stay 'Financial'.

    Detection: a series is "market data" if it has data_provenance (third-party
    source) and no accounting_basis (not company financial statements).
    """
    if not response or not response.metrics or not response.insights:
        return

    # Classify each series as market or company
    market_labels: set = set()
    company_labels: set = set()
    for s in response.metrics:
        is_market = (
            s.data_provenance
            and s.data_provenance.strip()
            and 'management' not in s.data_provenance.lower()
            and not (s.accounting_basis and s.accounting_basis.strip())
        )
        if is_market:
            market_labels.add(s.series_label.lower())
        else:
            company_labels.add(s.series_label.lower())

    if not market_labels:
        return  # no market data on this page

    all_market = len(company_labels) == 0

    corrections = 0
    for insight in response.insights:
        if insight.category != "Financial":
            continue

        content_lower = insight.content.lower()

        # Check which series this insight references by label match
        refs_market = any(label in content_lower for label in market_labels)
        refs_company = any(label in content_lower for label in company_labels)

        # Reclassify if:
        # 1. References market series but not company series, OR
        # 2. Entire page is market data and insight doesn't reference company
        if (refs_market and not refs_company) or (all_market and not refs_company):
            insight.category = "Market"
            corrections += 1

    if corrections:
        logger.info(
            f"📊 Category validator: {corrections} insight(s) reclassified "
            f"'Financial' → 'Market' (references third-party data)"
        )


def _build_column_manifest(
    x_axis_labels: List[str],
    ocr_context: str,
) -> Dict[str, List[str]]:
    """
    Match OCR numeric values to chart columns using X-coordinates.

    The VLM reads the image top-to-bottom and greedily assigns values to
    columns in visual scan order, which fails when positive and negative
    bar labels sit at different heights (e.g. waterfalls). This function
    uses the deterministic [Y, X] coordinates in the OCR grid to pair
    values with their correct columns.

    Strategy:
      1. Find each x-axis label's X-position in the bottom of the OCR grid.
      2. Compute zone boundaries as midpoints between adjacent labels.
      3. Assign every numeric OCR token to the zone it falls in.
    """
    if len(x_axis_labels) < 2 or not ocr_context:
        return {}

    # --- Parse OCR into (y, x, text) tuples ---
    tokens: List[tuple] = []
    for line in ocr_context.strip().split('\n'):
        m = _OCR_LINE_RE.match(line)
        if m:
            tokens.append((int(m.group(1)), int(m.group(2)), m.group(3).strip()))

    if not tokens:
        return {}

    # --- Locate x-axis labels in the bottom portion of the OCR ---
    # X-axis labels sit at the bottom of the chart (high Y values).
    max_y = max(t[0] for t in tokens)
    bottom_tokens = sorted(
        [(y, x, text) for y, x, text in tokens if y > max_y * 0.45],
        key=lambda t: t[1],  # sort by X
    )

    label_positions: List[tuple] = []  # [(label, x_center, y)]
    used_indices: set = set()

    for label in x_axis_labels:
        # First significant word (skip short connectors like '/', '&')
        words = [w for w in label.split() if len(w) > 1]
        if not words:
            continue
        first_word = words[0].lower()

        # Match to the leftmost unused bottom token with this word
        for j, (y, x, text) in enumerate(bottom_tokens):
            if j in used_indices:
                continue
            if text.lower() == first_word:
                label_positions.append((label, x, y))
                used_indices.add(j)
                break

    if len(label_positions) < 2:
        return {}

    # --- Compute Y-boundary to exclude headers/footers ---
    # Data values sit ABOVE the x-axis labels. Page numbers, footers sit
    # well below them. Allow a margin below the axis for negative bar labels.
    axis_y = max(lp[2] for lp in label_positions)
    y_margin = int(max_y * 0.05)
    y_floor = axis_y + y_margin  # exclude anything below this
    y_ceiling = int(max_y * 0.03)  # exclude top 3% (slide titles/headers)

    # --- Compute zone boundaries (midpoints between adjacent labels) ---
    max_x = max(t[1] for t in tokens) + 100
    zones: List[tuple] = []  # [(label, left_bound, right_bound)]
    for i, lp in enumerate(label_positions):
        label, x = lp[0], lp[1]
        left = 0 if i == 0 else (label_positions[i - 1][1] + x) // 2
        right = max_x if i == len(label_positions) - 1 else (x + label_positions[i + 1][1]) // 2
        zones.append((label, left, right))

    # --- Assign numeric tokens to zones ---
    manifest: Dict[str, List[str]] = {label: [] for label, _, _ in zones}

    for y, x, text in tokens:
        if y < y_ceiling or y > y_floor:
            continue  # skip headers, footers, page numbers
        if _NUMERIC_RE.match(text):
            for label, left, right in zones:
                if left <= x < right:
                    manifest[label].append(text)
                    break

    manifest = {k: v for k, v in manifest.items() if v}

    if not manifest:
        return {}

    # Skip manifest for dense zones (stacked/grouped bars).
    # The manifest fixes waterfall column-shift (1-2 values per zone).
    # For stacked bars (4+ values per zone) it adds noise without benefit
    # and buries the total alongside segments in an undifferentiated list.
    avg_density = sum(len(v) for v in manifest.values()) / len(manifest)
    if avg_density >= 4:
        return {}

    # Remove running subtotals / group annotations that are partial sums
    zone_order = [label for label, _, _ in zones]
    manifest = _remove_partial_sums(manifest, zone_order)

    return manifest


# ==============================================================================
# 👁️ VISUAL EXTRACTOR TOOL
# ==============================================================================
class VisualExtractor:
    def __init__(self):
        # 1. Initialize the Base LLM via Factory
        # [VERIFIED] Supports gpt-4.1-mini (April 2025 release)
        base_llm = SystemConfig.get_llm(
            model_name=SystemConfig.VISION_MODEL,
            temperature=VISION_TEMPERATURE,
            max_tokens=VISION_MAX_TOKENS,
            # We enforce timeout at the tool level now, but keep this as a fallback
            timeout=LLM_REQUEST_TIMEOUT 
        )

        # 2. Bind the "Cognitive Contract" (Structured Output)
        # We force the model to adhere to the `VisionPageResult` schema.
        self.vlm = base_llm.with_structured_output(VisionPageResult)

        logger.info(
            f"👁️ Visual Extractor initialized. "
            f"Model: {SystemConfig.VISION_MODEL} | Mode: {SystemConfig.DEPLOYMENT_MODE}"
        )

    def _process_image_bytes(
        self,
        img_bytes: bytes,
        max_dim: int = MAX_IMAGE_DIM
    ) -> str:
        """
        CPU-bound processing: Resize, Enhance, and Encode to JPEG.
        Executed in a separate thread to avoid blocking the Async Event Loop.
        """
        try:
            # Context Manager ensures buffer is closed immediately after use
            with io.BytesIO(img_bytes) as input_buf, Image.open(input_buf) as img:
                # 1. Handle Transparency (PNG -> RGB)
                if img.mode in ('RGBA', 'LA') or \
                   (img.mode == 'P' and 'transparency' in img.info):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert('RGB')

                # 2. Boost Contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(CONTRAST_FACTOR)

                # 3. Boost Sharpness
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(SHARPNESS_FACTOR)
                # ---------------------------------

                # Resize Logic
                width, height = img.size
                if max(width, height) > max_dim:
                    ratio = max_dim / max(width, height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Encode to Base64
                with io.BytesIO() as output_buf:
                    img.save(output_buf, format="JPEG", quality=JPEG_QUALITY)
                    return base64.b64encode(output_buf.getvalue()).decode("utf-8")
                    
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise

    # [FIX] Enhanced input masking for Parent Trace
    @traceable(
        name="Visual Extraction", 
        run_type="tool",
        process_inputs=mask_heavy_inputs
    )
    @async_retry_with_backoff(
        retries=RETRY_COUNT,
        backoff_in_seconds=RETRY_BACKOFF
    )
    async def analyze_full_page(
        self,
        image_data: bytes,
        ocr_context: str,
        layout_hint: Optional[LayoutHint]  # The "Analyzer" Hint
    ) -> Optional[VisionPageResult]:       # The "Forensic" Result
        """
        FORENSIC ANALYSIS ENTRY POINT.
        Wraps the implementation with a strict timeout to prevent pipeline hangs.
        """
        try:
            return await asyncio.wait_for(
                self._analyze_full_page_impl(image_data, ocr_context, layout_hint),
                timeout=PAGE_EXTRACTION_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(
                f"⏱️ Page extraction timed out after {PAGE_EXTRACTION_TIMEOUT}s. "
                f"Skipping page to preserve pipeline health."
            )
            return None

    async def _analyze_full_page_impl(
        self,
        image_data: bytes,
        ocr_context: str,
        layout_hint: Optional[LayoutHint]
    ) -> Optional[VisionPageResult]:
        """
        Implementation logic:
        1. Context: Injects FULL OCR Grid + Structured Layout Hints.
        2. Visuals: Injects the Enhanced Image.
        3. Logic: Injects the System Prompt with strict Protocols.
        """
        if not image_data:
            logger.warning("⚠️ No image data provided to VisualExtractor.")
            return None

        # 1. Offload Image Processing
        loop = asyncio.get_running_loop()
        try:
            base64_image = await loop.run_in_executor(
                _IMG_EXECUTOR, self._process_image_bytes, image_data
            )
        except Exception as e:
            logger.error(f"❌ Error encoding image: {e}")
            return None

        # 2. Construct the "Scout Dossier" (Structured Context)
        scout_summary = "=== STEP 2: DYNAMIC REGIONAL GUIDELINES ===\n"
        
        # Direct Pydantic Access (Cognitive Contract)
        # We trust that layout_hint matches the strict schema.
        charts = layout_hint.charts if layout_hint else []
        
        if charts:
            for i, chart in enumerate(charts):
                # Clean Enum Access
                vtype = (
                    chart.visual_type.value 
                    if hasattr(chart.visual_type, 'value') 
                    else str(chart.visual_type)
                )

                scout_summary += (
                    f"\nREGION_{i+1}:\n"
                    f"  BBox: {chart.bbox}\n"
                    f"  Type: {vtype}\n"
                    f"  Title: '{chart.title}'\n"
                )
                
                # Optional Enhancements (Check for existence or empty list)
                if chart.legend_keys:
                    scout_summary += f"  Legends: {', '.join(str(l) for l in chart.legend_keys)}\n"

                # Assume standard chart attributes; gracefully handle if specific axes are missing
                if hasattr(chart, 'x_axis_labels') and chart.x_axis_labels:
                    scout_summary += f"  X-Axis: {', '.join(str(l) for l in chart.x_axis_labels)}\n"

                if hasattr(chart, 'y_axis_labels') and chart.y_axis_labels:
                    scout_summary += f"  Y-Axis: {', '.join(str(l) for l in chart.y_axis_labels)}\n"

                # Build Column Manifest: pair each x-axis label with its
                # numeric values using OCR X-coordinates. This overrides the
                # VLM's visual top-to-bottom scan which causes column-shift
                # errors when positive and negative bars are at different heights.
                column_manifest = _build_column_manifest(
                    chart.x_axis_labels if hasattr(chart, 'x_axis_labels') else [],
                    ocr_context,
                )
                if column_manifest:
                    scout_summary += "  Column Manifest (values matched to columns by X-coordinate):\n"
                    for label, values in column_manifest.items():
                        scout_summary += f"    '{label}' → {', '.join(values)}\n"
                    scout_summary += (
                        "  CRITICAL: Use the Column Manifest for value-to-column "
                        "assignment. It is computed from spatial coordinates and "
                        "overrides visual scanning order.\n"
                    )
                else:
                    # Fallback: surface flat value lists when manifest can't be built
                    if hasattr(chart, 'aggregates') and chart.aggregates:
                        scout_summary += f"  Aggregates (totals/endpoints): {', '.join(str(a) for a in chart.aggregates)}\n"
                    if hasattr(chart, 'constituents') and chart.constituents:
                        scout_summary += f"  Constituents (drivers/segments): {', '.join(str(c) for c in chart.constituents)}\n"

                # Qualitative annotations detected by layout phase.
                # Signal to visual extractor: these MUST become insights.
                if hasattr(chart, 'annotation_texts') and chart.annotation_texts:
                    scout_summary += (
                        f"  Annotations (MUST generate an insight for each): "
                        f"{'; '.join(str(a) for a in chart.annotation_texts)}\n"
                    )

        else:
            scout_summary += (
                "No specific charts detected via layout analysis. "
                "Scan full page for embedded data tables or text logic.\n"
            )

        # 3. Build the Final Cognitive Prompt
        # [OBSERVABILITY] Warn if OCR context is truly massive (>100k chars)
        if len(ocr_context) > 100_000:
            logger.warning(
                f"⚠️ Large OCR context: {len(ocr_context):,} chars. "
                f"Page may have extremely dense text or complex layout."
            )

        # [RESTORED] The "Cognitive Contract" Instructions
        final_prompt_context = (
            f"=== STEP 1: SPATIAL ANCHORS (FULL OCR) ===\n"
            f"{ocr_context}\n\n"  
            
            f"{scout_summary}\n"
            
            f"=== INSTRUCTION ===\n"
            f"1. PERCEPTION ONLY: Capture data exactly as seen. Do NOT normalize (e.g. keep 'Rev', don't write 'Revenue').\n"
            f"2. AUDIT FIRST: Fill the `audit_log` before extracting any metrics. Analyze Units, Legends, and Periodicity here.\n"
            f"3. PERIODICITY: Identify the Page Default. If a specific series/point differs (e.g. 'LTM'), override it at the local level.\n"
            f"4. MEMO ROWS: Look BELOW charts for table rows (e.g. 'Memo: CapEx'). Extract these as new Series.\n"
            f"5. CONFLICT RESOLUTION: If OCR text disagrees with visual bar height, TRUST THE VISUAL BAR."
        )

        # 4. Build Messages
        messages = [
            SystemMessage(content=PROMPT_VISUAL_EXTRACTOR),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": final_prompt_context
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                ]
            )
        ]

        # 5. Invoke VLM with Structured Output
        # [CRITICAL FIX] Explicitly disable the RunTree Context.
        # This prevents LangChain from auto-creating a Child Span with the 10MB payload.
        # This supersedes config={"callbacks": []} which was found ineffective for auto-tracing.
        with tracing_v2_enabled(False):
            response = await self.vlm.ainvoke(messages)
        
        # 6. Post-extraction validators
        if response and response.metrics:
            chart_title = charts[0].title if charts else (response.title or "")
            _synthesize_stacked_total(response, chart_title, num_regions=len(charts) if charts else 1)
            _validate_summary_consistency(response)
            _validate_insight_categories(response)

        # 7. Observability Logging
        if response:
            metric_count = len(response.metrics) if response.metrics else 0
            insight_count = len(response.insights) if response.insights else 0
            logger.info(
                f"✅ Visual Extraction Complete | "
                f"Confidence: {response.confidence_score:.2f} | "
                f"Metrics: {metric_count} series | "
                f"Insights: {insight_count} items"
            )
            
            if response.confidence_score < 0.5:
                logger.warning(
                    f"⚠️ Low confidence extraction ({response.confidence_score:.2f}). "
                    f"Page may have poor visual quality or complex layout."
                )
        else:
            logger.error("❌ Visual extraction returned None")

        return response

# Singleton Instance
visual_extractor = VisualExtractor()