import base64
import io
import logging
from typing import Literal
from pathlib import Path

# Third-party
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import traceable

# Internal
from src.config import SystemConfig, SystemPaths
from src.utils.resilience import retry_with_backoff  # <--- NEW IMPORT

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
MAX_IMG_DIM = 1280  

class VisionTool:
    """
    The 'Eyes' of the Centaur.
    Uses a configurable VLM to analyze visual artifacts with financial rigor.
    """

    def __init__(self):
        logger.info(f"👁️ Vision Tool initialized with model: {SystemConfig.VISION_MODEL}")
        self.vlm = ChatOpenAI(
            model=SystemConfig.VISION_MODEL, 
            temperature=0.0, 
            max_tokens=1024
        )

    @staticmethod
    def _resize_for_api(image_path: Path, max_dim: int = MAX_IMG_DIM) -> str:
        """Downscales high-res images to safe limits."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if max(width, height) > max_dim:
                    ratio = max_dim / max(width, height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise

    @traceable(name="Analyze Image", run_type="tool")
    @retry_with_backoff(retries=2, backoff_in_seconds=2)  # <--- RESILIENCE ADDED
    def analyze(self, file_name: str, mode: Literal["chart", "general"] = "general", context: str = "") -> str:
        """
        Main entry point.
        Args:
            context: Optional text (OCR data or Legend Colors) passed from the Parser.
        """
        target_path = SystemPaths.LAYOUTS / file_name
        if not target_path.exists():
            target_path = SystemPaths.SHADOW_CACHE / file_name
        if not target_path.exists():
            return f"Error: Image artifact '{file_name}' not found."

        try:
            b64_image = self._resize_for_api(target_path)
        except Exception as e:
            return f"Error processing image: {e}"

        if mode == "chart":
            return self._analyze_chart_deplot(b64_image, context)
        else:
            return self._analyze_general(b64_image)

    def _analyze_chart_deplot(self, b64_image: str, context: str) -> str:
        """
        [MODE A] Strict Financial De-Plotting.
        Incorporates your 'Binding' and 'Validation' logic.
        """
        # We inject your specific color/OCR context if available
        evidence_block = ""
        if context:
            evidence_block = f"""
            ### EVIDENCE PROVIDED
            {context}
            """

        prompt = f"""
        You are a Financial Data De-Plotter. Reconstruct the underlying data from this chart.
        
        {evidence_block}
        
        ### TASKS
        1. **BINDING (Match Values to Categories):**
           - **Primary (Color):** If values are color-coded, use the visual evidence to group them.
           - **Secondary (Alignment):** Align floating text spatially with bars/lines.
           
        2. **MATHEMATICAL VALIDATION:**
           - **Stacked Bar:** Segments must sum to the approximate total height.
           - **Waterfall:** Logic must be sequential (Start + Steps = End).
           - **General:** Rounding differences (e.g. 19.9 vs 20.0) are acceptable.

        ### OUTPUT
        Return ONLY a Markdown table. Do not add conversational filler.
        | Category | Series/Item | Value |
        |---|---|---|
        """
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        )
        
        response = self.vlm.invoke([message])
        return response.content

    def _analyze_general(self, b64_image: str) -> str:
        """[MODE B] General Description."""
        prompt = """
        Analyze this image for a professional credit underwriting knowledge base.
        
        ### 1. Classification
        Type: (e.g., Diagram, Slide, Screenshot, Flowchart).
        
        ### 2. Transcription
        Transcribe all visible text exactly as it appears.
        
        ### 3. Structural Analysis
        - IF DIAGRAM: Describe the flow steps (A -> B -> C).
        - IF SLIDE: Summarize the key takeaway.
        """
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        )
        
        response = self.vlm.invoke([message])
        return response.content

# Singleton
vision_tool = VisionTool()