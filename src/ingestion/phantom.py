import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from playwright.async_api import async_playwright
from langsmith import traceable

from src.config import SystemPaths

logger = logging.getLogger(__name__)

class PhantomRenderer:
    """
    The 'Phantom' Engine.
    Converts native office documents (Excel) into 'Shadow PDFs'.
    This allows the frontend to render visual highlights for non-visual files.
    """

    def __init__(self):
        # Ensure the shadow cache directory exists
        self.cache_dir = SystemPaths.SHADOW_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @traceable(name="Render Shadow PDF", run_type="tool")
    async def render_excel_to_pdf(self, file_path: Path) -> Optional[str]:
        """
        Converts an Excel file to a PDF via HTML intermediate.
        Returns the relative path to the generated PDF.
        """
        if not file_path.exists():
            logger.error(f"Cannot render missing file: {file_path}")
            return None

        # 1. Define Output Path
        # We use the same filename but change extension to .pdf
        output_filename = file_path.stem + "_shadow.pdf"
        output_path = self.cache_dir / output_filename

        # Cache Hit Check: Don't re-render if it exists
        if output_path.exists():
            logger.info(f"👻 Phantom Cache Hit: {output_filename}")
            return f"shadow_cache/{output_filename}"

        logger.info(f"👻 Phantom Rendering: {file_path.name} -> PDF")

        try:
            # 2. Excel -> HTML (Using Pandas)
            # We treat the first sheet as the primary view for now
            df = pd.read_excel(file_path, sheet_name=0)
            
            # Generate a clean HTML table with borders
            html_content = df.to_html(index=False, border=1, classes="table table-striped")
            
            # Wrap in a basic HTML structure for better styling
            full_html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; padding: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h2>{file_path.name} (Shadow View)</h2>
                {html_content}
            </body>
            </html>
            """

            # 3. HTML -> PDF (Using Playwright)
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                # Load the HTML content directly
                await page.set_content(full_html)
                
                # Print to PDF
                await page.pdf(path=output_path, format="A4", margin={"top": "1cm", "bottom": "1cm"})
                await browser.close()

            logger.info(f"✅ Shadow PDF created: {output_filename}")
            return f"shadow_cache/{output_filename}"

        except Exception as e:
            logger.error(f"❌ Phantom Render Failed for {file_path.name}: {e}")
            # Non-blocking failure: We return None, and the pipeline continues without visual grounding
            return None