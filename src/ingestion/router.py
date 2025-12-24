import logging
from pathlib import Path
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)

class ProcessingRoute(Enum):
    HELIX_A_VISUAL = "visual_stream"  # PDF, Scanned Images (Docling)
    HELIX_B_NATIVE = "native_stream"  # Excel, Word, PPT (MarkItDown + Phantom)
    UNSUPPORTED = "unsupported"

class SmartRouter:
    """
    The 'Smart Router' Node.
    Directs files to the correct ingestion helix based on file signature.
    """

    SUPPORTED_EXTENSIONS = {
        # Visual Stream (Docling)
        ".pdf": ProcessingRoute.HELIX_A_VISUAL,
        ".png": ProcessingRoute.HELIX_A_VISUAL,
        ".jpg": ProcessingRoute.HELIX_A_VISUAL,
        ".jpeg": ProcessingRoute.HELIX_A_VISUAL,
        
        # Native Stream (MarkItDown / Phantom)
        ".xlsx": ProcessingRoute.HELIX_B_NATIVE,
        ".xls": ProcessingRoute.HELIX_B_NATIVE,
        ".xlsm": ProcessingRoute.HELIX_B_NATIVE,
        ".docx": ProcessingRoute.HELIX_B_NATIVE,
        ".pptx": ProcessingRoute.HELIX_B_NATIVE,
        ".md": ProcessingRoute.HELIX_B_NATIVE,
        ".txt": ProcessingRoute.HELIX_B_NATIVE,
    }

    @staticmethod
    def route(file_path: Path) -> Tuple[ProcessingRoute, str]:
        """
        Determines the processing route for a given file.
        Returns: (Route, Reason)
        """
        if not file_path.exists():
            return ProcessingRoute.UNSUPPORTED, "File does not exist"

        # 1. Check Extension
        ext = file_path.suffix.lower()
        if ext not in SmartRouter.SUPPORTED_EXTENSIONS:
            return ProcessingRoute.UNSUPPORTED, f"Extension {ext} not supported"

        # 2. Determine Route
        route = SmartRouter.SUPPORTED_EXTENSIONS[ext]
        
        logger.info(f"🚦 Routing {file_path.name} -> {route.value}")
        return route, "Extension match"