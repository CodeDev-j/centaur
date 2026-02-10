"""
Smart Router: The Traffic Controller
====================================

1. THE MISSION
--------------
To strictly determine *how* a file should be processed based on its physical
attributes (file extension). It acts as the "Switchboard Operator" for the
Ingestion Pipeline.

2. THE MECHANISM
----------------
- **Static Mapping:** Uses a deterministic dictionary to map file extensions
  to specific processing streams (Helices).
- **Zero-Touch Logic:** Does NOT open files or read content. It makes fast,
  metadata-based decisions to keep the pipeline performant.

3. THE CONTRACT
---------------
- **Input:** A `pathlib.Path` object pointing to a file.
- **Output:** A tuple containing:
    1. `ProcessingRoute` (Enum): The assigned destination.
    2. `str`: A human-readable reason for the decision.
- **Safety:** Returns `UNSUPPORTED` for unknown extensions rather than crashing.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class ProcessingRoute(Enum):
    """
    Defines the available architectural pathways (Helices) for ingestion.
    """
    # 🎨 Helix A: The Visual Brain
    # Used for "Dead" documents where layout/visuals matter (PDF, Scans).
    HELIX_A_VISUAL = "visual_stream"

    # 🧬 Helix B: The Native Brain
    # Used for "Live" documents with structured metadata (Excel, Word, PPT).
    HELIX_B_NATIVE = "native_stream"

    # ⛔ Unsupported
    # Files that cannot be safely processed.
    UNSUPPORTED = "unsupported"


class SmartRouter:
    """
    The Routing Node.
    Directs files to the correct ingestion helix based on file signature.
    """

    # Central Registry of Supported Extensions
    SUPPORTED_EXTENSIONS: Dict[str, ProcessingRoute] = {
        # --- 🎨 Visual Stream (Docling + VLM) ---
        ".pdf":  ProcessingRoute.HELIX_A_VISUAL,
        ".png":  ProcessingRoute.HELIX_A_VISUAL,
        ".jpg":  ProcessingRoute.HELIX_A_VISUAL,
        ".jpeg": ProcessingRoute.HELIX_A_VISUAL,

        # --- 🧬 Native Stream (MarkItDown / Phantom) ---
        ".xlsx": ProcessingRoute.HELIX_B_NATIVE,
        ".xls":  ProcessingRoute.HELIX_B_NATIVE,
        ".xlsm": ProcessingRoute.HELIX_B_NATIVE,
        ".docx": ProcessingRoute.HELIX_B_NATIVE,
        ".pptx": ProcessingRoute.HELIX_B_NATIVE,
        ".md":   ProcessingRoute.HELIX_B_NATIVE,
        ".txt":  ProcessingRoute.HELIX_B_NATIVE,
    }

    @staticmethod
    def route(file_path: Path) -> Tuple[ProcessingRoute, str]:
        """
        Determines the processing route for a given file.

        Args:
            file_path (Path): The file to route.

        Returns:
            Tuple[ProcessingRoute, str]: (The Route Enum, Reason String)
        """
        if not file_path.exists():
            return ProcessingRoute.UNSUPPORTED, "File does not exist"

        # 1. Normalize Extension (Lowercase for consistency)
        ext = file_path.suffix.lower()

        # 2. Check Registry
        if ext not in SmartRouter.SUPPORTED_EXTENSIONS:
            return ProcessingRoute.UNSUPPORTED, f"Extension '{ext}' not supported"

        # 3. Assign Route
        route = SmartRouter.SUPPORTED_EXTENSIONS[ext]

        logger.info(f"🚦 Routing {file_path.name} -> {route.value}")
        return route, "Extension match"