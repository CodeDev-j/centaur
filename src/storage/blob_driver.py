"""
Blob Driver: Local File System Artifact Management
==================================================

1. THE MISSION
--------------
This module serves as the "Content Truth" abstraction layer. It manages the 
physical storage and retrieval of ingestion artifacts (JSON, Markdown) on the 
local file system.

2. THE MECHANISM
----------------
- **Storage:** Uses asynchronous I/O (`aiofiles`) for non-blocking writes during 
  heavy ingestion pipelines.
- **Retrieval:** Uses synchronous I/O for immediate access during retrieval phases.
- **Observability:** Intentionally NOT traced to prevent "Payload Explosion" 
  (uploading massive file contents to LangSmith).

3. THE CONTRACT
---------------
- **Inputs:** Strict typing for data and folder categories.
- **Outputs:** Returns relative paths (strings) for database storage.
- **Safety:** Enforces allowed directory paths via `SystemPaths`.
"""

import json
import logging
from typing import Any, Dict

import aiofiles

from src.config import SystemPaths

logger = logging.getLogger(__name__)


class BlobDriver:
    """
    Abstraction layer for file system operations.
    Maps logical artifact types to physical storage locations.
    """

    @staticmethod
    async def save_json(data: Dict[str, Any], folder: str, filename: str) -> str:
        """
        Asynchronously saves a dictionary as a formatted JSON artifact.

        Args:
            data (Dict[str, Any]): The data to serialize.
            folder (str): The logical folder type ('layouts', 'definitions', 'tables').
            filename (str): The target filename (e.g., 'doc_123.json').

        Returns:
            str: The relative path to the saved artifact (e.g., 'layouts/doc_123.json').

        Raises:
            ValueError: If the folder type is invalid.
            IOError: If the write operation fails.
        """
        # 1. Path Resolution (Strict Whitelist)
        if folder == "layouts":
            base_dir = SystemPaths.LAYOUTS
        elif folder == "definitions":
            base_dir = SystemPaths.DEFINITIONS
        elif folder == "tables":
            base_dir = SystemPaths.TABLES
        else:
            raise ValueError(f"Invalid blob folder category: '{folder}'")

        target_path = base_dir / filename

        # 2. Async Write Execution
        try:
            # Ensure the directory exists (redundancy check)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(target_path, mode='w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2))

            logger.info(f"✅ Saved JSON artifact: {folder}/{filename}")
            return f"{folder}/{filename}"

        except Exception as e:
            logger.error(f"❌ Failed to save JSON blob '{filename}': {e}")
            raise

    @staticmethod
    async def save_markdown(content: str, folder: str, filename: str) -> str:
        """
        Asynchronously saves a string as a Markdown artifact.

        Args:
            content (str): The text content to save.
            folder (str): The logical folder type (e.g., 'tables').
            filename (str): The target filename.

        Returns:
            str: The relative path to the saved artifact.
        """
        # 1. Path Resolution
        if folder == "tables":
            base_dir = SystemPaths.TABLES
        else:
            raise ValueError(f"Invalid blob folder category: '{folder}'")

        target_path = base_dir / filename

        # 2. Async Write Execution
        try:
            # Ensure the directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(target_path, mode='w', encoding='utf-8') as f:
                await f.write(content)

            logger.info(f"✅ Saved Markdown artifact: {folder}/{filename}")
            return f"{folder}/{filename}"

        except Exception as e:
            logger.error(f"❌ Failed to save Markdown blob '{filename}': {e}")
            raise

    @staticmethod
    def load_json(blob_path: str) -> Dict[str, Any]:
        """
        Synchronously loads a JSON artifact.

        Args:
            blob_path (str): Relative path format, e.g., "layouts/chunk_123.json".

        Returns:
            Dict[str, Any]: The parsed JSON data.

        Raises:
            FileNotFoundError: If the artifact does not exist.
        """
        
        full_path = SystemPaths.BLOBS / blob_path

        if not full_path.exists():
            logger.warning(f"⚠️ Artifact not found: {full_path}")
            raise FileNotFoundError(f"Blob not found: {full_path}")

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Corrupt JSON in artifact '{blob_path}': {e}")
            raise

    @staticmethod
    def load_markdown(blob_path: str) -> str:
        """
        Synchronously loads a Markdown artifact.

        Args:
            blob_path (str): Relative path format.

        Returns:
            str: The raw file content.
        """

        full_path = SystemPaths.BLOBS / blob_path

        if not full_path.exists():
            logger.warning(f"⚠️ Artifact not found: {full_path}")
            raise FileNotFoundError(f"Blob not found: {full_path}")

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"❌ Failed to read artifact '{blob_path}': {e}")
            raise