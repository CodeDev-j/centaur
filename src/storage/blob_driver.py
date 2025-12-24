import json
import aiofiles
from typing import Dict, Any
import logging
from langsmith import traceable

from src.config import SystemPaths

logger = logging.getLogger(__name__)

class BlobDriver:
    """
    Abstraction layer for 'Content Truth' storage.
    Currently maps to local file system (data/blobs).
    Future: Will map to Azure Blob Storage SDK.
    """

    @staticmethod
    @traceable(name="Save JSON Blob", run_type="tool")
    async def save_json(data: Dict[str, Any], folder: str, filename: str) -> str:
        """
        Saves a JSON artifact.
        
        Allowed Folders:
        - layouts: Coordinate maps of documents (PDFParser).
        - tables: Structured HTML/JSON of complex tables (PDFParser).
        - definitions: Extracted legal terms (TermInjector).
        """
        # Determine strict path based on folder type
        if folder == "layouts":
            base_dir = SystemPaths.LAYOUTS
        elif folder == "definitions":
            base_dir = SystemPaths.DEFINITIONS
        elif folder == "tables":  # <--- CRITICAL FIX
            base_dir = SystemPaths.TABLES
        else:
            raise ValueError(f"Invalid blob folder: {folder}")

        target_path = base_dir / filename
        
        try:
            async with aiofiles.open(target_path, mode='w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2))
            
            logger.info(f"Saved artifact: {folder}/{filename}")
            return f"{folder}/{filename}"
        except Exception as e:
            logger.error(f"Failed to save JSON blob {filename}: {e}")
            raise

    @staticmethod
    @traceable(name="Save Markdown Blob", run_type="tool")
    async def save_markdown(content: str, folder: str, filename: str) -> str:
        """
        Saves a Markdown artifact (e.g., Full Table).
        """
        if folder == "tables":
            base_dir = SystemPaths.TABLES
        else:
            raise ValueError(f"Invalid blob folder: {folder}")

        target_path = base_dir / filename
        
        try:
            async with aiofiles.open(target_path, mode='w', encoding='utf-8') as f:
                await f.write(content)
            
            logger.info(f"Saved artifact: {folder}/{filename}")
            return f"{folder}/{filename}"
        except Exception as e:
            logger.error(f"Failed to save Markdown blob {filename}: {e}")
            raise

    @staticmethod
    @traceable(name="Load JSON Blob", run_type="tool")
    def load_json(blob_path: str) -> Dict[str, Any]:
        """
        Retrieves a JSON artifact.
        blob_path format: "layouts/chunk_123.json"
        """
        # Construct full local path
        full_path = SystemPaths.ARTIFACTS / blob_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Blob not found: {full_path}")
            
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    @traceable(name="Load Markdown Blob", run_type="tool")
    def load_markdown(blob_path: str) -> str:
        """
        Retrieves a Markdown artifact.
        """
        full_path = SystemPaths.ARTIFACTS / blob_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Blob not found: {full_path}")
            
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()