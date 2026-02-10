"""
src/utils/geometry.py

Spatial Indexing for Vector Graphics (The "Geometry" Phase).

GOAL: This module optimizes 2D geometric queries. It allows the pipeline to 
instantaneously answer questions like "What vector shapes are near this text?" 
without iterating through thousands of SVG paths on the page.

ARCHITECTURAL NOTE:
- Spatial Hashing: We map the continuous 2D page space into discrete grid buckets.
- Performance: Transforms geometric search from O(N) linear scans into O(1) 
  constant-time lookups (average case). Crucial for handling dense charts 
  where a single page may contain 5,000+ vector elements.
"""

import collections
import math
from typing import Any, Dict, Generator, List, Tuple

import fitz  # PyMuPDF


class SpatialIndex:
    """
    A lightweight spatial hashing grid for O(1) 2D lookups.
    
    Usage:
    1. Initialize with page dimensions.
    2. Insert all vector drawings (shapes).
    3. Query with a bounding box (e.g., a text label's rect) to find 
       nearby visual elements.
    """
    
    def __init__(self, page_rect: fitz.Rect, cell_size: int = 50):
        """
        Initializes the spatial grid.
        
        Args:
            page_rect: The dimensions of the PDF page.
            cell_size: The bucket size in pixels. 50px is used for 
                       standard chart legends and axis labels.
        """
        self.cell_size = cell_size
        self.cols = math.ceil(page_rect.width / cell_size)
        self.rows = math.ceil(page_rect.height / cell_size)
        
        # Hash Map: (col, row) -> List[Shape Dict]
        self.grid = collections.defaultdict(list)

    def _get_keys(self, rect: fitz.Rect) -> Generator[Tuple[int, int], None, None]:
        """
        Maps a continuous Rectangle to a set of discrete Grid Keys (Cells).
        Handles shapes that span multiple grid cells.
        """
        start_col = max(0, int(rect.x0 // self.cell_size))
        end_col = min(self.cols, int(rect.x1 // self.cell_size) + 1)
        
        start_row = max(0, int(rect.y0 // self.cell_size))
        end_row = min(self.rows, int(rect.y1 // self.cell_size) + 1)
        
        for c in range(start_col, end_col):
            for r in range(start_row, end_row):
                yield (c, r)

    def insert(self, shape: Dict[str, Any]) -> None:
        """
        Inserts a shape (from PyMuPDF `get_drawings`) into the grid.
        Populates all grid cells that the shape's bounding box touches.
        """
        # Ensure we have a fitz.Rect object (PyMuPDF sometimes returns tuples)
        rect = fitz.Rect(shape['rect'])
        for key in self._get_keys(rect):
            self.grid[key].append(shape)

    def query(self, rect: fitz.Rect) -> List[Dict[str, Any]]:
        """
        Retrieves all shapes that might intersect with the query rectangle.
        
        Mechanism:
        1. Identify which grid cells the query rect touches.
        2. Retrieve all shapes stored in those cells.
        3. De-duplicate results (since a shape often lives in multiple cells).
        
        Returns:
            List[Dict[str, Any]]: Candidate shapes for precise intersection checks.
        """
        candidates = set()
        results = []
        
        for key in self._get_keys(rect):
            for shape in self.grid[key]:
                # Use object ID for fast de-duplication
                s_id = id(shape)
                if s_id not in candidates:
                    candidates.add(s_id)
                    results.append(shape)
                    
        return results