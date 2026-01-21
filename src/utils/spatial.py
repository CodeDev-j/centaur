# src/utils/spatial.py
import collections
import fitz  # PyMuPDF
import math

from typing import Generator, List, Tuple

class SpatialIndex:
    """
    A lightweight spatial hashing grid for O(1) 2D lookups.
    Used to quickly find vector shapes near text elements.
    """
    def __init__(self, page_rect: fitz.Rect, cell_size: int = 50):
        self.cell_size = cell_size
        self.cols = math.ceil(page_rect.width / cell_size)
        self.rows = math.ceil(page_rect.height / cell_size)
        self.grid = collections.defaultdict(list)

    def _get_keys(self, rect: fitz.Rect) -> Generator[Tuple[int, int], None, None]:
        start_col = max(0, int(rect.x0 // self.cell_size))
        end_col = min(self.cols, int(rect.x1 // self.cell_size) + 1)
        start_row = max(0, int(rect.y0 // self.cell_size))
        end_row = min(self.rows, int(rect.y1 // self.cell_size) + 1)
        for c in range(start_col, end_col):
            for r in range(start_row, end_row):
                yield (c, r)

    def insert(self, shape: dict) -> None:
        """Inserts a shape (from PyMuPDF get_drawings) into the grid."""
        rect = fitz.Rect(shape['rect'])
        for key in self._get_keys(rect):
            self.grid[key].append(shape)

    def query(self, rect: fitz.Rect) -> List[dict]:
        """Returns all shapes that might intersect with the query rect."""
        candidates = set()
        results = []
        for key in self._get_keys(rect):
            for shape in self.grid[key]:
                s_id = id(shape)
                if s_id not in candidates:
                    candidates.add(s_id)
                    results.append(shape)
        return results