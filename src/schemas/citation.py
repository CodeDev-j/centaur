"""
Schema definitions for Citations and Visual Grounding.
These objects form the "Trust Layer," allowing users to verify AI claims
against the original source document coordinates.
"""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    """
    Normalized coordinates (0.0 to 1.0) for visual highlighting.
    Includes 'Smart Validation' to auto-correct inverted coordinates
    and ensure safety for the Frontend Viewer.
    """
    page_number: int
    x: float = Field(..., description="Left X coordinate (normalized 0-1)")
    y: float = Field(..., description="Top Y coordinate (normalized 0-1)")
    width: float = Field(..., description="Width of the box (normalized 0-1)")
    height: float = Field(..., description="Height of the box (normalized 0-1)")

    @model_validator(mode='before')
    @classmethod
    def sanitize_coordinates(cls, data: dict):
        """
        The 'Single Source of Truth' for coordinate logic.
        1. Auto-corrects transposed coordinates (if negative width/height).
        2. Clamps values to 0.0-1.0 to prevent viewer crashes.
        """
        # Extract raw inputs with defaults to prevent KeyErrors
        x = data.get('x', 0)
        y = data.get('y', 0)
        w = data.get('width', 0)
        h = data.get('height', 0)

        # 1. Handle Transposed Coordinates (Negative Width/Height)
        # If width is negative, it means x was actually x_right, not x_left.
        if w < 0:
            x = x + w  # Shift x back to the real left
            w = abs(w)  # Make width positive

        if h < 0:
            y = y + h  # Shift y back to the real top
            h = abs(h)  # Make height positive

        # 2. Clamp to Unit Square (0.0 to 1.0)
        # Prevents "drawing off the canvas" errors in the frontend
        def clamp(val):
            return max(0.0, min(1.0, float(val)))

        data['x'] = clamp(x)
        data['y'] = clamp(y)
        data['width'] = clamp(w)
        data['height'] = clamp(h)

        return data


class Citation(BaseModel):
    """
    The 'Trust Artifact'. Every answer must allow the user to click
    and see exactly where the data came from.
    """
    source_file: str
    page_number: int
    blurb: str = Field(..., description="The specific text snippet quoted")
    bbox: Optional[BoundingBox] = None
    native_id: Optional[str] = Field(
        None,
        description="Excel Cell ID (e.g., 'C15') or HTML Element ID."
    )