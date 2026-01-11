from typing import List, Optional, Union
from pydantic import BaseModel, Field, model_validator

class BoundingBox(BaseModel):
    """
    Standardized Bounding Box for Visual Citations.
    
    Dual-Mode Support:
    1. 'box_2d' (Backend/VLM): Integer coordinates [ymin, xmin, ymax, xmax] (0-1000 scale).
    2. 'rect' (Frontend): Float coordinates {x, y, width, height} (0.0-1.0 scale).
    
    The validator automatically hydrates 'rect' from 'box_2d' if missing.
    """
    page_number: int = Field(..., description="1-based page number")
    
    # Primary Format (VLM Output)
    box_2d: Optional[List[int]] = Field(
        None, 
        min_items=4, max_items=4, 
        description="[ymin, xmin, ymax, xmax] on 0-1000 scale"
    )
    
    # Derived Format (Frontend Ready)
    x: float = Field(default=0.0, description="Left X (0-1)")
    y: float = Field(default=0.0, description="Top Y (0-1)")
    width: float = Field(default=0.0, description="Width (0-1)")
    height: float = Field(default=0.0, description="Height (0-1)")
    
    # Audit Trail
    source: str = Field(default="unknown", description="'vlm_estimate' or 'exact_ocr_snap'")

    @model_validator(mode='before')
    @classmethod
    def unify_coordinates(cls, data: dict):
        """
        Auto-converts between VLM integers (0-1000) and Frontend floats (0-1).
        """
        # Case A: Input is VLM style (box_2d)
        if 'box_2d' in data and data['box_2d']:
            # box_2d is typically [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = data['box_2d']
            
            # Normalize to 0-1 floats for Frontend
            x = xmin / 1000.0
            y = ymin / 1000.0
            w = (xmax - xmin) / 1000.0
            h = (ymax - ymin) / 1000.0
            
            data['x'], data['y'] = x, y
            data['width'], data['height'] = w, h
            
        # Case B: Input is Frontend style (x, y, w, h)
        elif 'x' in data and 'width' in data:
            # Back-calculate box_2d for consistency
            x, y = data.get('x', 0), data.get('y', 0)
            w, h = data.get('width', 0), data.get('height', 0)
            
            data['box_2d'] = [
                int(y * 1000),      # ymin
                int(x * 1000),      # xmin
                int((y + h) * 1000),# ymax
                int((x + w) * 1000) # xmax
            ]

        # Sanity Check: Clamp Floats to 0.0-1.0
        for k in ['x', 'y', 'width', 'height']:
            if k in data:
                data[k] = max(0.0, min(1.0, data[k]))

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
    native_id: Optional[str] = Field(None, description="Excel Cell ID (e.g., 'C15')")