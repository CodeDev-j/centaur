"""
Color Resolution Logic for Vector Graphics (The "Semantic" Phase).

GOAL: This module bridges the gap between raw RGB pixels and human-readable 
color names. It solves the "Three Shades of Blue" problem common in financial 
charts by converting colors into semantic descriptions (e.g., "Dark Blue" 
vs. "Light Blue") for the VLM.

ARCHITECTURAL NOTE:
- HSL Mapping: We convert RGB to HSL to determine the 'Base Hue' (Red, Blue, etc.).
- Luminance Sorting: When multiple series share a base hue (collisions), we 
  sort them by perceived brightness (Rec. 709 standard) to generate relative 
  modifiers (Dark, Medium, Light).
"""

import collections
import colorsys
from typing import Tuple, List, TypedDict

# Using TypedDict for strict data structure enforcement
class LegendBinding(TypedDict):
    text: str
    rgb: Tuple[int, int, int]
    hex: str
    confirmed: bool


class ColorResolver:
    """
    Resolves ambiguous colors (e.g., three shades of red) into semantic names
    like 'Dark Red', 'Medium Red', 'Light Red'.
    Used by `pdf_parser.py` to generate the [VECTOR LEGEND HINTS] block.
    """
    
    @staticmethod
    def get_luminance(rgb: Tuple[int, int, int]) -> float:
        """
        Calculates perceived brightness using the Standard Rec. 709 formula.
        
        Why Rec. 709? Humans are more sensitive to Green than Blue. 
        Raw averaging ((R+G+B)/3) is inaccurate for sorting visual contrast.
        Formula: 0.299*R + 0.587*G + 0.114*B
        """
        return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])

    @staticmethod
    def get_base_hue(rgb: Tuple[int, int, int]) -> str:
        """
        Robustly maps RGB to a Base Hue Bucket using HSL Cylinders.
        Prevents 'Dark Red' from being misclassified as 'Black' or 'Brown'.
        
        Logic:
        1. Convert RGB to HSL (Hue, Saturation, Lightness).
        2. Filter Achromatic colors (Gray/Black/White) via Saturation < 0.15.
        3. Map Hue Degree (0-360) to primary color buckets.
        
        Returns:
            str: 'Red', 'Blue', 'Gray', etc.
        """
        r, g, b = rgb
        # Normalize 0-255 to 0.0-1.0 for Python's colorsys
        h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)  # noqa: E741
        
        # Achromatic Check: If Saturation is low, it's Gray regardless of Hue.
        if s < 0.15: 
            return "Gray"
        
        deg = h * 360
        
        # Hue Buckets (Degrees)
        if deg < 15 or deg > 345:
            return "Red"
        if deg < 45:
            return "Orange"
        if deg < 70:
            return "Yellow"
        if deg < 150:
            return "Green"
        if deg < 200:
            return "Cyan"
        if deg < 260:
            return "Blue"
        if deg < 300:
            return "Purple"
        return "Pink"

    @classmethod
    def resolve_names(cls, bindings: List[LegendBinding]) -> List[str]:
        """
        Resolves color collisions (e.g., two distinct Blues) into unique names.
        
        Algorithm:
        1. Group all bindings by their 'Base Hue' (Red, Blue, etc.).
        2. If a group has >1 item, sort them by Luminance (Darkest to Lightest).
        3. Apply relative modifiers ('Dark', 'Medium', 'Light') based on rank.
        
        Returns:
            List[str]: Formatted strings ready for VLM injection.
            Example: "[CONFIRMED LEGEND] 'Revenue' == #0000FF (Dark Blue)"
        """
        # 1. Group by Base Hue
        groups = collections.defaultdict(list)
        for item in bindings:
            base = cls.get_base_hue(item['rgb'])
            groups[base].append({
                "label": item['text'], 
                "rgb": item['rgb'], 
                "hex": item['hex'], 
                "lum": cls.get_luminance(item['rgb']),
                "confirmed": item['confirmed']
            })

        results = []

        # 2. Process each group to resolve collisions
        for base_name, items in groups.items():
            if len(items) == 1:
                # Case A: No collision (e.g., only one Red line).
                i = items[0]
                prefix = "[CONFIRMED LEGEND]" if i['confirmed'] else "[POSSIBLE LEGEND]"
                results.append(f"{prefix} '{i['label']}' == {i['hex']} ({base_name})")
            else:
                # Case B: Collision (e.g., three Blue bars).
                # Sort by Luminance: Low (Dark) -> High (Light)
                items.sort(key=lambda x: x["lum"])
                
                count = len(items)
                for idx, item in enumerate(items):
                    # Assign relative modifier based on sort order
                    if count == 2:
                        mod = "Dark" if idx == 0 else "Light"
                    elif count == 3:
                        if idx == 0:
                            mod = "Dark"
                        elif idx == 1:
                            mod = "Medium"
                        else:
                            mod = "Light"
                    else:
                        # Fallback for complex charts (e.g. 5 shades of Blue)
                        intensity = int((idx / (count - 1)) * 100)
                        mod = f"Luminance-{intensity}"
                    
                    prefix = "[CONFIRMED LEGEND]" if item['confirmed'] else "[POSSIBLE LEGEND]"
                    results.append(f"{prefix} '{item['label']}' == {item['hex']} ({mod} {base_name})")

        return results