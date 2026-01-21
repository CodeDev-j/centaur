# src/utils/colors.py

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
    """
    
    @staticmethod
    def get_luminance(rgb: Tuple[int, int, int]) -> float:
        """Calculates perceived brightness (Standard Rec. 709)."""
        return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])

    @staticmethod
    def get_base_hue(rgb: Tuple[int, int, int]) -> str:
        """
        Robustly maps RGB to a Base Hue Bucket using HSL.
        Prevents 'Dark Red' from being misclassified as 'Black' or 'Brown'.
        """
        r, g, b = rgb
        h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)  # noqa: E741
        
        # Achromatic check (Low Saturation)
        if s < 0.15: 
            return "Gray"
        
        deg = h * 360
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
        Handles dictionary inputs with 'confirmed' flag.
        Input: List of LegendBinding objects
        Output: List of strings "Legend Key: 'Label' == Hex (Semantic Name)"
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
                # No collision: Just use the Base Name
                i = items[0]
                prefix = "[CONFIRMED LEGEND]" if i['confirmed'] else "[POSSIBLE LEGEND]"
                results.append(f"{prefix} '{i['label']}' == {i['hex']} ({base_name})")
            else:
                # Collision: Sort by Luminance (Darkest to Lightest)
                items.sort(key=lambda x: x["lum"])
                
                count = len(items)
                for idx, item in enumerate(items):
                    # Assign relative modifier
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
                        # Fallback for many shades
                        intensity = int((idx / (count - 1)) * 100)
                        mod = f"Luminance-{intensity}"
                    
                    prefix = "[CONFIRMED LEGEND]" if item['confirmed'] else "[POSSIBLE LEGEND]"
                    results.append(f"{prefix} '{item['label']}' == {item['hex']} ({mod} {base_name})")

        return results