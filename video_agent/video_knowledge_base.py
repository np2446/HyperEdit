"""
Knowledge base for video processing operations.
Stores information about video processing capabilities, effects, and GPU resources.
"""

import json
import os
from typing import Dict, List, Optional

class VideoKnowledgeBase:
    """Knowledge base for video processing operations."""
    
    def __init__(self):
        """Initialize the video knowledge base."""
        self.effects_info = {
            "blur": {
                "description": "Gaussian blur effect",
                "parameters": ["strength"],
                "gpu_requirements": {"min_vram_gb": 2.0}
            },
            "sharpen": {
                "description": "Sharpening effect",
                "parameters": ["strength"],
                "gpu_requirements": {"min_vram_gb": 2.0}
            },
            "color_adjust": {
                "description": "Color adjustment effect",
                "parameters": ["contrast", "saturation", "brightness"],
                "gpu_requirements": {"min_vram_gb": 2.0}
            }
        }
        
        self.transition_info = {
            "fade": {
                "description": "Fade transition between scenes",
                "parameters": ["duration"],
                "gpu_requirements": {"min_vram_gb": 4.0}
            },
            "dissolve": {
                "description": "Dissolve transition between scenes",
                "parameters": ["duration"],
                "gpu_requirements": {"min_vram_gb": 4.0}
            }
        }
        
        self.composition_info = {
            "split_screen": {
                "description": "Side-by-side video comparison",
                "parameters": ["layout"],
                "gpu_requirements": {"min_vram_gb": 6.0}
            },
            "picture_in_picture": {
                "description": "Picture-in-picture effect",
                "parameters": ["position", "size"],
                "gpu_requirements": {"min_vram_gb": 6.0}
            }
        }
    
    def get_effect_info(self, effect_name: str) -> Optional[Dict]:
        """Get information about a specific video effect."""
        return self.effects_info.get(effect_name)
    
    def get_transition_info(self, transition_name: str) -> Optional[Dict]:
        """Get information about a specific transition type."""
        return self.transition_info.get(transition_name)
    
    def get_composition_info(self, composition_name: str) -> Optional[Dict]:
        """Get information about a specific composition type."""
        return self.composition_info.get(composition_name)
    
    def estimate_gpu_requirements(self, effects: List[str], transitions: List[str], compositions: List[str]) -> Dict:
        """Estimate GPU requirements for a combination of effects."""
        max_vram = 2.0  # Base requirement
        
        for effect in effects:
            info = self.get_effect_info(effect)
            if info:
                max_vram = max(max_vram, info["gpu_requirements"]["min_vram_gb"])
        
        for transition in transitions:
            info = self.get_transition_info(transition)
            if info:
                max_vram = max(max_vram, info["gpu_requirements"]["min_vram_gb"])
        
        for composition in compositions:
            info = self.get_composition_info(composition)
            if info:
                max_vram = max(max_vram, info["gpu_requirements"]["min_vram_gb"])
        
        return {
            "min_vram_gb": max_vram,
            "gpu_count": 1  # For now, assume single GPU
        }
    
    def get_supported_effects(self) -> List[str]:
        """Get list of all supported video effects."""
        return list(self.effects_info.keys())
    
    def get_supported_transitions(self) -> List[str]:
        """Get list of all supported transitions."""
        return list(self.transition_info.keys())
    
    def get_supported_compositions(self) -> List[str]:
        """Get list of all supported composition types."""
        return list(self.composition_info.keys()) 