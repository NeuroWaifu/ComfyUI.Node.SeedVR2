"""
Interfaces package for SeedVR2
Contains user interface integrations (ComfyUI, etc.)
"""

# ComfyUI Interfaces Module
# Handles ComfyUI node integration and user interface

from .comfyui_node import (
    # Main ComfyUI node class
    SeedVR2,
    
    # ComfyUI mappings
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,

)

__all__ = [
    # Core node interface
    'SeedVR2',
    
    # ComfyUI mappings
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    
] 