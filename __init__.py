"""
SeedVR2 Video Upscaler - Transition progressive vers architecture modulaire

Ce fichier gère la transition entre:
- Ancien code monolithique (seedvr2.py)
- Nouvelle architecture modulaire (src/)

Migration en cours...
"""

# 🆕 TENTATIVE: Nouvelle architecture modulaire
from .src.interfaces.comfyui_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
USING_MODULAR = True


# Export pour ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Métadonnées
__version__ = "1.5.0-transition" if not USING_MODULAR else "2.0.0-modular"
