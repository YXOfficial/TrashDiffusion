"""Backend-agnostic ComfyUI/Forge node definitions.

Thin wrappers around :mod:`core.guidance`. No Forge/Comfy imports here;
``comfy_v2.py`` (API v2, optional) is the only file allowed to touch
``comfy_api`` and it does so lazily.
"""

from .asag_node import ASAGGuidance
from .cfg_zero import CFGZeroNode
from .fdg import FDGNode

__all__ = [
    "ASAGGuidance",
    "CFGZeroNode",
    "FDGNode",
]
