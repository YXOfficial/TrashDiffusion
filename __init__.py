from .comfy_support import (
    FDGNodeV2, 
    ZeResFDGNodeV2, 
    CFGZeroNodeV2, 
    S2GuidanceNodeV2
)

# Traditional ComfyUI Node Mappings
# Mapping các class V2 mới
NODE_CLASS_MAPPINGS = {
    "YX_FDG_V2": FDGNodeV2,
    "YX_ZeResFDG_V2": ZeResFDGNodeV2,
    "YX_CFGZero_V2": CFGZeroNodeV2,
    "YX_S2Guidance_V2": S2GuidanceNodeV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YX_FDG_V2": "YX Frequency-Decoupled Guidance (V2)",
    "YX_ZeResFDG_V2": "YX ZeResFDG Guidance (V2)",
    "YX_CFGZero_V2": "YX CFG-Zero Guidance (V2)",
    "YX_S2Guidance_V2": "YX S2-Guidance (V2)",
}

# --- Support for New ComfyUI API (Test-reForge) ---
try:
    from comfy_api.latest import ComfyExtension
    from typing_extensions import override

    class CrazyDiffusionExtension(ComfyExtension):
        @override
        async def get_node_list(self) -> list:
            return [
                FDGNodeV2,
                ZeResFDGNodeV2,
                CFGZeroNodeV2,
                S2GuidanceNodeV2,
            ]

    async def comfy_entrypoint() -> CrazyDiffusionExtension:
        return CrazyDiffusionExtension()
except ImportError:
    pass

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
