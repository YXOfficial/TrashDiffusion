try:
    from .nodes_fdg import NODE_CLASS_MAPPINGS as FDG_CLASS, NODE_DISPLAY_NAME_MAPPINGS as FDG_DISPLAY
    from .nodes_zeresfdg import NODE_CLASS_MAPPINGS as ZERES_CLASS, NODE_DISPLAY_NAME_MAPPINGS as ZERES_DISPLAY
    from .nodes_cfg_zero import NODE_CLASS_MAPPINGS as CFG_CLASS, NODE_DISPLAY_NAME_MAPPINGS as CFG_DISPLAY
    from .nodes_s2_guidance import NODE_CLASS_MAPPINGS as S2_CLASS, NODE_DISPLAY_NAME_MAPPINGS as S2_DISPLAY
except ImportError:
    FDG_CLASS, FDG_DISPLAY = {}, {}
    ZERES_CLASS, ZERES_DISPLAY = {}, {}
    CFG_CLASS, CFG_DISPLAY = {}, {}
    S2_CLASS, S2_DISPLAY = {}, {}

# Traditional ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    **FDG_CLASS,
    **ZERES_CLASS,
    **CFG_CLASS,
    **S2_CLASS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **FDG_DISPLAY,
    **ZERES_DISPLAY,
    **CFG_DISPLAY,
    **S2_DISPLAY,
}

# --- Support for New ComfyUI API (V2 / Test-reForge) ---
try:
    from comfy_api.latest import ComfyExtension
    from typing_extensions import override

    class CrazyDiffusionExtension(ComfyExtension):
        @override
        async def get_node_list(self) -> list:
            # Traditional nodes are already handled by NODE_CLASS_MAPPINGS
            return []

    async def comfy_entrypoint() -> CrazyDiffusionExtension:
        return CrazyDiffusionExtension()
except ImportError:
    pass

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
