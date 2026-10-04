"""TrashDiffusion core: pure, backend-agnostic guidance math.

Rules for everything under ``core/``:
* torch-only. NEVER import ``modules``, ``backend``, ``comfy``,
  ``gradio`` or ``ldm_patched`` at module top level.
* Forge/Comfy specific access (``p.sd_model...``, ``devices``,
  ``calc_cond_uncond_batch``) lives in :mod:`core.forge_compat` and is
  resolved lazily with fallbacks, so an upstream restructure degrades
  to a warning instead of an import-time crash.
* Frontends (``sd-webui-forge-classic/``, ``ComfyUI/``) are thin
  adapters that import from here.
"""

from .guidance import (
    CFGCtrlState,
    GuidancePipeline,
    GuidanceState,
    default_base_builder,
    ensure_guidance_pipeline,
    get_initial_sigma,
    make_cfg_ctrl_base_builder,
    make_cfg_ctrl_modifier,
    make_cfg_zero_base_builder,
    make_fdg_modifier,
)

__all__ = [
    "CFGCtrlState",
    "GuidancePipeline",
    "GuidanceState",
    "default_base_builder",
    "ensure_guidance_pipeline",
    "get_initial_sigma",
    "make_cfg_ctrl_base_builder",
    "make_cfg_ctrl_modifier",
    "make_cfg_zero_base_builder",
    "make_fdg_modifier",
]
