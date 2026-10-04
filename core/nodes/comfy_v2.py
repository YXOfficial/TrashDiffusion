import logging
import torch
try:
    from core.guidance import (
        ensure_guidance_pipeline,
        make_fdg_modifier,
        make_cfg_zero_base_builder,
        get_initial_sigma,
    )
except ImportError:  # pragma: no cover
    from ..guidance import (
        ensure_guidance_pipeline,
        make_fdg_modifier,
        make_cfg_zero_base_builder,
        get_initial_sigma,
    )

# Exact import from your ComfyUI repo
try:
    from comfy_api.latest import io
    # Must inherit io.ComfyNode to auto-generate INPUT_TYPES, RETURN_TYPES, etc.
    BaseNode = io.ComfyNode
except ImportError:
    # Without API V2, use an empty class with manual definitions (Standard build)
    io = None
    BaseNode = object

class FDGNodeV2(BaseNode):
    @classmethod
    def define_schema(cls) -> "io.Schema":
        return io.Schema(
            node_id="YX_FDG_V2",
            display_name="YX Frequency-Decoupled Guidance (V2)",
            category="advanced/model_patches",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("enabled", options=["true", "false"], default="true"),
                io.Float.Input("w_low", default=1.0, min=0.0, max=10.0, step=0.1),
                io.Float.Input("w_high", default=1.0, min=0.0, max=10.0, step=0.1),
                io.Int.Input("levels", default=3, min=2, max=8, step=1),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enabled, w_low, w_high, levels) -> "io.NodeOutput":
        if enabled == "false": return io.NodeOutput(model)
        patched_model = model.clone()
        pipeline = ensure_guidance_pipeline(patched_model)
        pipeline.add_modifier("fdg", make_fdg_modifier(w_low, w_high, int(levels)))
        return io.NodeOutput(patched_model)

    # Fallback for standard ComfyUI without API V2
    if io is None:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model": ("MODEL",),
                    "enabled": (["true", "false"], {"default": "true"}),
                    "w_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                    "w_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                    "levels": ("INT", {"default": 3, "min": 2, "max": 8, "step": 1}),
                }
            }
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "patch"
        CATEGORY = "advanced/model_patches"
        def patch(self, **kwargs):
            res = self.execute(**kwargs)
            return (res[0],)

class CFGZeroNodeV2(BaseNode):
    @classmethod
    def define_schema(cls) -> "io.Schema":
        return io.Schema(
            node_id="YX_CFGZero_V2",
            display_name="YX CFG-Zero Guidance (V2)",
            category="advanced/model_patches",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("enabled", options=["true", "false"], default="true"),
                io.Combo.Input("zero_init_first_step", options=["true", "false"], default="false"),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enabled, zero_init_first_step) -> "io.NodeOutput":
        if enabled == "false": return io.NodeOutput(model)
        patched_model = model.clone()
        pipeline = ensure_guidance_pipeline(patched_model)
        initial_sigma = get_initial_sigma(patched_model)
        pipeline.set_base_builder(make_cfg_zero_base_builder((zero_init_first_step == "true"), initial_sigma))
        return io.NodeOutput(patched_model)

    if io is None:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model": ("MODEL",),
                    "enabled": (["true", "false"], {"default": "true"}),
                    "zero_init_first_step": (["true", "false"], {"default": "false"}),
                }
            }
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "patch"
        CATEGORY = "advanced/model_patches"
        def patch(self, **kwargs):
            res = self.execute(**kwargs)
            return (res[0],)

class ASAGNodeV2(BaseNode):
    @classmethod
    def define_schema(cls) -> "io.Schema":
        return io.Schema(
            node_id="YX_ASAG_V2",
            display_name="YX ASAG Guidance (V2)",
            category="advanced/model_patches",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("scale", default=1.5, min=0.0, max=100.0, step=0.1),
                io.Int.Input("sinkhorn_iters", default=2, min=1, max=8),
                io.Combo.Input("unet_block", options=["input", "middle", "output"], default="middle"),
                io.Int.Input("unet_block_id", default=0),
                io.Float.Input("sigma_start", default=-1.0),
                io.Float.Input("sigma_end", default=-1.0),
                io.Float.Input("rescale", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("rescale_mode", options=["full", "partial", "snf"], default="full"),
                io.String.Input("unet_block_list", default=""),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, **kwargs) -> "io.NodeOutput":
        try:
            from core.nodes.asag_node import ASAGGuidance
        except ImportError:  # pragma: no cover
            from .asag_node import ASAGGuidance
        model = kwargs.pop("model")
        # Convert parameter names if needed
        res = ASAGGuidance().patch(model, **kwargs)
        if io: return io.NodeOutput(res[0])
        return res

    if io is None:
        @classmethod
        def INPUT_TYPES(cls):
            try:
                from core.nodes.asag_node import ASAGGuidance
            except ImportError:  # pragma: no cover
                from .asag_node import ASAGGuidance
            return ASAGGuidance.INPUT_TYPES()
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "patch"
        CATEGORY = "advanced/model_patches"
        def patch(self, **kwargs):
            res = self.execute(**kwargs)
            return (res[0],)
