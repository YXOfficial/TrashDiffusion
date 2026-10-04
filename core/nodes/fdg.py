import logging

try:
    from core.guidance import ensure_guidance_pipeline, make_fdg_modifier
except ImportError:  # pragma: no cover - allows direct package-relative import
    from ..guidance import ensure_guidance_pipeline, make_fdg_modifier


class FDGNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "fdg_enabled": ("BOOLEAN", {"default": False}),
                "w_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "w_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "fdg_levels": ("INT", {"default": 3, "min": 2, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Applies Frequency-Decoupled Guidance scaling."

    def patch(self, model, fdg_enabled: bool = False, w_low: float = 1.0, w_high: float = 1.0, fdg_levels: int = 3):
        if not fdg_enabled:
            return (model,)

        patched_model = model.clone()

        if hasattr(model, "_guidance_pipeline"):
            patched_model._guidance_pipeline = model._guidance_pipeline
            patched_model.set_model_sampler_post_cfg_function(
                patched_model._guidance_pipeline.run, "custom_guidance_pipeline"
            )

        pipeline = ensure_guidance_pipeline(patched_model)
        pipeline.add_modifier("fdg", make_fdg_modifier(w_low, w_high, int(fdg_levels)))
        logging.debug(f"Patched model with FDG enabled={fdg_enabled}, w_low={w_low}, w_high={w_high}, levels={fdg_levels}")
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "FDGNode": FDGNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FDGNode": "YX-Frequency-Decoupled Guidance",
}
