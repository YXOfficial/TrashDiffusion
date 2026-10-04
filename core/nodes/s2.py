import logging

try:
    from core.guidance import ensure_guidance_pipeline, make_s2_modifier
except ImportError:  # pragma: no cover
    from ..guidance import ensure_guidance_pipeline, make_s2_modifier


class S2GuidanceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "s2_guidance_enabled": ("BOOLEAN", {"default": False}),
                "s2_scale_omega": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05, "round": 0.01}),
                "s2_drop_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Applies S²-Guidance block dropping during sampling."

    def patch(self, model, s2_guidance_enabled: bool = False, s2_scale_omega: float = 0.25, s2_drop_ratio: float = 0.1):
        if not s2_guidance_enabled:
            return (model,)

        patched_model = model.clone()

        if hasattr(model, "_guidance_pipeline"):
            patched_model._guidance_pipeline = model._guidance_pipeline
            patched_model.set_model_sampler_post_cfg_function(
                patched_model._guidance_pipeline.run, "custom_guidance_pipeline"
            )

        pipeline = ensure_guidance_pipeline(patched_model)
        pipeline.add_modifier("s2_guidance", make_s2_modifier(patched_model, s2_drop_ratio, s2_scale_omega))
        logging.debug(
            f"Patched model with S2-Guidance enabled={s2_guidance_enabled}, omega={s2_scale_omega}, drop_ratio={s2_drop_ratio}"
        )
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "S2GuidanceNode": S2GuidanceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S2GuidanceNode": "YX-S²-Guidance",
}
