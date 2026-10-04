import logging

try:
    from core.guidance import ensure_guidance_pipeline, get_initial_sigma, make_cfg_zero_base_builder
except ImportError:  # pragma: no cover
    from ..guidance import ensure_guidance_pipeline, get_initial_sigma, make_cfg_zero_base_builder


class CFGZeroNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cfg_zero_enabled": ("BOOLEAN", {"default": False}),
                "zero_init_first_step": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Applies CFG-Zero scaling to the sampler guidance."

    def patch(self, model, cfg_zero_enabled: bool = False, zero_init_first_step: bool = False):
        if not cfg_zero_enabled:
            return (model,)

        patched_model = model.clone()

        if hasattr(model, "_guidance_pipeline"):
            patched_model._guidance_pipeline = model._guidance_pipeline
            patched_model.set_model_sampler_post_cfg_function(
                patched_model._guidance_pipeline.run, "custom_guidance_pipeline"
            )

        pipeline = ensure_guidance_pipeline(patched_model)
        initial_sigma = get_initial_sigma(patched_model)
        pipeline.set_base_builder(make_cfg_zero_base_builder(zero_init_first_step, initial_sigma))
        logging.debug(
            f"Patched model with CFG-Zero enabled={cfg_zero_enabled}, zero_init_first_step={zero_init_first_step}"
        )
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "CFGZeroNode": CFGZeroNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFGZeroNode": "YX-CFG-Zero Guidance",
}
