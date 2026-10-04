import logging

try:
    from core.guidance import ensure_guidance_pipeline, make_zeresfdg_base_builder, make_zeresfdg_modifier
except ImportError:  # pragma: no cover
    from ..guidance import ensure_guidance_pipeline, make_zeresfdg_base_builder, make_zeresfdg_modifier


class ZeResFDGNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "zeresfdg_enabled": ("BOOLEAN", {"default": False}),
                "w_low": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 10.0, "step": 0.05, "round": 0.01}),
                "w_high": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.05, "round": 0.01}),
                "alpha": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "tau_lo": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "tau_hi": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "beta": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "controller_enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Applies ZeResFDG (CADE 2.5) guidance with spectral EMA switching."

    def patch(
        self,
        model,
        zeresfdg_enabled: bool = False,
        w_low: float = 0.6,
        w_high: float = 1.3,
        alpha: float = 0.7,
        tau_lo: float = 0.45,
        tau_hi: float = 0.6,
        beta: float = 0.8,
        controller_enabled: bool = True,
    ):
        if not zeresfdg_enabled:
            return (model,)

        patched_model = model.clone()

        if hasattr(model, "_guidance_pipeline"):
            patched_model._guidance_pipeline = model._guidance_pipeline
            patched_model.set_model_sampler_post_cfg_function(
                patched_model._guidance_pipeline.run, "custom_guidance_pipeline"
            )

        pipeline = ensure_guidance_pipeline(patched_model)
        pipeline.set_base_builder(make_zeresfdg_base_builder())
        pipeline.add_modifier(
            "zeresfdg",
            make_zeresfdg_modifier(
                w_low=w_low,
                w_high=w_high,
                alpha=alpha,
                tau_lo=tau_lo,
                tau_hi=tau_hi,
                beta=beta,
                controller_enabled=controller_enabled,
            ),
        )

        logging.debug(
            "Patched model with ZeResFDG enabled=%s, w_low=%s, w_high=%s, alpha=%s, tau_lo=%s, tau_hi=%s, beta=%s, controller=%s",
            zeresfdg_enabled,
            w_low,
            w_high,
            alpha,
            tau_lo,
            tau_hi,
            beta,
            controller_enabled,
        )
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "ZeResFDGNode": ZeResFDGNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZeResFDGNode": "YX-ZeResFDG Guidance",
}
