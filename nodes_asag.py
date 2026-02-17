from functools import partial
import torch
from .asag_utils import (
    asag_attention,
    parse_unet_blocks,
    rescale_guidance,
    set_model_options_patch_replace,
    snf_guidance,
)

# Detect backend
try:
    from comfy.samplers import calc_cond_batch
    BACKEND = "ComfyUI"
except ImportError:
    try:
        from ldm_patched.modules.samplers import calc_cond_uncond_batch
        BACKEND = "reForge"
    except ImportError:
        BACKEND = "Forge"

class ASAGGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sinkhorn_iters": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "unet_block": (["input", "middle", "output"], {"default": "middle"}),
                "unet_block_id": ("INT", {"default": 0}),
                "sigma_start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01}),
                "sigma_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01}),
                "rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_mode": (["full", "partial", "snf"], {"default": "full"}),
            },
            "optional": {
                "unet_block_list": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"

    def patch(
        self,
        model,
        scale: float = 1.5,
        sinkhorn_iters: int = 2,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
        unet_block_list: str = "",
    ):
        m = model.clone()
        sigma_start_val = float("inf") if sigma_start < 0 else sigma_start
        single_block = (unet_block, unet_block_id, None)
        blocks, _ = parse_unet_blocks(model, unet_block_list, "attn1") if unet_block_list else ([single_block], None)

        def post_cfg_function(args):
            model_patcher = args["model"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            if scale == 0 or not (sigma_end < sigma[0] <= sigma_start_val):
                return cfg_result

            asag_attn = partial(asag_attention, sinkhorn_iters=sinkhorn_iters)
            for block in blocks:
                layer, number, index = block
                model_options = set_model_options_patch_replace(model_options, asag_attn, "attn1", layer, number, index)

            if BACKEND == "ComfyUI":
                from comfy.samplers import calc_cond_batch as comfy_calc
                (asag_cond_pred,) = comfy_calc(model_patcher, [cond], x, sigma, model_options)
            else:
                from ldm_patched.modules.samplers import calc_cond_uncond_batch as forge_calc
                (asag_cond_pred, _) = forge_calc(model_patcher, cond, None, x, sigma, model_options)

            guidance = (cond_pred - asag_cond_pred) * scale

            if rescale_mode == "snf":
                if uncond_pred.any():
                    return uncond_pred + snf_guidance(cfg_result - uncond_pred, guidance)
                return cfg_result + guidance

            return cfg_result + rescale_guidance(guidance, cond_pred, cfg_result, rescale, rescale_mode)

        m.set_model_sampler_post_cfg_function(post_cfg_function, rescale_mode == "snf")
        return (m,)

NODE_CLASS_MAPPINGS = {"ASAGGuidance": ASAGGuidance}
NODE_DISPLAY_NAME_MAPPINGS = {"ASAGGuidance": "ASAG (Adversarial Sinkhorn Attention Guidance)"}
