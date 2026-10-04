import logging
from typing import Callable, Dict, Optional, Tuple

import torch
from kornia.geometry.transform import build_laplacian_pyramid

class GuidanceState:
    def __init__(self, base_prediction: torch.Tensor, guidance_term: torch.Tensor):
        self.base_prediction = base_prediction
        self.guidance_term = guidance_term

    @property
    def prediction(self) -> torch.Tensor:
        return self.base_prediction + self.guidance_term

class GuidancePipeline:
    def __init__(self):
        self.base_builder: Callable = default_base_builder
        self.modifiers: Dict[str, Callable] = {}

    def set_base_builder(self, builder: Callable):
        self.base_builder = builder

    def add_modifier(self, name: str, modifier: Callable):
        self.modifiers[name] = modifier

    def run(self, args):
        state = self.base_builder(args)
        for modifier in self.modifiers.values():
            state = modifier(args, state)
        return state.prediction

def default_base_builder(args) -> GuidanceState:
    uncond_denoised = args["uncond_denoised"]
    cond_denoised = args["cond_denoised"]
    cond_scale = args["cond_scale"]
    guidance_term = (cond_denoised - uncond_denoised) * cond_scale
    return GuidanceState(uncond_denoised, guidance_term)

def ensure_guidance_pipeline(model) -> GuidancePipeline:
    if not hasattr(model, "_guidance_pipeline"):
        pipeline = GuidancePipeline()
        model._guidance_pipeline = pipeline
        model.set_model_sampler_post_cfg_function(pipeline.run, "custom_guidance_pipeline")
    return model._guidance_pipeline

def get_initial_sigma(model) -> float:
    try:
        return model.model.model_sampling.sigma_max
    except AttributeError:
        logging.warning("Custom Guidance: Could not determine initial_sigma.")
        return float("inf")

def make_cfg_zero_base_builder(zero_init_first_step: bool, initial_sigma: float) -> Callable:
    def builder(args) -> GuidanceState:
        w, cond, uncond = args["cond_scale"], args["cond_denoised"], args["uncond_denoised"]
        dims = tuple(range(1, cond.ndim))
        s = torch.sum(cond * uncond, dim=dims, keepdim=True) / (torch.sum(uncond ** 2, dim=dims, keepdim=True) + 1e-8)
        return GuidanceState(uncond * s, (cond - uncond * s) * w)
    return builder

class CFGCtrlState:
    """State holder for CFG-Ctrl (SMC-CFG) sliding mode control."""
    def __init__(self):
        self.prev_guidance_eps: torch.Tensor = None
        self.step_counter: int = 0

    def reset(self):
        self.prev_guidance_eps = None
        self.step_counter = 0

def make_cfg_ctrl_modifier(
    smc_cfg_enable: bool = False,
    smc_cfg_lambda: float = 5.0,
    smc_cfg_K: float = 0.3,
    no_cfg_warmup_steps: int = 0,
    initial_sigma: float = None,
) -> Callable:
    """
    Creates a modifier that applies SMC-CFG (Sliding Mode Control CFG) on top of any base builder.
    This allows CFG-Ctrl to be combined with CFG-Zero or other base builders.
    Exact implementation from https://github.com/THU-SI/CFG-Ctrl
    Modified for Forge stability (Adaptive K based on guidance magnitude).
    """
    ctrl_state = CFGCtrlState()

    def modifier(args, state: GuidanceState) -> GuidanceState:
        cond_scale = args["cond_scale"]
        cond_denoised = args["cond_denoised"]
        uncond_denoised = args["uncond_denoised"]

        # Detect first step
        current_sigma = args.get("sigma")
        is_first_step = (ctrl_state.step_counter == 0)
        if current_sigma is not None and initial_sigma is not None:
            try:
                sigma_val = current_sigma[0].item() if torch.is_tensor(current_sigma) else float(current_sigma)
                if sigma_val >= initial_sigma * 0.999:
                    is_first_step = True
            except (TypeError, ValueError, IndexError):
                pass

        if is_first_step:
            ctrl_state.reset()

        progress_id = ctrl_state.step_counter
        warmup_no_cfg = no_cfg_warmup_steps > 0 and progress_id < no_cfg_warmup_steps

        # Use the guidance_term from the base builder (or previous modifiers)
        # guidance_eps = cond_denoised - uncond_denoised  # Original approach
        guidance_eps = state.guidance_term / cond_scale if cond_scale != 0 else state.guidance_term

        if smc_cfg_enable and not warmup_no_cfg:
            if ctrl_state.prev_guidance_eps is None:
                ctrl_state.prev_guidance_eps = guidance_eps.detach()

            # s = (e_t - e_{t-1}) + lambda * e_{t-1}
            s = (guidance_eps - ctrl_state.prev_guidance_eps) + smc_cfg_lambda * ctrl_state.prev_guidance_eps

            # Adaptive u_sw: Scale K by the magnitude of guidance to prevent burning
            # This makes K a relative factor, which is more stable across models/steps
            eps_std = torch.std(guidance_eps) + 1e-8
            u_sw = -smc_cfg_K * torch.sign(s) * eps_std

            # Update guidance
            guidance_eps = guidance_eps + u_sw

            # state.prev_guidance_eps = guidance_eps.detach()
            ctrl_state.prev_guidance_eps = guidance_eps.detach()

            ctrl_state.step_counter += 1
            return GuidanceState(state.base_prediction, guidance_eps * cond_scale)

        if warmup_no_cfg:
            ctrl_state.step_counter += 1
            # Return base_prediction only (no guidance)
            return GuidanceState(state.base_prediction, torch.zeros_like(state.guidance_term))

        ctrl_state.step_counter += 1
        return GuidanceState(state.base_prediction, state.guidance_term)

    return modifier

def make_cfg_ctrl_base_builder(
    smc_cfg_enable: bool = False,
    smc_cfg_lambda: float = 5.0,
    smc_cfg_K: float = 0.3,
    no_cfg_warmup_steps: int = 0,
    initial_sigma: float = None,
) -> Callable:
    """
    Creates a base builder that applies SMC-CFG (Sliding Mode Control CFG).
    Deprecated: Use make_cfg_ctrl_modifier instead for better compatibility.
    This is kept for backward compatibility.
    """
    modifier_fn = make_cfg_ctrl_modifier(
        smc_cfg_enable=smc_cfg_enable,
        smc_cfg_lambda=smc_cfg_lambda,
        smc_cfg_K=smc_cfg_K,
        no_cfg_warmup_steps=no_cfg_warmup_steps,
        initial_sigma=initial_sigma,
    )

    def builder(args) -> GuidanceState:
        # Start with default base, then apply CFG-Ctrl modifier
        default_state = GuidanceState(
            args["uncond_denoised"],
            (args["cond_denoised"] - args["uncond_denoised"]) * args["cond_scale"]
        )
        return modifier_fn(args, default_state)

    return builder

def make_fdg_modifier(w_low: float, w_high: float, fdg_levels: int) -> Callable:
    def modifier(args, state: GuidanceState) -> GuidanceState:
        cond_scale = args["cond_scale"]
        if cond_scale != 0:
            guidance_direction = state.guidance_term / cond_scale
        else:
            guidance_direction = state.guidance_term

        # Handle 5D tensors [B, C, T, H, W]
        original_shape = guidance_direction.shape
        is_5d = len(original_shape) == 5
        if is_5d:
            b, c, t, h, w = original_shape
            guidance_direction = guidance_direction.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        guidance_low_freq_scaled = guidance_direction * w_low
        guidance_high_freq_scaled = guidance_direction * w_high
        levels = max(2, int(fdg_levels))
        
        low_freq_part_from_low = build_laplacian_pyramid(guidance_low_freq_scaled, levels)[-1]
        low_freq_part_from_high = build_laplacian_pyramid(guidance_high_freq_scaled, levels)[-1]

        if low_freq_part_from_high.shape != guidance_high_freq_scaled.shape:
            low_freq_part_from_high = torch.nn.functional.interpolate(
                low_freq_part_from_high,
                size=guidance_high_freq_scaled.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if low_freq_part_from_low.shape != guidance_high_freq_scaled.shape:
            low_freq_part_from_low = torch.nn.functional.interpolate(
                low_freq_part_from_low,
                size=guidance_high_freq_scaled.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        high_freq_part = guidance_high_freq_scaled - low_freq_part_from_high
        final_guidance_term = (high_freq_part + low_freq_part_from_low) * cond_scale

        if is_5d:
            final_guidance_term = final_guidance_term.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)

        return GuidanceState(state.base_prediction, final_guidance_term)

    return modifier

# NOTE: Forge processing-object helpers (reset_unet_if_needed,
# clear_generation_params*, GUIDANCE_PARAM_KEYS) used to live here and reached
# directly into ``p.sd_model.forge_objects.unet``. They have moved to
# ``core.forge_compat`` with multi-layout fallbacks so upstream restructures
# degrade to warnings instead of crashes. Import them from there.
