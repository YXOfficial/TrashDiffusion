import logging
import random
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
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


def _percentiles_per_sample(
    x: torch.Tensor, q_low: float, q_high: float, per_channel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sample (optionally per-channel) quantiles in FP32."""

    b = x.shape[0]
    if per_channel and x.dim() >= 3:
        flat = x.reshape(b, x.shape[1], -1).to(torch.float32)
        l = torch.quantile(flat, q_low, dim=2)
        h = torch.quantile(flat, q_high, dim=2)
        shape = (b, x.shape[1]) + (1,) * (x.dim() - 2)
    else:
        flat = x.reshape(b, -1).to(torch.float32)
        l = torch.quantile(flat, q_low, dim=1)
        h = torch.quantile(flat, q_high, dim=1)
        shape = (b,) + (1,) * (x.dim() - 1)

    return l.view(shape), h.view(shape)


def qsilk_micrograin(
    x: torch.Tensor,
    q_low: float = 0.001,
    q_high: float = 0.999,
    alpha: float = 2.0,
    vp_mix: float = 0.15,
    late_weight: Optional[float] = None,
    per_channel: bool = False,
) -> torch.Tensor:
    """Tanh soft-clamp micrograin with optional VP-mix and late-step blending."""

    if x.dim() < 2:
        return x

    orig_dtype = x.dtype
    use_per_channel = per_channel and x.dim() >= 3

    x32 = x.to(torch.float32)
    x32 = torch.nan_to_num(x32, nan=0.0, posinf=1e6, neginf=-1e6)

    dims = tuple(range(2, x32.dim())) if use_per_channel else tuple(range(1, x32.dim()))

    mu0 = x32.mean(dim=dims, keepdim=True)
    var0 = x32.var(dim=dims, unbiased=False, keepdim=True)
    std0 = torch.sqrt(var0 + 1e-6)

    l, h = _percentiles_per_sample(x32, q_low, q_high, per_channel=use_per_channel)
    m = 0.5 * (l + h)
    delta = 0.5 * (h - l)

    y = (x32 - m) / (delta + 1e-4)
    x_soft = m + delta * torch.tanh(alpha * y)

    mu1 = x_soft.mean(dim=dims, keepdim=True)
    var1 = x_soft.var(dim=dims, unbiased=False, keepdim=True)
    std1 = torch.sqrt(var1 + 1e-4)

    y2 = (x_soft - mu1) / (std1 + 1e-4)
    x_vp = y2 * std0 + mu0

    mix = float(max(0.0, min(1.0, vp_mix)))
    x_out = x_soft.lerp(x_vp, mix)

    if late_weight is not None:
        lw = float(max(0.0, min(1.0, late_weight)))
        x_out = x32.lerp(x_out, lw)

    return x_out.to(orig_dtype)


def ndtri(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Inverse normal CDF via erfinv for tensor inputs."""

    q_clamped = q.clamp(eps, 1.0 - eps)
    return torch.sqrt(torch.tensor(2.0, device=q.device, dtype=q.dtype)) * torch.erfinv(2 * q_clamped - 1)


def qsilk_aqclip_lite(
    x: torch.Tensor,
    tile_size: int = 32,
    stride: int = 16,
    alpha: float = 2.0,
    ema_state: Optional[Dict[str, torch.Tensor]] = None,
    ema_beta: float = 0.8,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Adaptive quantile clipping with tile-wise EMA corridors."""

    if ema_state is None:
        ema_state = {}

    if x.dim() != 4:
        return x, ema_state

    dtype = x.dtype
    device = x.device
    b, c, h, w = x.shape
    x_float = x.float()

    z_m = x_float.mean(dim=1, keepdim=True)

    g_x = F.pad(z_m[..., 2:] - z_m[..., :-2], (1, 1, 0, 0))
    g_y = F.pad(z_m[..., 2:, :] - z_m[..., :-2, :], (0, 0, 1, 1))
    g = torch.sqrt(g_x**2 + g_y**2)

    g_abs = g.abs()
    g_max = g_abs.amax(dim=(2, 3), keepdim=True)
    if torch.all(g_max <= eps):
        return x, ema_state

    g_norm = g_abs / (g_max + eps)
    g_grid = F.avg_pool2d(g_norm, kernel_size=tile_size, stride=stride)

    q_low_tile = 0.5 * (g_grid**2)
    q_high_tile = 1.0 - 0.5 * ((1.0 - g_grid) ** 2)

    unfold = torch.nn.Unfold(kernel_size=tile_size, stride=stride)
    fold = torch.nn.Fold(output_size=(h, w), kernel_size=tile_size, stride=stride)
    x_flat = x_float.view(b * c, 1, h, w)
    patches = unfold(x_flat)

    mu = patches.mean(dim=1, keepdim=True)
    sigma = patches.std(dim=1, keepdim=True) + eps

    q_low_flat = q_low_tile.view(b, 1, -1)
    q_high_flat = q_high_tile.view(b, 1, -1)
    q_low_bc = q_low_flat.repeat_interleave(c, dim=0)
    q_high_bc = q_high_flat.repeat_interleave(c, dim=0)

    l = mu + sigma * ndtri(q_low_bc, eps=eps)
    h_corr = mu + sigma * ndtri(q_high_bc, eps=eps)

    if "l" not in ema_state or ema_state.get("l") is None or ema_state["l"].shape != l.shape:
        ema_state["l"] = l.detach()
        ema_state["h"] = h_corr.detach()
    else:
        ema_state["l"] = ema_beta * ema_state["l"] + (1.0 - ema_beta) * l.detach()
        ema_state["h"] = ema_beta * ema_state["h"] + (1.0 - ema_beta) * h_corr.detach()

    m = (ema_state["l"] + ema_state["h"]) / 2
    delta = (ema_state["h"] - ema_state["l"]) / 2

    normed = (patches - m) / (delta + eps)
    patched = m + delta * torch.tanh(alpha * normed)

    x_clipped_flat = fold(patched)
    
    # Normalization map (fold(1))
    ones = torch.ones_like(x_flat)
    norm_map = fold(unfold(ones))
    
    x_clipped_flat = x_clipped_flat / (norm_map + 1e-4)
    x_clipped = x_clipped_flat.view(b, c, h, w)

    return x_clipped.to(dtype=dtype), ema_state


def make_qsilk_modifier(
    micro_q_low: float = 0.001,
    micro_q_high: float = 0.999,
    micro_alpha: float = 2.0,
    micro_vp_mix: float = 0.15,
    per_channel: bool = False,
    enable_late_weighting: bool = True,
    use_aqclip: bool = False,
    tile_size: int = 32,
    stride: int = 16,
    aqclip_alpha: float = 2.0,
    ema_beta: float = 0.8,
) -> Callable:
    ema_state: Dict[str, torch.Tensor] = {}

    def _smoothstep(x: float, edge0: float = 0.35, edge1: float = 0.85) -> float:
        t = (x - edge0) / (edge1 - edge0 + 1e-6)
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _get_sigma_bounds(args) -> Tuple[Optional[float], Optional[float]]:
        sigma_max_val = args.get("sigma_max")
        sigma_min_val = args.get("sigma_min")

        sigmas = args.get("sigmas")
        if sigmas is not None:
            try:
                if torch.is_tensor(sigmas) and sigmas.numel() > 0:
                    sigma_max_val = float(torch.max(sigmas).item())
                    sigma_min_val = float(torch.min(sigmas).item())
                elif isinstance(sigmas, (list, tuple)) and len(sigmas) > 0:
                    sigma_max_val = float(max(sigmas))
                    sigma_min_val = float(min(sigmas))
            except Exception:
                logging.debug("QSilk: Unable to derive sigma bounds from 'sigmas'.")

        try:
            sigma_max_val = float(sigma_max_val) if sigma_max_val is not None else None
            sigma_min_val = float(sigma_min_val) if sigma_min_val is not None else None
        except (TypeError, ValueError):
            sigma_max_val, sigma_min_val = None, None

        return sigma_max_val, sigma_min_val

    def _compute_late_weight(args) -> Optional[float]:
        if not enable_late_weighting:
            return None

        sigma_val = None
        if "sigma" in args:
            try:
                sigma_raw = args["sigma"]
                if torch.is_tensor(sigma_raw):
                    sigma_val = float(sigma_raw.detach().view(-1)[0].item())
                else:
                    sigma_val = float(sigma_raw)
            except Exception:
                logging.debug("QSilk: Unable to parse sigma value for late weighting.")

        sigma_max_val, sigma_min_val = _get_sigma_bounds(args)

        if sigma_val is not None and sigma_max_val is not None and sigma_min_val is not None:
            if sigma_max_val > sigma_min_val:
                frac = 1.0 - (sigma_val - sigma_min_val) / (sigma_max_val - sigma_min_val + 1e-6)
                return _smoothstep(frac)

        step = args.get("step")
        total_steps = args.get("total_steps") or args.get("steps")
        try:
            if step is not None and total_steps is not None:
                frac = float(step) / max(float(total_steps), 1.0)
                return _smoothstep(frac)
        except Exception:
            logging.debug("QSilk: Unable to compute step-based late weighting.")

        return None

    def modifier(args, state: GuidanceState) -> GuidanceState:
        nonlocal ema_state
        pred = state.prediction
        late_weight = _compute_late_weight(args)
        pred = qsilk_micrograin(
            pred,
            q_low=micro_q_low,
            q_high=micro_q_high,
            alpha=micro_alpha,
            vp_mix=micro_vp_mix,
            late_weight=late_weight,
            per_channel=per_channel,
        )

        if use_aqclip:
            pred, ema_state = qsilk_aqclip_lite(
                pred,
                tile_size=tile_size,
                stride=stride,
                alpha=aqclip_alpha,
                ema_state=ema_state,
                ema_beta=ema_beta,
            )

        guidance_term = pred - state.base_prediction
        return GuidanceState(state.base_prediction, guidance_term)

    return modifier


def get_initial_sigma(model) -> float:
    try:
        return model.model.model_sampling.sigma_max
    except AttributeError:
        logging.warning("Custom Guidance: Could not determine initial_sigma.")
        return float("inf")


def make_cfg_zero_base_builder(zero_init_first_step: bool, initial_sigma: float) -> Callable:
    def builder(args) -> GuidanceState:
        cond_scale = args["cond_scale"]
        cond_denoised = args["cond_denoised"]
        uncond_denoised = args["uncond_denoised"]

        if zero_init_first_step:
            current_sigma_val = args["sigma"][0].item()
            if abs(current_sigma_val - initial_sigma) < 1e-5:
                zero_tensor = uncond_denoised * 0.0
                return GuidanceState(zero_tensor, zero_tensor)

        original_shape = cond_denoised.shape
        batch_size = original_shape[0] if len(original_shape) > 0 else 1
        positive_flat = cond_denoised.reshape(batch_size, -1)
        negative_flat = uncond_denoised.reshape(batch_size, -1)
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm_negative = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        st_star = dot_product / squared_norm_negative
        st_star_reshaped = st_star.view(batch_size, *([1] * (len(original_shape) - 1)))
        base_pred = uncond_denoised * st_star_reshaped
        guidance_direction = cond_denoised - base_pred
        guidance_term = guidance_direction * cond_scale
        return GuidanceState(base_pred, guidance_term)

    return builder


def make_zeresfdg_base_builder() -> Callable:
    def builder(args) -> GuidanceState:
        cond_scale = args["cond_scale"]
        cond_denoised = args["cond_denoised"]
        uncond_denoised = args["uncond_denoised"]

        original_shape = cond_denoised.shape
        batch_size = original_shape[0] if len(original_shape) > 0 else 1
        positive_flat = cond_denoised.reshape(batch_size, -1)
        negative_flat = uncond_denoised.reshape(batch_size, -1)
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm_negative = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        alpha_parallel = torch.clamp(dot_product / squared_norm_negative, min=0.0)
        alpha_parallel = alpha_parallel.view(batch_size, *([1] * (len(original_shape) - 1)))
        base_pred = uncond_denoised * alpha_parallel
        residual = cond_denoised - base_pred
        guidance_term = residual * cond_scale
        return GuidanceState(base_pred, guidance_term)

    return builder


def make_fdg_modifier(w_low: float, w_high: float, fdg_levels: int) -> Callable:
    def modifier(args, state: GuidanceState) -> GuidanceState:
        cond_scale = args["cond_scale"]
        if cond_scale != 0:
            guidance_direction = state.guidance_term / cond_scale
        else:
            guidance_direction = state.guidance_term

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

        return GuidanceState(state.base_prediction, final_guidance_term)

    return modifier


def make_zeresfdg_modifier(
    w_low: float,
    w_high: float,
    alpha: float,
    tau_lo: float,
    tau_hi: float,
    beta: float,
    controller_enabled: bool,
    sigma: float = 1.0,
):
    eps = 1e-12
    rho = None
    mode_state = None

    def gaussian_lowpass(tensor: torch.Tensor) -> torch.Tensor:
        input_dim = tensor.dim()
        x = tensor.unsqueeze(0) if input_dim == 3 else tensor
        k_size = int(max(3, 2 * round(3 * sigma) + 1))
        if k_size % 2 == 0: k_size += 1
        out = gaussian_blur2d(x, kernel_size=(k_size, k_size), sigma=(sigma, sigma), border_type="reflect")
        return out.squeeze(0) if input_dim == 3 else out

    def ensure_mask(mask, target: torch.Tensor) -> torch.Tensor:
        if mask is None: return None
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=target.device, dtype=target.dtype)
        else:
            mask = mask.to(device=target.device, dtype=target.dtype)
        if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3: mask = mask.unsqueeze(1)
        if mask.shape[0] == 1 and target.shape[0] > 1:
            mask = mask.expand(target.shape[0], *mask.shape[1:])
        if mask.shape[-2:] != target.shape[-2:]:
            mask = F.interpolate(mask, size=target.shape[-2:], mode="bilinear", align_corners=False)
        if mask.shape[1] == 1 and target.shape[1] > 1:
            mask = mask.expand(target.shape[0], target.shape[1], *mask.shape[-2:])
        return mask

    def apply_mask(tensor: torch.Tensor, mask) -> torch.Tensor:
        mask_tensor = ensure_mask(mask, tensor)
        if mask_tensor is None: return tensor
        return tensor * mask_tensor

    def update_mode(ratio: torch.Tensor):
        nonlocal mode_state
        if not controller_enabled:
            return torch.ones_like(ratio, dtype=torch.bool)
        if mode_state is None or mode_state.shape != ratio.shape:
            mode_state = ratio > tau_hi
        new_mode = mode_state.clone()
        new_mode = torch.where(ratio > tau_hi, torch.ones_like(new_mode, dtype=torch.bool), new_mode)
        new_mode = torch.where(ratio < tau_lo, torch.zeros_like(new_mode, dtype=torch.bool), new_mode)
        mode_state = new_mode
        return mode_state

    def modifier(args, state: GuidanceState) -> GuidanceState:
        nonlocal rho
        cond_scale = args["cond_scale"]
        cond_denoised = args["cond_denoised"]
        uncond_denoised = args["uncond_denoised"]
        guidance_mask = args.get("guidance_mask") or args.get("mask") or args.get("g")
        g_mask = ensure_mask(guidance_mask, cond_denoised)

        # 1. FDG Preparation (yc - yu)
        delta_rescale = cond_denoised - uncond_denoised
        dl_rescale = gaussian_lowpass(delta_rescale)
        dh_rescale = delta_rescale - dl_rescale
        
        # Spectral Controller Input (r_hf) using Energy (Sum of Squares)
        batch_size = cond_denoised.shape[0]
        dims = tuple(range(1, cond_denoised.dim()))
        
        low_energy = torch.sum(dl_rescale.reshape(batch_size, -1).to(torch.float32) ** 2, dim=1)
        high_energy = torch.sum(dh_rescale.reshape(batch_size, -1).to(torch.float32) ** 2, dim=1)
        r_hf = high_energy / (low_energy + high_energy + eps)
        r_hf = r_hf.to(cond_denoised.dtype)
        
        # EMA Update: rho = beta * rho + (1 - beta) * r_hf
        if rho is None or rho.shape != r_hf.shape:
            rho = r_hf.detach()
        else:
            rho = beta * rho + (1 - beta) * r_hf.detach()
            
        mode = update_mode(rho)
        m = mode.to(dtype=cond_denoised.dtype).view(batch_size, *([1] * (cond_denoised.dim() - 1)))
        
        target_std = cond_denoised.std(dim=dims, keepdim=True)

        # --- BRANCH 1: ZERO (Conservative / Zero-Projection) ---
        # Explicitly calculate projection here to be safe (ignore external base_builder)
        # Algorithm 1 Step 4(a): alpha_parallel = <yc, yu> / <yu, yu>
        # NO CLAMPING: Strictly follow the projection math.
        flat_c = cond_denoised.reshape(batch_size, -1)
        flat_u = uncond_denoised.reshape(batch_size, -1)
        dot_prod = torch.sum(flat_c * flat_u, dim=1, keepdim=True)
        norm_sq = torch.sum(flat_u ** 2, dim=1, keepdim=True) + eps
        
        alpha_par = dot_prod / norm_sq
        alpha_par = alpha_par.view(batch_size, *([1] * (cond_denoised.dim() - 1)))
        
        u_proj = uncond_denoised * alpha_par
        
        delta_z = cond_denoised - u_proj
        dl_z = gaussian_lowpass(delta_z)
        dh_z = delta_z - dl_z
        
        fdg_z = w_low * dl_z + w_high * dh_z
        if g_mask is not None: 
            fdg_z = fdg_z * g_mask
            
        # GuidanceState simply adds base + guidance.
        # So we MUST multiply by cond_scale here to get the correct magnitude.
        guidance_z = fdg_z * cond_scale
        
        # --- BRANCH 2: RESCALE (Detail-seeking) ---
        # 1. Simulate y_cfg = yu + s * FDG(yc - yu)
        fdg_rescale = w_low * dl_rescale + w_high * dh_rescale
        if g_mask is not None: 
            fdg_rescale = fdg_rescale * g_mask
            
        y_cfg_sim = uncond_denoised + cond_scale * fdg_rescale
        
        # 2. Rescale & Blend
        y_cfg_std = y_cfg_sim.std(dim=dims, keepdim=True)
        r_factor = target_std / (y_cfg_std + eps)
        y_cfg_rescaled = y_cfg_sim * r_factor
        
        y_target_rescale = alpha * y_cfg_rescaled + (1.0 - alpha) * y_cfg_sim
        
        # 3. Calculate Guidance Term
        # We need (Base + Guidance) = y_target
        # Since Base for Rescale mode is uncond_denoised,
        # Guidance = y_target - uncond_denoised
        guidance_r = y_target_rescale - uncond_denoised

        # --- BLENDING ---
        # Final Base switches between u_proj (Zero mode) and yu (Rescale mode)
        final_base = (1.0 - m) * u_proj + m * uncond_denoised
        final_guidance = (1.0 - m) * guidance_z + m * guidance_r

        logging.info(f"ZeResFDG | rho:{rho.mean().item():.4f} | r_hf:{r_hf.mean().item():.4f} | Mode:{'RESCALE' if m.mean() > 0.5 else 'ZERO'}")
        return GuidanceState(final_base, final_guidance)

    return modifier


def get_prediction_with_dropped_blocks(diffusion_model, model_function, drop_ratio: float, model_kwargs):
    target_modules = []
    if hasattr(diffusion_model, "blocks"):
        target_modules = diffusion_model.blocks
    elif hasattr(diffusion_model, "output_blocks") and hasattr(diffusion_model, "input_blocks"):
        target_modules = diffusion_model.input_blocks + diffusion_model.output_blocks

    if not target_modules:
        logging.warning("S2-Guidance: Could not find target blocks to drop. S2-Guidance will have no effect.")
        return model_function(**model_kwargs)

    num_blocks_to_drop = int(len(target_modules) * drop_ratio)
    if num_blocks_to_drop == 0:
        return model_function(**model_kwargs)

    original_forwards = {}
    blocks_to_drop_indices = random.sample(range(len(target_modules)), num_blocks_to_drop)
    identity_forward = lambda x, *args, **kwargs: x

    try:
        for i in blocks_to_drop_indices:
            module = target_modules[i]
            original_forwards[i] = module.forward
            module.forward = identity_forward

        logging.debug(f"S2-Guidance: Dropped {num_blocks_to_drop} blocks.")
        prediction = model_function(**model_kwargs)
    finally:
        for i, original_forward in original_forwards.items():
            target_modules[i].forward = original_forward

    return prediction


def make_s2_modifier(model, s2_drop_ratio: float, s2_scale_omega: float) -> Callable:
    def modifier(args, state: GuidanceState) -> GuidanceState:
        model_options = args.get("model_options", {}) or {}
        cond_list = args["cond"]
        sigma = args["sigma"]
        x = args["input"]
        t = sigma

        if not (isinstance(cond_list, list) and len(cond_list) > 0 and isinstance(cond_list[0], dict)):
            logging.warning("S2-Guidance: unexpected cond format; skip S2 this step.")
            return state
        h0 = cond_list[0]

        c_crossattn_tensor = h0.get("cross_attn", None)
        if c_crossattn_tensor is None:
            logging.warning("S2-Guidance: cross_attn tensor not found; skip S2 this step.")
            return state

        needs_y = getattr(model.model.diffusion_model, "num_classes", None) is not None
        y_tensor = h0.get("pooled_output", None) if needs_y else None
        if needs_y and y_tensor is None:
            logging.warning("S2-Guidance: model needs 'y' but pooled_output not found; skip S2 this step.")
            return state

        c_concat = None
        mc = h0.get("model_conds", None)
        if isinstance(mc, dict) and "c_concat" in mc:
            v = mc["c_concat"]
            c_concat = getattr(v, "cond", v)

        control = h0.get("control", None)

        model_kwargs = {
            "x": x,
            "t": t,
            "c_crossattn": c_crossattn_tensor,
            "transformer_options": model_options,
        }
        if needs_y:
            model_kwargs["y"] = y_tensor
        if c_concat is not None:
            model_kwargs["c_concat"] = c_concat
        if control is not None:
            model_kwargs["control"] = control

        denoised_s = get_prediction_with_dropped_blocks(
            model.model.diffusion_model,
            model.model.apply_model,
            s2_drop_ratio,
            model_kwargs,
        )
        new_prediction = state.prediction - s2_scale_omega * denoised_s
        zero_term = torch.zeros_like(state.guidance_term)
        return GuidanceState(new_prediction, zero_term)

    return modifier


GUIDANCE_PARAM_KEYS = [
    "CFG-Zero Enabled",
    "CFG-Zero Init First Step",
    "FDG Enabled",
    "FDG w_low",
    "FDG w_high",
    "FDG Levels",
    "ZeResFDG Enabled",
    "ZeResFDG λ_l",
    "ZeResFDG λ_h",
    "ZeResFDG α",
    "ZeResFDG τ_lo",
    "ZeResFDG τ_hi",
    "ZeResFDG β",
    "ZeResFDG Controller",
    "S2-Guidance Enabled",
    "S2-Guidance Omega",
    "S2-Guidance Drop Ratio",
    "QSilk Enabled",
    "QSilk micro_q_low",
    "QSilk micro_q_high",
    "QSilk micro_alpha",
    "QSilk use_aqclip",
    "QSilk tile_size",
    "QSilk stride",
    "QSilk aqclip_alpha",
    "QSilk ema_beta",
    "TPSO Enabled",
    "TPSO Steps",
    "TPSO LR",
    "TPSO Lambda",
    "TPSO r",
]


def reset_unet_if_needed(p) -> bool:
    if not hasattr(p, "_guidance_original_unet"):
        p._guidance_original_unet = p.sd_model.forge_objects.unet.clone()
    if getattr(p, "_guidance_unet_restored", False):
        return False
    p.sd_model.forge_objects.unet = p._guidance_original_unet.clone()
    p._guidance_unet_restored = True
    return True


def clear_generation_params(p, keys):
    for param in keys:
        p.extra_generation_params.pop(param, None)


def clear_generation_params_once(p, keys=GUIDANCE_PARAM_KEYS):
    if getattr(p, "_guidance_params_cleared", False):
        return False
    clear_generation_params(p, keys)
    p._guidance_params_cleared = True
    return True
