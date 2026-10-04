import torch
import numpy as np
import gradio as gr
import types
from einops import rearrange
from PIL import Image

from ..base import GuidanceProcessor
from ..registry import register_processor
from core.forge_compat import get_unet, set_unet


def _adain_fp32(target: torch.Tensor, style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Standard AdaIN in FP32. Maintains manifold integrity by aligning
    target feature distribution to match the clean reference.
    """
    target_f32 = target.float()
    style_f32 = style.float()

    t_mean = target_f32.mean(dim=1, keepdim=True)
    s_mean = style_f32.mean(dim=1, keepdim=True)

    t_std = target_f32.var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt()
    s_std = style_f32.var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt()

    aligned = (target_f32 - t_mean) / t_std * s_std + s_mean
    return aligned.to(target.dtype)


def make_patched_block_forward(original_block_forward):
    def patched_block_forward(self, x_B_T_H_W_D: torch.Tensor, emb_B_T_D: torch.Tensor, crossattn_emb: torch.Tensor,
                              *args, **kwargs):
        to = kwargs.get('transformer_options', {}) or {}
        active = to.get('fdt_active', False)
        target_b = to.get('fdt_target_batch_size', 0)
        style_blend = to.get('fdt_style_blend', 0.5)

        # 1. Native Execution
        out_B_T_H_W_D = original_block_forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, *args, **kwargs)

        # 2. Block-Exit AdaIN Alignment
        if active and out_B_T_H_W_D.shape[0] > target_b:
            out_tar = out_B_T_H_W_D[:target_b]
            out_ref = out_B_T_H_W_D[target_b:]

            b, t, h, w, d = out_tar.shape
            out_tar_flat = rearrange(out_tar, "b t h w d -> b (t h w) d")
            out_ref_flat = rearrange(out_ref, "b t h w d -> b (t h w) d")

            out_tar_aligned_flat = _adain_fp32(out_tar_flat, out_ref_flat)
            out_tar_aligned = rearrange(out_tar_aligned_flat, "b (t h w) d -> b t h w d", t=t, h=h, w=w)

            # Linear Manifold Blending
            out_tar = out_tar * (1.0 - style_blend) + out_tar_aligned * style_blend
            out_B_T_H_W_D = torch.cat([out_tar, out_ref], dim=0)

        return out_B_T_H_W_D

    return patched_block_forward


def make_model_wrapper(ref_latent, style_blend):
    def model_function_wrapper(apply_model, args):
        input_x = args['input']
        timestep = args['timestep']
        c = args['c'].copy()
        target_b = input_x.shape[0]

        if ref_latent is not None:
            device = input_x.device
            dtype = input_x.dtype

            t_val = timestep.max().item()
            sigma = t_val / 1000.0 if t_val > 1.0 else t_val
            sigma = max(0.0, min(1.0, float(sigma)))

            # PERFORMANCE OPTIMIZATION: Exit dual-batch at low sigma to restore 1X speed
            if sigma < 0.1:
                return apply_model(input_x, timestep, **c)

            ref_clean = ref_latent.to(device=device, dtype=dtype)
            if ref_clean.shape[-2:] != input_x.shape[-2:]:
                spatial_dims = ref_clean.ndim - 2
                ref_clean = torch.nn.functional.interpolate(
                    ref_clean, size=input_x.shape[-spatial_dims:],
                    mode="trilinear" if spatial_dims == 3 else "bilinear", align_corners=False
                )

            # Manifold Noise Sync
            generator = torch.Generator(device=device).manual_seed(42)
            noise = torch.randn(ref_clean.shape, device=device, dtype=dtype, generator=generator)
            ref_noisy = (1.0 - sigma) * ref_clean + sigma * noise

            # Prevent numeric shock (RMS Alignment)
            rms_input = input_x.float().pow(2).mean().sqrt()
            rms_ref = ref_noisy.float().pow(2).mean().sqrt()
            ref_noisy = ref_noisy * (rms_input / (rms_ref + 1e-6))

            input_for_model = torch.cat([input_x, ref_noisy.to(dtype)], dim=0)
            timestep_for_model = torch.cat([timestep, timestep[0:1]], dim=0) if timestep.shape[
                                                                                    0] == target_b else timestep

            c_dict = c.copy()
            for k_cond, v_cond in list(c_dict.items()):
                if k_cond != 'transformer_options' and isinstance(v_cond, torch.Tensor) and v_cond.shape[0] == target_b:
                    c_dict[k_cond] = torch.cat([v_cond, v_cond[0:1]], dim=0)

            to = c_dict.get('transformer_options', {}).copy()
            to['fdt_active'] = True
            to['fdt_target_batch_size'] = target_b

            # FIDELITY ENHANCEMENT: style_blend fades as sigma approaches zero
            # This allows VNA to lock sharp micro-details in final steps
            to['fdt_style_blend'] = style_blend * (sigma ** 0.5)
            c_dict['transformer_options'] = to

            raw_result = apply_model(input_for_model, timestep_for_model, **c_dict)
            return raw_result[:target_b]

        return apply_model(input_x, timestep, **c)

    return model_function_wrapper


class FeatureDistributionTransferProcessor(GuidanceProcessor):
    def name(self) -> str:
        return "Feature Distribution Transfer"

    def create_ui(self):
        with gr.Tab(label="Style Transfer (Block-Exit)"):
            gr.Markdown("### Block-Exit Style Transfer\n**Global Manifold Sync**")
            enabled = gr.Checkbox(label="Enable Transfer", value=False)
            ref_image = gr.Image(label="Reference Image", type="pil")
            style_blend = gr.Slider(label="Style Strength", minimum=0.0, maximum=1.0, step=0.05, value=0.85)

        return [enabled, ref_image, style_blend]

    def process(self, p, enabled, ref_image, style_blend):
        if not enabled or ref_image is None: return

        # VAE RESOLUTION LOCK: Prevent Tiled VAE Artifacts
        max_vae_res = 1536
        w, h = ref_image.size
        if max(w, h) > max_vae_res:
            scale = max_vae_res / max(w, h)
            ref_image = ref_image.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)
            print(f"[FDT] Reference locked to {ref_image.size} to force Global VAE Statistics.")

        if hasattr(ref_image, 'convert'): ref_image = ref_image.convert("RGB")
        img_np = np.array(ref_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        _holder = getattr(p.sd_model, "forge_objects", p.sd_model)
        vae_obj = getattr(_holder, "vae", None)
        if vae_obj is None:
            print("[FDT] No VAE handle found; skipping.")
            return
        img_tensor = img_tensor.to(device=vae_obj.device, dtype=vae_obj.vae_dtype)

        _encode = getattr(p.sd_model, "encode_first_stage", None)
        if _encode is None:
            print("[FDT] encode_first_stage missing; skipping.")
            return
        with torch.inference_mode():
            ref_latent = _encode(img_tensor * 2.0 - 1.0)
            if torch.isnan(ref_latent).any():
                print("[FDT] VAE NaN detected. Re-encoding in FP32...")
                ref_latent = _encode((img_tensor * 2.0 - 1.0).float()).cpu()
            else:
                ref_latent = ref_latent.cpu()

        unet = get_unet(p).clone()
        model = unet.model.diffusion_model

        patched_count = 0
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'Block':
                if not hasattr(module, '_original_forward'):
                    module._original_forward = module.forward
                    module.forward = types.MethodType(make_patched_block_forward(module._original_forward), module)
                    patched_count += 1

        model_wrapper = make_model_wrapper(ref_latent, float(style_blend))
        unet.set_model_unet_function_wrapper(model_wrapper)

        set_unet(p, unet)
        print(f"[FDT] Patched {patched_count} blocks. Global Manifold Transfer active.")

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        pass


register_processor(FeatureDistributionTransferProcessor)