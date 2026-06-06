import logging
import torch
import numpy as np
import gradio as gr
import types

from ..base import GuidanceProcessor
from ..registry import register_processor


def _adain(target: torch.Tensor, style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    t_mean = target.mean(dim=1, keepdim=True)
    s_mean = style.mean(dim=1, keepdim=True)
    t_std = target.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    s_std = style.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    return (target - t_mean) / t_std * s_std + s_mean


def build_scale_vector(
        head_dim: int,
        high_scale: float,
        low_scale: float,
        beta: float,
        device: torch.device,
        dtype: torch.dtype
) -> torch.Tensor:
    # Tái tạo lại logic chia trục (3D RoPE) mà Anima support đang dùng,
    # nhưng tính toán nội suy và scale phải CHUẨN như bản ComfyUI.
    dim_h = (head_dim // 6) * 2
    dim_w = dim_h
    dim_t = head_dim - (dim_h + dim_w)

    axes_dims = [dim_t, dim_h, dim_w] if dim_t > 0 else [head_dim]
    is_3axis = len(axes_dims) == 3

    pieces = []
    for axis_idx, axis_dim in enumerate(axes_dims):
        n_pairs = axis_dim // 2
        if n_pairs <= 0:
            pieces.append(torch.ones(axis_dim, device=device, dtype=dtype))
            continue

        if is_3axis and axis_idx == 0:
            # Chiều thời gian (Time) giữ Low Scale
            pair_scales = torch.full((n_pairs,), float(low_scale), device=device, dtype=torch.float32)
        else:
            # Chiều không gian (Spatial) áp dụng nội suy curve
            d_tilde = (
                torch.zeros(1, device=device, dtype=torch.float32)
                if n_pairs == 1
                else torch.linspace(0.0, 1.0, n_pairs, device=device, dtype=torch.float32)
            )
            pair_scales = high_scale + (low_scale - high_scale) * d_tilde.pow(float(beta))

        # [QUAN TRỌNG NHẤT]: Phải dùng repeat_interleave(2) thay vì torch.cat([x, x])
        # để các tần số được lặp kề nhau [f1, f1, f2, f2] đúng chuẩn RoPE.
        pieces.append(pair_scales.to(dtype=dtype).repeat_interleave(2))

        # Nếu dimension bị lẻ, đệm số 1 vào cuối
        if axis_dim % 2:
            pieces.append(torch.ones(1, device=device, dtype=dtype))

    out = torch.cat(pieces, dim=0)

    # Đảm bảo tensor khớp đúng số lượng head_dim
    if out.numel() >= head_dim:
        out = out[:head_dim]
    else:
        out = torch.nn.functional.pad(out, (0, head_dim - out.numel()), value=1.0)

    # (Đã xoá bỏ dòng torch.clamp(0.0, 4.0) bị ảo giác của AI cũ)
    return out.view(*([1] * (out.ndim + 2)), head_dim)


def make_patched_forward(original_forward):
    def patched_forward(self, x: torch.Tensor, *args, **kwargs):
        transformer_options = kwargs.get('transformer_options', {})
        context = kwargs.get('context', None)
        rope_emb = kwargs.get('rope_emb', None)

        if not kwargs:
            if len(args) >= 1: context = args[0]
            if len(args) >= 2: rope_emb = args[1]
            if len(args) >= 3: transformer_options = args[2]

        to = transformer_options or {}
        active = to.get('untwist_rope_active', False)
        target_b = to.get('untwist_target_b', 0)

        if active and self.is_SelfAttn and context is None:
            # GIAI ĐOẠN 1: Chạy song song (Dual-Batch) để nạp đầy Cache
            if x.shape[0] > target_b:
                q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)

                q_tar, q_ref = q[:target_b], q[target_b:]
                k_tar, k_ref = k[:target_b], k[target_b:]
                v_tar, v_ref = v[:target_b], v[target_b:]

                high_scale = to.get('untwist_high_scale', 0.0)
                low_scale = to.get('untwist_low_scale', 3.0)
                beta = to.get('untwist_beta', 50.0)
                adain_strength = to.get('untwist_adain', 0.5)

                if adain_strength > 0.0:
                    q_ref_exp = q_ref.expand(target_b, *([-1] * (q_ref.ndim - 1)))
                    k_ref_exp = k_ref.expand(target_b, *([-1] * (k_ref.ndim - 1)))

                    q_tar_adain = _adain(q_tar, q_ref_exp)
                    k_tar_adain = _adain(k_tar, k_ref_exp)

                    q_tar = q_tar * (1.0 - adain_strength) + q_tar_adain * adain_strength
                    k_tar = k_tar * (1.0 - adain_strength) + k_tar_adain * adain_strength

                scale_vec = build_scale_vector(
                    head_dim=self.head_dim,
                    high_scale=high_scale,
                    low_scale=low_scale,
                    beta=beta,
                    device=k.device,
                    dtype=k.dtype
                )
                k_ref_scaled = k_ref * scale_vec

                # [NẠP VÀO BỘ NHỚ ĐỆM CỦA LAYER]
                self._cached_k_ref_scaled = k_ref_scaled.detach().clone()
                self._cached_v_ref = v_ref.detach().clone()

                k_ref_exp = k_ref_scaled.expand(target_b, *([-1] * (k_ref.ndim - 1)))
                v_ref_exp = v_ref.expand(target_b, *([-1] * (v_ref.ndim - 1)))

                k_tar_joint = torch.cat([k_tar, k_ref_exp], dim=1)
                v_tar_joint = torch.cat([v_tar, v_ref_exp], dim=1)

                out_tar = self.compute_attention(q_tar, k_tar_joint, v_tar_joint,
                                                 transformer_options=transformer_options)
                out_ref = self.compute_attention(q_ref, k_ref, v_ref, transformer_options=transformer_options)

                return torch.cat([out_tar, out_ref], dim=0)

            # GIAI ĐOẠN 2: Sử dụng bộ nhớ đệm Cache
            elif x.shape[0] == target_b and hasattr(self, '_cached_k_ref_scaled'):
                q, k_tar, v_tar = self.compute_qkv(x, context, rope_emb=rope_emb)

                # Lấy K_ref và V_ref từ Cache
                k_ref_scaled = self._cached_k_ref_scaled.to(device=k_tar.device, dtype=k_tar.dtype)
                v_ref = self._cached_v_ref.to(device=v_tar.device, dtype=v_tar.dtype)

                # [ĐOẠN CODE PHÁ VỠ GIỚI HẠN]: Áp dụng Fading Factor nếu được kích hoạt
                fade_factor = to.get('untwist_fade_factor', 1.0)
                if fade_factor < 1.0:
                    k_ref_scaled = k_ref_scaled * fade_factor

                k_ref_exp = k_ref_scaled.expand(target_b, *([-1] * (k_ref_scaled.ndim - 1)))
                v_ref_exp = v_ref.expand(target_b, *([-1] * (v_ref.ndim - 1)))

                k_tar_joint = torch.cat([k_tar, k_ref_exp], dim=1)
                v_tar_joint = torch.cat([v_tar, v_ref_exp], dim=1)

                out_tar = self.compute_attention(q, k_tar_joint, v_tar_joint, transformer_options=transformer_options)
                return out_tar

        return original_forward(x, *args, **kwargs)

    return patched_forward


def make_anima_model_wrapper(ref_latent, high_scale, low_scale, beta, adain_strength, guidance_ratio, enable_fading):
    def model_function_wrapper(apply_model, args):
        input_x = args['input']
        timestep = args['timestep']
        c = args['c'].copy()

        target_b = input_x.shape[0]

        if ref_latent is not None:
            device = input_x.device
            dtype = input_x.dtype

            ref_clean = ref_latent.to(device=device, dtype=dtype)

            if ref_clean.shape[-2:] != input_x.shape[-2:]:
                spatial_dims = ref_clean.ndim - 2
                ref_clean = torch.nn.functional.interpolate(
                    ref_clean,
                    size=input_x.shape[-spatial_dims:],
                    mode="trilinear" if spatial_dims == 3 else "bilinear",
                    align_corners=False
                )

            # Tính toán tiến trình chuẩn hóa (sigma từ 1.0 về 0.0)
            t_val = timestep.max().item()
            sigma = t_val / 1000.0 if t_val > 1.0 else t_val
            sigma = max(0.0, min(1.0, float(sigma)))

            cutoff_threshold = 1.0 - guidance_ratio
            is_active = (sigma > cutoff_threshold)

            # Tính toán hệ số fade_factor giảm dần mượt mà
            fade_factor = 1.0
            if not is_active and enable_fading:
                if cutoff_threshold > 0.0:
                    fade_factor = sigma / cutoff_threshold
                else:
                    fade_factor = 0.0
                fade_factor = max(0.0, min(1.0, float(fade_factor)))

            # QUYẾT ĐỊNH PHƯƠNG THỨC CHẠY:

            # THỂ TÍCH 1: Giai đoạn nạp Cache (Dual-batch active)
            if is_active:
                generator = torch.Generator(device=device).manual_seed(42)
                noise = torch.randn(ref_clean.shape, device=device, dtype=dtype, generator=generator)
                ref_noisy = (1.0 - sigma) * ref_clean + sigma * noise

                input_for_model = torch.cat([input_x, ref_noisy], dim=0)

                if timestep.ndim > 0 and timestep.shape[0] == target_b:
                    timestep_for_model = torch.cat([timestep, timestep[0:1]], dim=0)
                else:
                    timestep_for_model = timestep

                c_dict = c.copy()
                for k_cond, v_cond in c_dict.items():
                    if k_cond == 'transformer_options':
                        continue
                    if isinstance(v_cond, torch.Tensor) and v_cond.shape[0] == target_b:
                        c_dict[k_cond] = torch.cat([v_cond, v_cond[0:1]], dim=0)

                to = c_dict.get('transformer_options', {}).copy()
                to['untwist_rope_active'] = True
                to['untwist_target_b'] = target_b
                to['untwist_high_scale'] = high_scale
                to['untwist_low_scale'] = low_scale
                to['untwist_beta'] = beta
                to['untwist_adain'] = adain_strength
                to['untwist_fade_factor'] = 1.0
                c_dict['transformer_options'] = to

                raw_result = apply_model(input_for_model, timestep_for_model, **c_dict)
                return raw_result[:target_b]

            # THỂ TÍCH 2: Giai đoạn mờ dần Fading (Sử dụng Cache, tốc độ 1x)
            elif enable_fading and fade_factor > 0.0:
                to = c.get('transformer_options', {}).copy()
                to['untwist_rope_active'] = True
                to['untwist_target_b'] = target_b
                to['untwist_high_scale'] = high_scale
                to['untwist_low_scale'] = low_scale
                to['untwist_beta'] = beta
                to['untwist_adain'] = adain_strength
                to['untwist_fade_factor'] = fade_factor  # Bơm hệ số mờ dần vào Attention
                c['transformer_options'] = to

                return apply_model(input_x, timestep, **c)

            # THỂ TÍCH 3: Trả về tính toán gốc hoàn toàn (Khi tắt Fading hoặc t_norm đã lùi về sâu)
            else:
                return apply_model(input_x, timestep, **c)

        return apply_model(input_x, timestep, **c)

    return model_function_wrapper


class AnimaUntwistingRoPEProcessor(GuidanceProcessor):
    def name(self) -> str:
        return "Anima Untwisting RoPE"

    def create_ui(self):
        with gr.Tab(label="Anima Untwisting RoPE"):
            gr.Markdown("### Anima Untwisting RoPE (Zero-Shot Style Transfer)")
            enabled = gr.Checkbox(label="Enable Untwisting RoPE", value=False)
            ref_image = gr.Image(label="Style Reference Image", type="pil")

            with gr.Row():
                high_scale = gr.Slider(label="High Freq Scale", minimum=0.0, maximum=1.0, step=0.05, value=0.0)
                low_scale = gr.Slider(label="Low Freq Scale", minimum=0.5, maximum=3.0, step=0.1, value=1.0)

            with gr.Row():
                beta = gr.Slider(label="Interpolation Power (Beta)", minimum=1.0, maximum=50.0, step=1.0, value=2.0)
                adain_strength = gr.Slider(label="AdaIN Strength", minimum=0.0, maximum=1.0, step=0.05, value=0.5)

            with gr.Row():
                guidance_ratio = gr.Slider(label="Guidance Step Ratio", minimum=0.0, maximum=1.0, step=0.05, value=0.20)
                # THÊM CHECKBOX BẬT TẮT FADING:
                enable_fading = gr.Checkbox(label="Enable Attention Fading (Làm mờ dần bố cục mẫu)", value=True)

        return [enabled, ref_image, high_scale, low_scale, beta, adain_strength, guidance_ratio, enable_fading]

    def process(self, p, enabled, ref_image, high_scale, low_scale, beta, adain_strength, guidance_ratio,
                enable_fading):
        if not enabled:
            return

        ref_latent = None

        if ref_image is not None:
            logging.info("Anima Untwisting RoPE: Encoding reference style image via VAE")
            if hasattr(ref_image, 'convert'):
                ref_image = ref_image.convert("RGB")

            img_np = np.array(ref_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

            vae_obj = p.sd_model.forge_objects.vae
            img_tensor = img_tensor.to(device=vae_obj.device, dtype=vae_obj.vae_dtype)

            with torch.inference_mode():
                ref_latent = p.sd_model.encode_first_stage(img_tensor * 2.0 - 1.0).cpu()

            if torch.isnan(ref_latent).any():
                logging.error("Anima Untwisting RoPE: ERROR: VAE produced NaNs in latent!")
                return

        unet = p.sd_model.forge_objects.unet.clone()
        model = unet.model.diffusion_model

        patched_count = 0
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'SelfCrossAttention':
                if getattr(module, 'is_SelfAttn', False):
                    if not hasattr(module, '_original_forward'):
                        module._original_forward = module.forward

                    module.forward = types.MethodType(
                        make_patched_forward(module._original_forward),
                        module
                    )
                    patched_count += 1

        # Truyền tham số fading vào wrapper
        model_wrapper = make_anima_model_wrapper(
            ref_latent=ref_latent,
            high_scale=float(high_scale),
            low_scale=float(low_scale),
            beta=float(beta),
            adain_strength=float(adain_strength),
            guidance_ratio=float(guidance_ratio),
            enable_fading=bool(enable_fading)
        )
        unet.set_model_unet_function_wrapper(model_wrapper)

        p.sd_model.forge_objects.unet = unet
        logging.info(f"Anima Untwisting RoPE: Active. Patched {patched_count} blocks.")

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        pass


register_processor(AnimaUntwistingRoPEProcessor)