import logging
import torch
import gradio as gr
from functools import partial
import types
import numpy as np

from ..base import GuidanceProcessor
from ..registry import register_processor

# We assume asag_utils is available in the root
try:
    from asag_utils import asag_attention, rescale_guidance, snf_guidance, set_model_options_patch_replace
except ImportError:
    # If not found, we'll try to import from the root path.
    import sys
    import os

    # processors -> guidance_pack -> scripts -> root
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if root_path not in sys.path:
        sys.path.append(root_path)
    try:
        from asag_utils import asag_attention, rescale_guidance, snf_guidance, set_model_options_patch_replace
    except ImportError:
        # Try one more level up just in case
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        if root_path not in sys.path:
            sys.path.append(root_path)
        from asag_utils import asag_attention, rescale_guidance, snf_guidance, set_model_options_patch_replace

try:
    from yx_guidance_utils import ensure_guidance_pipeline, GuidanceState
except ImportError:
    import sys
    import os

    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from yx_guidance_utils import ensure_guidance_pipeline, GuidanceState


def make_asag_modifier(scale, sinkhorn_iters, final_blocks, sigma_start_val, sigma_end, rescale, rescale_mode):
    asag_marker = "asag_active"

    def modifier(args, state: GuidanceState) -> GuidanceState:
        model_patcher = args["model"]
        cond_pred = args["cond_denoised"]
        cond = args["cond"]
        sigma = args["sigma"]
        x = args["input"]

        # current_prediction is the result from previous modifiers/base_builder
        current_prediction = state.prediction

        if scale == 0 or not (sigma_end < sigma[0] <= sigma_start_val):
            return state

        model_options = args["model_options"].copy()
        if "transformer_options" not in model_options:
            model_options["transformer_options"] = {}
        else:
            model_options["transformer_options"] = model_options["transformer_options"].copy()

        model_options["transformer_options"]["asag_sinkhorn_iters"] = int(sinkhorn_iters)

        for block_idx in final_blocks:
            model_options = set_model_options_patch_replace(model_options, asag_marker, "attn1", "blocks", block_idx)

        try:
            from backend.sampling.sampling_function import calc_cond_uncond_batch as forge_calc
        except ImportError:
            from ldm_patched.modules.samplers import calc_cond_uncond_batch as forge_calc

        (asag_cond_pred, _) = forge_calc(model_patcher, cond, None, x, sigma, model_options)

        # --- STABILIZED GUIDANCE CALCULATION ---
        asag_guidance = (cond_pred - asag_cond_pred) * scale
        asag_guidance = torch.nan_to_num(asag_guidance, nan=0.0)

        # Magnitude Limiter: Ensure guidance doesn't explode
        mag_cond = torch.std(cond_pred, dim=tuple(range(1, cond_pred.ndim)), keepdim=True) + 1e-6
        mag_guid = torch.std(asag_guidance, dim=tuple(range(1, asag_guidance.ndim)), keepdim=True) + 1e-6

        # Clamp scale to safe level
        safe_scale = torch.clamp(mag_cond * 2.0 / mag_guid, max=1.0)
        asag_guidance = asag_guidance * safe_scale

        # --- 5D-AWARE RESCALE ---
        if rescale > 0:
            target_result = current_prediction + asag_guidance if rescale_mode == "full" else cond_pred + asag_guidance
            std_cond = torch.std(cond_pred, dim=tuple(range(1, cond_pred.ndim)), keepdim=True)
            std_target = torch.std(target_result, dim=tuple(range(1, target_result.ndim)), keepdim=True)
            r_factor = rescale * (std_cond / (std_target + 1e-8)) + (1.0 - rescale)
            asag_guidance = asag_guidance * r_factor

        # Accumulate ASAG guidance into the existing guidance term
        return GuidanceState(state.base_prediction, state.guidance_term + asag_guidance)

    return modifier


class AnimaASAGProcessor(GuidanceProcessor):
    def name(self) -> str:
        return "Anima ASAG"

    def create_ui(self):
        with gr.Tab(label="Anima ASAG"):
            gr.Markdown("### Anima ASAG\nAdversarial Sinkhorn Attention Guidance for Anima models.")
            enabled = gr.Checkbox(label="Enable Anima ASAG", value=False)
            scale = gr.Slider(label="ASAG Scale", minimum=0.0, maximum=10.0, step=0.1, value=1.5)
            sinkhorn_iters = gr.Slider(label="Sinkhorn Iterations", minimum=1, maximum=8, step=1, value=2)

            blocks_list = gr.Textbox(label="Anima Blocks (e.g. 0-9 or 0,2,4)", value="0-9")

            with gr.Row():
                sigma_start = gr.Slider(label="Sigma Start (-1 for auto)", minimum=-1.0, maximum=1000.0, step=0.01,
                                        value=-1.0)
                sigma_end = gr.Slider(label="Sigma End (-1 for auto)", minimum=-1.0, maximum=1000.0, step=0.01,
                                      value=-1.0)

            with gr.Row():
                rescale = gr.Slider(label="Rescale Factor", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                rescale_mode = gr.Dropdown(label="Rescale Mode", choices=["full", "partial", "snf"], value="full")

        return [enabled, scale, sinkhorn_iters, blocks_list, sigma_start, sigma_end, rescale, rescale_mode]

    def process(self, p, enabled, scale, sinkhorn_iters, blocks_list, sigma_start, sigma_end, rescale, rescale_mode):
        if not enabled:
            return

        unet_patcher = p.sd_model.forge_objects.unet
        model = unet_patcher.model.diffusion_model

        if model.__class__.__name__ != "Anima":
            logging.warning("Anima ASAG: Current model is not Anima. Skipping.")
            return

        self.ensure_anima_patched_for_asag(model)
        total_blocks = len(model.blocks)
        final_blocks = self.parse_anima_blocks(blocks_list, total_blocks)

        sigma_start_val = float("inf") if sigma_start < 0 else sigma_start

        pipeline = ensure_guidance_pipeline(unet_patcher)
        pipeline.add_modifier(
            "asag",
            make_asag_modifier(
                scale=scale,
                sinkhorn_iters=sinkhorn_iters,
                final_blocks=final_blocks,
                sigma_start_val=sigma_start_val,
                sigma_end=sigma_end,
                rescale=rescale,
                rescale_mode=rescale_mode
            )
        )
        logging.info(f"Anima ASAG: Integrated into Pipeline. Scale={scale}")

    def ensure_anima_patched_for_asag(self, model):
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'SelfCrossAttention' and getattr(module, 'is_SelfAttn', False):
                if not hasattr(module, '_asag_patched'):
                    parts = name.split('.')
                    if len(parts) >= 2 and parts[0] == 'blocks':
                        try:
                            block_idx = int(parts[1])
                        except ValueError:
                            continue

                        orig_compute_attention = module.compute_attention

                        def make_patched_compute_attention(idx, original_func):
                            def patched_compute_attention(this, q, k, v, transformer_options={}):
                                patches_cfg = transformer_options.get("patches_replace", {}).get("attn1", {})
                                if ("blocks", idx) in patches_cfg:

                                    # LẤY ĐỘ DÀI CHUỖI ĐỂ KIỂM TRA
                                    s_q = q.shape[1]
                                    s_kv = k.shape[1]

                                    # [BẢO VỆ BĂNG THÔNG T4]: Nếu đang chạy Dual-Batch nạp cache của RoPE (s_kv > s_q),
                                    # bỏ qua ASAG để tránh nổ dung lượng ma trận lên 4.5GB.
                                    if s_kv > s_q:
                                        return original_func(q, k, v, transformer_options=transformer_options)

                                    # --- PHẦN TOÁN HỌC ASAG CHẠY CHO CÁC BƯỚC SAU (Khi s_kv == s_q) ---
                                    orig_dtype = q.dtype

                                    q = torch.nan_to_num(q, nan=0.0).clamp(-64.0, 64.0).float()
                                    k = torch.nan_to_num(k, nan=0.0).clamp(-64.0, 64.0).float()
                                    v_f = torch.nan_to_num(v, nan=0.0).clamp(-128.0, 128.0).float()

                                    b, s_q, h, d = q.shape
                                    scale_f = 1.0 / (d ** 0.5)

                                    q_t = q.transpose(1, 2)
                                    k_t = k.transpose(1, 2)

                                    sim = torch.matmul(q_t, k_t.transpose(-1, -2))
                                    logits = torch.clamp(-scale_f * sim, min=-20.0, max=20.0)
                                    logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
                                    K = torch.exp(logits - logits_max)

                                    iters = transformer_options.get("asag_sinkhorn_iters", 2)
                                    u = torch.full((b, h, s_q, 1), 1.0 / s_q, device=q.device, dtype=torch.float32)
                                    v_v = torch.full((b, h, s_q, 1), 1.0 / s_q, device=q.device, dtype=torch.float32)

                                    for _ in range(iters):
                                        Kv = torch.matmul(K, v_v).clamp_min(1e-12)
                                        u = (1.0 / s_q) / Kv
                                        KTu = torch.matmul(K.transpose(-1, -2), u).clamp_min(1e-12)
                                        v_v = (1.0 / s_q) / KTu

                                    P = K * u * v_v.transpose(-1, -2)
                                    v_t = v_f.transpose(1, 2)
                                    out = torch.matmul(P, v_t)

                                    out = out.transpose(1, 2).reshape(b, s_q, h * d).to(orig_dtype)
                                    out = torch.nan_to_num(out, nan=0.0).clamp(-128.0, 128.0)

                                    return this.output_dropout(this.output_proj(out))

                                return original_func(q, k, v, transformer_options=transformer_options)

                            return patched_compute_attention

                        module.compute_attention = types.MethodType(
                            make_patched_compute_attention(block_idx, orig_compute_attention), module
                        )
                        module._asag_patched = True

    def parse_anima_blocks(self, blocks_str, total_blocks):
        blocks = []
        for part in blocks_str.split(','):
            part = part.strip()
            if '-' in part:
                try:
                    s_e = part.split('-')
                    start = int(s_e[0])
                    end = int(s_e[1])
                    blocks.extend(range(start, min(end + 1, total_blocks)))
                except (ValueError, IndexError):
                    pass
            else:
                try:
                    idx = int(part)
                    if 0 <= idx < total_blocks:
                        blocks.append(idx)
                except ValueError:
                    pass
        if not blocks:
            blocks = list(range(total_blocks))
        return sorted(list(set(blocks)))

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        pass


register_processor(AnimaASAGProcessor)