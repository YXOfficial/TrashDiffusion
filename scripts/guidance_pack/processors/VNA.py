import logging
import torch
import gradio as gr
import types

from ..base import GuidanceProcessor
from ..registry import register_processor

try:
    from yx_guidance_utils import ensure_guidance_pipeline, GuidanceState
except ImportError:
    import sys
    import os

    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from yx_guidance_utils import ensure_guidance_pipeline, GuidanceState


def make_vna_modifier(scale, final_blocks, sigma_start_val, sigma_end, rescale):
    vna_marker = "vna_active"

    def modifier(args, state: GuidanceState) -> GuidanceState:
        model_patcher = args["model"]
        cond_pred = args["cond_denoised"]
        cond = args["cond"]
        sigma = args["sigma"]
        x = args["input"]

        current_prediction = state.prediction

        if scale == 0 or not (sigma_end < sigma[0] <= sigma_start_val):
            return state

        model_options = args["model_options"].copy()
        if "transformer_options" not in model_options:
            model_options["transformer_options"] = {}
        else:
            model_options["transformer_options"] = model_options["transformer_options"].copy()

        for block_idx in final_blocks:
            from asag_utils import set_model_options_patch_replace
            model_options = set_model_options_patch_replace(model_options, vna_marker, "attn1", "blocks", block_idx)

        try:
            from backend.sampling.sampling_function import calc_cond_uncond_batch as forge_calc
        except ImportError:
            from ldm_patched.modules.samplers import calc_cond_uncond_batch as forge_calc

        (vna_cond_pred, _) = forge_calc(model_patcher, cond, None, x, sigma, model_options)

        vna_guidance = (cond_pred - vna_cond_pred) * scale
        vna_guidance = torch.nan_to_num(vna_guidance, nan=0.0)

        if rescale > 0:
            target_result = current_prediction + vna_guidance
            std_cond = torch.std(cond_pred, dim=tuple(range(1, cond_pred.ndim)), keepdim=True)
            std_target = torch.std(target_result, dim=tuple(range(1, target_result.ndim)), keepdim=True)
            r_factor = rescale * (std_cond / (std_target + 1e-8)) + (1.0 - rescale)
            vna_guidance = vna_guidance * r_factor

        return GuidanceState(state.base_prediction, state.guidance_term + vna_guidance)

    return modifier


class AnimaVNAProcessor(GuidanceProcessor):
    def name(self) -> str:
        return "Anima VNA (Contrast Guidance)"

    def create_ui(self):
        with gr.Tab(label="Anima VNA"):
            gr.Markdown(
                "### Anima VNA\nVelocity Norm Alignment")
            enabled = gr.Checkbox(label="Enable Anima VNA", value=False)
            scale = gr.Slider(label="Contrast Scale", minimum=0.0, maximum=10.0, step=0.1, value=3)
            blocks_list = gr.Textbox(label="Anima Blocks (e.g. 0-9)", value="9")

            with gr.Row():
                sigma_start = gr.Slider(label="Sigma Start", minimum=-1.0, maximum=1000.0, step=0.01, value=-1.0)
                sigma_end = gr.Slider(label="Sigma End", minimum=-1.0, maximum=1000.0, step=0.01, value=-1.0)

            rescale = gr.Slider(label="Rescale Factor (Anti-Saturate)", minimum=0.0, maximum=1.0, step=0.01, value=0)

        return [enabled, scale, blocks_list, sigma_start, sigma_end, rescale]

    def process(self, p, enabled, scale, blocks_list, sigma_start, sigma_end, rescale):
        xyz_settings = getattr(p, "_guidance_xyz", {})
        vna_xyz = xyz_settings.get("vna", {})
        if "enabled" in vna_xyz:
            enabled = str(vna_xyz["enabled"]).lower() == "true"
        if "scale" in vna_xyz:
            scale = float(vna_xyz["scale"])
        if "blocks" in vna_xyz:
            blocks_list = str(vna_xyz["blocks"])
        if "sigma_start" in vna_xyz:
            sigma_start = float(vna_xyz["sigma_start"])
        if "sigma_end" in vna_xyz:
            sigma_end = float(vna_xyz["sigma_end"])
        if "rescale" in vna_xyz:
            rescale = float(vna_xyz["rescale"])

        if not enabled:
            return

        unet_patcher = p.sd_model.forge_objects.unet
        model = unet_patcher.model.diffusion_model

        if model.__class__.__name__ != "Anima":
            logging.warning("Anima VNA: Current model is not Anima. Skipping.")
            return

        self.ensure_anima_patched_for_vna(model)
        total_blocks = len(model.blocks)
        final_blocks = self.parse_anima_blocks(blocks_list, total_blocks)

        sigma_start_val = float("inf") if sigma_start < 0 else sigma_start

        pipeline = ensure_guidance_pipeline(unet_patcher)
        pipeline.add_modifier(
            "vna",
            make_vna_modifier(
                scale=scale,
                final_blocks=final_blocks,
                sigma_start_val=sigma_start_val,
                sigma_end=sigma_end,
                rescale=rescale
            )
        )
        logging.info(f"Anima VNA: Active on blocks {final_blocks}. Scale={scale}")

    def ensure_anima_patched_for_vna(self, model):
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'SelfCrossAttention' and getattr(module, 'is_SelfAttn', False):
                if not hasattr(module, '_vna_patched'):
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
                                    # [1, 4096, 16, 128]
                                    b, s, h, d = v.shape
                                    orig_dtype = v.dtype
                                    # 1. Velocity Norm
                                    v_norm = v.norm(dim=-1, keepdim=True)
                                    # 2. Refine Velocity Field
                                    # This is where i set Sharpening/Fidelity.
                                    scale_factor = 1.15  # Hardcoded value, must be bad
                                    v_refined = v * (v_norm.clamp(min=1e-6) ** (scale_factor - 1.0))

                                    # 3. Merge Heads to model_channels (2048)
                                    # [b, s, h, d] -> [b, s, h*d]
                                    out = v_refined.reshape(b, s, h * d).to(orig_dtype)

                                    # 4. Linear Projection
                                    return this.output_dropout(this.output_proj(out))

                                return original_func(q, k, v, transformer_options=transformer_options)

                            return patched_compute_attention

                        module.compute_attention = types.MethodType(
                            make_patched_compute_attention(block_idx, orig_compute_attention), module
                        )
                        module._vna_patched = True

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
        from functools import partial
        options = [
            xyz_grid.AxisOption(
                label="(Anima VNA) Enabled",
                type=str,
                apply=partial(set_guidance_value_func, feature="vna", field="enabled"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(label="(Anima VNA) Scale", type=float, apply=partial(set_guidance_value_func, feature="vna", field="scale")),
            xyz_grid.AxisOption(label="(Anima VNA) Blocks", type=str, apply=partial(set_guidance_value_func, feature="vna", field="blocks")),
            xyz_grid.AxisOption(label="(Anima VNA) Sigma Start", type=float, apply=partial(set_guidance_value_func, feature="vna", field="sigma_start")),
            xyz_grid.AxisOption(label="(Anima VNA) Sigma End", type=float, apply=partial(set_guidance_value_func, feature="vna", field="sigma_end")),
            xyz_grid.AxisOption(label="(Anima VNA) Rescale", type=float, apply=partial(set_guidance_value_func, feature="vna", field="rescale")),
        ]
        xyz_grid.axis_options.extend(options)


register_processor(AnimaVNAProcessor)