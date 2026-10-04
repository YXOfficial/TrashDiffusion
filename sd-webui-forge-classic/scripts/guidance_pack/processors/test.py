import logging
import torch
import gradio as gr
import types

from ..base import GuidanceProcessor
from ..registry import register_processor
from core.forge_compat import get_unet
from core.guidance import ensure_guidance_pipeline, GuidanceState


def make_attn2_probe_modifier(k_scale, v_scale, final_blocks, sigma_start_val, sigma_end):
    probe_marker = "attn2_probe_active"

    def modifier(args, state: GuidanceState) -> GuidanceState:
        model_patcher = args["model"]
        cond_pred = args["cond_denoised"]
        cond = args["cond"]
        sigma = args["sigma"]
        x = args["input"]

        # No-op probe returns the input unchanged
        if (k_scale == 1.0 and v_scale == 1.0) or not (sigma_end < sigma[0] <= sigma_start_val):
            return state

        model_options = args["model_options"].copy()
        if "transformer_options" not in model_options:
            model_options["transformer_options"] = {}
        else:
            model_options["transformer_options"] = model_options["transformer_options"].copy()

        # Pass probe values through transformer kwargs
        model_options["transformer_options"]["probe_k_scale"] = k_scale
        model_options["transformer_options"]["probe_v_scale"] = v_scale

        for block_idx in final_blocks:
            from core.asag import set_model_options_patch_replace
            # ONLY target attn2
            model_options = set_model_options_patch_replace(model_options, probe_marker, "attn2", "blocks", block_idx)

        try:
            from backend.sampling.sampling_function import calc_cond_uncond_batch as forge_calc
        except ImportError:
            from ldm_patched.modules.samplers import calc_cond_uncond_batch as forge_calc

        # Second pass with modified attn2
        (probe_cond_pred, _) = forge_calc(model_patcher, cond, None, x, sigma, model_options)

        # Delta from the probe, added straight into the prediction
        probe_guidance = (probe_cond_pred - cond_pred)
        probe_guidance = torch.nan_to_num(probe_guidance, nan=0.0)

        return GuidanceState(state.base_prediction, state.guidance_term + probe_guidance)

    return modifier


class AnimaAttn2ProbeProcessor(GuidanceProcessor):
    def name(self) -> str:
        return "Anima Attn2 Probe (Text Injection)"

    def create_ui(self):
        with gr.Tab(label="Attn2 Probe"):
            gr.Markdown(
                "### Anima Attn2 Probe\nText -> Image matrix probe. No VNA math, only K/V intensity.")
            enabled = gr.Checkbox(label="Enable Attn2 Probe", value=False)

            # Safe default is 1.0 (no change)
            k_scale = gr.Slider(label="K Scale (Focus / Temp)", minimum=0.0, maximum=5.0, step=0.05, value=1.0)
            v_scale = gr.Slider(label="V Scale (Volume / Intensity)", minimum=0.0, maximum=5.0, step=0.05, value=1.0)

            blocks_list = gr.Textbox(label="Anima Blocks (e.g. 0-27)", value="0-27")

            with gr.Row():
                sigma_start = gr.Slider(label="Sigma Start", minimum=-1.0, maximum=1000.0, step=0.01, value=-1.0)
                sigma_end = gr.Slider(label="Sigma End", minimum=-1.0, maximum=1000.0, step=0.01, value=-1.0)

        return [enabled, k_scale, v_scale, blocks_list, sigma_start, sigma_end]

    def infotext_fields(self):
        return [
            "Attn2 Probe Enabled",
            "Attn2 Probe K",
            "Attn2 Probe V",
            "Attn2 Probe blocks",
            "Attn2 Probe sigma start",
            "Attn2 Probe sigma end",
        ]

    def process(self, p, enabled, k_scale, v_scale, blocks_list, sigma_start, sigma_end):
        xyz_settings = getattr(p, "_guidance_xyz", {})
        probe_xyz = xyz_settings.get("attn2_probe", {})
        if "enabled" in probe_xyz:
            enabled = str(probe_xyz["enabled"]).lower() == "true"
        if "k_scale" in probe_xyz:
            k_scale = float(probe_xyz["k_scale"])
        if "v_scale" in probe_xyz:
            v_scale = float(probe_xyz["v_scale"])
        if "blocks" in probe_xyz:
            blocks_list = str(probe_xyz["blocks"])
        if "sigma_start" in probe_xyz:
            sigma_start = float(probe_xyz["sigma_start"])
        if "sigma_end" in probe_xyz:
            sigma_end = float(probe_xyz["sigma_end"])

        if not enabled:
            return

        self.record_params(p, {
            "Attn2 Probe Enabled": True,
            "Attn2 Probe K": float(k_scale),
            "Attn2 Probe V": float(v_scale),
            "Attn2 Probe blocks": str(blocks_list),
            "Attn2 Probe sigma start": float(sigma_start),
            "Attn2 Probe sigma end": float(sigma_end),
        })

        unet_patcher = get_unet(p)
        model = unet_patcher.model.diffusion_model

        if model.__class__.__name__ != "Anima":
            logging.warning("Attn2 Probe: Current model is not Anima. Skipping.")
            return

        self.ensure_anima_patched_for_probe(model)
        total_blocks = len(model.blocks)
        final_blocks = self.parse_anima_blocks(blocks_list, total_blocks)

        sigma_start_val = float("inf") if sigma_start < 0 else sigma_start

        pipeline = ensure_guidance_pipeline(unet_patcher)
        pipeline.add_modifier(
            "attn2_probe",
            make_attn2_probe_modifier(
                k_scale=k_scale,
                v_scale=v_scale,
                final_blocks=final_blocks,
                sigma_start_val=sigma_start_val,
                sigma_end=sigma_end
            )
        )
        logging.info(f"Attn2 Probe: Active on blocks {final_blocks}. K={k_scale}, V={v_scale}")

    def ensure_anima_patched_for_probe(self, model):
        for name, module in model.named_modules():
            # CRITICAL: only patch Cross-Attention modules (where is_SelfAttn = False)
            if module.__class__.__name__ == 'SelfCrossAttention' and getattr(module, 'is_SelfAttn', False) == False:
                if not hasattr(module, '_attn2_probe_patched'):
                    parts = name.split('.')
                    if len(parts) >= 2 and parts[0] == 'blocks':
                        try:
                            block_idx = int(parts[1])
                        except ValueError:
                            continue

                        orig_compute_attention = module.compute_attention

                        def make_patched_compute_attention(idx, original_func):
                            def patched_compute_attention(this, q, k, v, transformer_options={}):
                                patches_cfg = transformer_options.get("patches_replace", {}).get("attn2", {})
                                if ("blocks", idx) in patches_cfg:
                                    # Read probe params from kwargs, default 1.0
                                    k_scale_val = transformer_options.get("probe_k_scale", 1.0)
                                    v_scale_val = transformer_options.get("probe_v_scale", 1.0)

                                    # SIMPLIFIED: scalar multiply only. No norm, no exponents.
                                    # K controls whether text aims at the right pixels (Focus)
                                    k_probed = k * k_scale_val

                                    # V controls how much text signal each pixel receives (Intensity)
                                    v_probed = v * v_scale_val

                                    # Keep the tensor flow intact, forward into Comfy's original attention
                                    return original_func(q, k_probed, v_probed, transformer_options=transformer_options)

                                return original_func(q, k, v, transformer_options=transformer_options)

                            return patched_compute_attention

                        module.compute_attention = types.MethodType(
                            make_patched_compute_attention(block_idx, orig_compute_attention), module
                        )
                        module._attn2_probe_patched = True

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
                label="(Attn2 Probe) Enabled",
                type=str,
                apply=partial(set_guidance_value_func, feature="attn2_probe", field="enabled"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(
                label="(Attn2 Probe) K Scale",
                type=float,
                apply=partial(set_guidance_value_func, feature="attn2_probe", field="k_scale"),
            ),
            xyz_grid.AxisOption(
                label="(Attn2 Probe) V Scale",
                type=float,
                apply=partial(set_guidance_value_func, feature="attn2_probe", field="v_scale"),
            ),
            xyz_grid.AxisOption(
                label="(Attn2 Probe) Blocks",
                type=str,
                apply=partial(set_guidance_value_func, feature="attn2_probe", field="blocks"),
            ),
            xyz_grid.AxisOption(
                label="(Attn2 Probe) Sigma Start",
                type=float,
                apply=partial(set_guidance_value_func, feature="attn2_probe", field="sigma_start"),
            ),
            xyz_grid.AxisOption(
                label="(Attn2 Probe) Sigma End",
                type=float,
                apply=partial(set_guidance_value_func, feature="attn2_probe", field="sigma_end"),
            ),
        ]
        xyz_grid.axis_options.extend(options)


register_processor(AnimaAttn2ProbeProcessor)