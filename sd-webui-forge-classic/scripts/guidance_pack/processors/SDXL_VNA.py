import logging
import torch
import gradio as gr
from ..base import GuidanceProcessor
from ..registry import register_processor
from core.forge_compat import get_unet
from core.guidance import ensure_guidance_pipeline, GuidanceState

def vna_attn_patch(q, k, v, extra_options):
    """
    VNA Logic: Align velocity norm in self-attention.
    Bypasses traditional attention computation for a refined velocity field.
    """
    heads = extra_options.get("n_heads", 8)
    dim_head = extra_options.get("dim_head", 64)
    b, s, c = v.shape
    
    # Reshape to [batch, sequence, heads, dim_head]
    v = v.view(b, s, heads, dim_head)
    orig_dtype = v.dtype
    
    # 1. Calculate Velocity Norm
    v_norm = v.norm(dim=-1, keepdim=True)
    
    # 2. Refine Velocity Field (Sharpening/Contrast alignment)
    # 1.15 is the standard scale factor for VNA effects
    scale_factor = 1.15 
    v_refined = v * (v_norm.clamp(min=1e-6) ** (scale_factor - 1.0))
    
    # 3. Merge Heads back to original channel dimension
    out = v_refined.reshape(b, s, c).to(orig_dtype)
    return out

def make_sdxl_vna_modifier(scale, target_blocks, sigma_start_val, sigma_end, rescale):
    def modifier(args, state: GuidanceState) -> GuidanceState:
        model_patcher = args["model"]
        cond_pred = args["cond_denoised"]
        cond = args["cond"]
        sigma = args["sigma"]
        x = args["input"]

        if scale == 0 or not (sigma_end < sigma[0] <= sigma_start_val):
            return state

        model_options = args["model_options"].copy()
        if "transformer_options" not in model_options:
            model_options["transformer_options"] = {}
        else:
            model_options["transformer_options"] = model_options["transformer_options"].copy()

        # Apply VNA patch to selected SDXL blocks via patches_replace
        from core.asag import set_model_options_patch_replace
        for block_type, block_id, transformer_idx in target_blocks:
            model_options = set_model_options_patch_replace(
                model_options, 
                vna_attn_patch, 
                "attn1", 
                block_type, 
                block_id, 
                transformer_idx
            )

        try:
            from backend.sampling.sampling_function import calc_cond_uncond_batch as forge_calc
        except ImportError:
            from ldm_patched.modules.samplers import calc_cond_uncond_batch as forge_calc

        # Calculate VNA-patched prediction
        (vna_cond_pred, _) = forge_calc(model_patcher, cond, None, x, sigma, model_options)

        # Apply contrast guidance
        vna_guidance = (cond_pred - vna_cond_pred) * scale
        vna_guidance = torch.nan_to_num(vna_guidance, nan=0.0)

        if rescale > 0:
            current_prediction = state.prediction
            target_result = current_prediction + vna_guidance
            std_cond = torch.std(cond_pred, dim=tuple(range(1, cond_pred.ndim)), keepdim=True)
            std_target = torch.std(target_result, dim=tuple(range(1, target_result.ndim)), keepdim=True)
            r_factor = rescale * (std_cond / (std_target + 1e-8)) + (1.0 - rescale)
            vna_guidance = vna_guidance * r_factor

        return GuidanceState(state.base_prediction, state.guidance_term + vna_guidance)

    return modifier

class SDXLVNAProcessor(GuidanceProcessor):
    def name(self) -> str:
        return "SDXL VNA"

    def create_ui(self):
        with gr.Tab(label="SDXL VNA"):
            gr.Markdown("### SDXL VNA\nVelocity Norm Alignment specifically optimized for SDXL structure.")
            enabled = gr.Checkbox(label="Enable SDXL VNA", value=False)
            scale = gr.Slider(label="Contrast Scale", minimum=0.0, maximum=10.0, step=0.1, value=3.0)
            blocks_list = gr.Textbox(label="SDXL Blocks (d=down/input, m=mid, u=up/output)", value="d4-d8, m0, u0-u5")

            with gr.Row():
                sigma_start = gr.Slider(label="Sigma Start", minimum=-1.0, maximum=1000.0, step=0.01, value=-1.0)
                sigma_end = gr.Slider(label="Sigma End", minimum=-1.0, maximum=1000.0, step=0.01, value=-1.0)

            rescale = gr.Slider(label="Rescale Factor", minimum=0.0, maximum=1.0, step=0.01, value=0.0)

        return [enabled, scale, blocks_list, sigma_start, sigma_end, rescale]

    def infotext_fields(self):
        return [
            "SDXL VNA Enabled",
            "SDXL VNA scale",
            "SDXL VNA blocks",
            "SDXL VNA sigma start",
            "SDXL VNA sigma end",
            "SDXL VNA rescale",
        ]

    def process(self, p, enabled, scale, blocks_list, sigma_start, sigma_end, rescale):
        xyz_settings = getattr(p, "_guidance_xyz", {})
        sdxl_vna_xyz = xyz_settings.get("sdxl_vna", {})
        if "enabled" in sdxl_vna_xyz:
            enabled = str(sdxl_vna_xyz["enabled"]).lower() == "true"
        if "scale" in sdxl_vna_xyz:
            scale = float(sdxl_vna_xyz["scale"])
        if "blocks" in sdxl_vna_xyz:
            blocks_list = str(sdxl_vna_xyz["blocks"])
        if "sigma_start" in sdxl_vna_xyz:
            sigma_start = float(sdxl_vna_xyz["sigma_start"])
        if "sigma_end" in sdxl_vna_xyz:
            sigma_end = float(sdxl_vna_xyz["sigma_end"])
        if "rescale" in sdxl_vna_xyz:
            rescale = float(sdxl_vna_xyz["rescale"])

        if not enabled: return

        self.record_params(p, {
            "SDXL VNA Enabled": True,
            "SDXL VNA scale": float(scale),
            "SDXL VNA blocks": str(blocks_list),
            "SDXL VNA sigma start": float(sigma_start),
            "SDXL VNA sigma end": float(sigma_end),
            "SDXL VNA rescale": float(rescale),
        })

        unet_patcher = get_unet(p)
        
        # Identify SDXL blocks within the IntegratedUNet architecture
        target_blocks = self.parse_sdxl_blocks(blocks_list, unet_patcher.model.diffusion_model)
        if not target_blocks:
            logging.warning(f"SDXL VNA: No compatible transformer blocks found for: {blocks_list}")
            return

        sigma_start_val = float("inf") if sigma_start < 0 else sigma_start

        pipeline = ensure_guidance_pipeline(unet_patcher)
        pipeline.add_modifier(
            "sdxl_vna",
            make_sdxl_vna_modifier(
                scale=scale,
                target_blocks=target_blocks,
                sigma_start_val=sigma_start_val,
                sigma_end=sigma_end,
                rescale=rescale
            )
        )
        logging.info(f"SDXL VNA: Active on {len(target_blocks)} sub-blocks. Scale={scale}")

    def parse_sdxl_blocks(self, blocks_str, model):
        """
        Parses strings like 'd4-d8, m0, u0-u5' into Forge-compatible block identifiers.
        """
        import re
        target_blocks = []
        available_map = {}
        
        for name, module in model.named_modules():
            if module.__class__.__name__ == "BasicTransformerBlock":
                parts = name.split('.')
                # input_blocks.4.1.transformer_blocks.0 -> parts length 5
                # middle_block.1.transformer_blocks.0 -> parts length 4
                if len(parts) >= 4:
                    b_type_raw = parts[0].split('_')[0] 
                    b_type = b_type_raw
                    
                    try:
                        # For middle, it's always id 0 in user terms, but index 1 in Forge
                        b_id = 0 if b_type == "middle" else int(parts[1])
                        t_idx = int(parts[-1])
                        key = (b_type, b_id)
                        if key not in available_map: available_map[key] = []
                        available_map[key].append(t_idx)
                    except: continue

        logging.info(f"SDXL VNA: Found available transformer blocks: {list(available_map.keys())}")

        for part in [p.strip() for p in blocks_str.split(',')]:
            if not part: continue
            prefix = part[0].lower()
            target_type = {"d": "input", "m": "middle", "u": "output"}.get(prefix)
            if not target_type: continue
            
            # Extract numbers even if they contain prefixes (e.g., '4-d8' -> [4, 8])
            nums = re.findall(r'\d+', part)
            if not nums: continue
            
            try:
                if '-' in part and len(nums) >= 2:
                    s, e = int(nums[0]), int(nums[1])
                    ids = range(s, e + 1)
                else:
                    ids = [int(nums[0])]
            except: continue
                
            for bid in ids:
                if (target_type, bid) in available_map:
                    for t_idx in available_map[(target_type, bid)]:
                        target_blocks.append((target_type, bid, t_idx))
        
        if target_blocks:
            logging.info(f"SDXL VNA: Selected {len(target_blocks)} transformer blocks.")
        return target_blocks

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        from functools import partial
        options = [
            xyz_grid.AxisOption(
                label="(SDXL VNA) Enabled",
                type=str,
                apply=partial(set_guidance_value_func, feature="sdxl_vna", field="enabled"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(label="(SDXL VNA) Scale", type=float, apply=partial(set_guidance_value_func, feature="sdxl_vna", field="scale")),
            xyz_grid.AxisOption(label="(SDXL VNA) Blocks", type=str, apply=partial(set_guidance_value_func, feature="sdxl_vna", field="blocks")),
            xyz_grid.AxisOption(label="(SDXL VNA) Sigma Start", type=float, apply=partial(set_guidance_value_func, feature="sdxl_vna", field="sigma_start")),
            xyz_grid.AxisOption(label="(SDXL VNA) Sigma End", type=float, apply=partial(set_guidance_value_func, feature="sdxl_vna", field="sigma_end")),
            xyz_grid.AxisOption(label="(SDXL VNA) Rescale", type=float, apply=partial(set_guidance_value_func, feature="sdxl_vna", field="rescale")),
        ]
        xyz_grid.axis_options.extend(options)

register_processor(SDXLVNAProcessor)
