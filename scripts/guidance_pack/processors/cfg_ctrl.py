import logging
import gradio as gr
from functools import partial
from ..base import GuidanceProcessor
from ..registry import register_processor

try:
    from yx_guidance_utils import ensure_guidance_pipeline, get_initial_sigma, make_cfg_ctrl_base_builder
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
    from yx_guidance_utils import ensure_guidance_pipeline, get_initial_sigma, make_cfg_ctrl_base_builder


class CFGCtrlProcessor(GuidanceProcessor):
    def __init__(self):
        self.cfg_ctrl_enabled = False
        self.smc_cfg_enable = False
        self.smc_cfg_lambda = 0.05
        self.smc_cfg_K = 0.3
        self.no_cfg_warmup_steps = 0

    def name(self) -> str:
        return "CFG-Ctrl"

    def create_ui(self):
        with gr.Tab(label="CFG-Ctrl"):
            gr.Markdown("CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance (CVPR 2026)")
            gr.Markdown("SMC-CFG adds nonlinear feedback stabilization for better semantic alignment.")
            
            cfg_ctrl_enabled = gr.Checkbox(label="Enable CFG-Ctrl", value=self.cfg_ctrl_enabled)
            smc_cfg_enable = gr.Checkbox(label="Enable SMC-CFG", value=self.smc_cfg_enable, info="Sliding Mode Control for stability")
            smc_cfg_lambda = gr.Slider(label="SMC Lambda", minimum=0.0, maximum=100.0, step=0.01, value=self.smc_cfg_lambda, info="Exponential decay coefficient (Default: 0.05, Try 5.0 for Flux/SD3)")
            smc_cfg_K = gr.Slider(label="SMC K (Switching Gain)", minimum=0.0, maximum=10.0, step=0.01, value=self.smc_cfg_K, info="Gain for sliding mode (Default: 0.3, Try 0.1~0.5)")
            no_cfg_warmup_steps = gr.Slider(label="No-CFG Warmup Steps", minimum=0, maximum=100, step=1, value=self.no_cfg_warmup_steps)
            
            gr.Markdown("**Original Repo Recommendations:** FLUX: λ=5.0, K=0.2 | SD3: λ=5.0, K=0.2")
        
        return [cfg_ctrl_enabled, smc_cfg_enable, smc_cfg_lambda, smc_cfg_K, no_cfg_warmup_steps]

    def process(self, p, *args):
        self.cfg_ctrl_enabled, self.smc_cfg_enable, self.smc_cfg_lambda, self.smc_cfg_K, self.no_cfg_warmup_steps = args

        # Check XYZ
        xyz_settings = getattr(p, "_guidance_xyz", {})
        cfg_ctrl_xyz = xyz_settings.get("cfg_ctrl", {})
        if "cfg_ctrl_enabled" in cfg_ctrl_xyz:
            self.cfg_ctrl_enabled = str(cfg_ctrl_xyz["cfg_ctrl_enabled"]).lower() == "true"
        if "smc_cfg_enable" in cfg_ctrl_xyz:
            self.smc_cfg_enable = str(cfg_ctrl_xyz["smc_cfg_enable"]).lower() == "true"
        if "smc_cfg_lambda" in cfg_ctrl_xyz:
            self.smc_cfg_lambda = float(cfg_ctrl_xyz["smc_cfg_lambda"])
        if "smc_cfg_K" in cfg_ctrl_xyz:
            self.smc_cfg_K = float(cfg_ctrl_xyz["smc_cfg_K"])
        if "no_cfg_warmup_steps" in cfg_ctrl_xyz:
            self.no_cfg_warmup_steps = int(cfg_ctrl_xyz["no_cfg_warmup_steps"])

        # Apply
        if self.cfg_ctrl_enabled:
            patched_unet = p.sd_model.forge_objects.unet.clone()
            
            if hasattr(p.sd_model.forge_objects.unet, "_guidance_pipeline"):
                patched_unet._guidance_pipeline = p.sd_model.forge_objects.unet._guidance_pipeline
                patched_unet.set_model_sampler_post_cfg_function(
                    patched_unet._guidance_pipeline.run, "custom_guidance_pipeline"
                )

            pipeline = ensure_guidance_pipeline(patched_unet)
            initial_sigma = get_initial_sigma(patched_unet)
            pipeline.set_base_builder(
                make_cfg_ctrl_base_builder(
                    smc_cfg_enable=self.smc_cfg_enable,
                    smc_cfg_lambda=self.smc_cfg_lambda,
                    smc_cfg_K=self.smc_cfg_K,
                    no_cfg_warmup_steps=self.no_cfg_warmup_steps,
                    initial_sigma=initial_sigma,
                )
            )
            
            p.sd_model.forge_objects.unet = patched_unet
            p.extra_generation_params["CFG-Ctrl Enabled"] = self.cfg_ctrl_enabled
            p.extra_generation_params["CFG-Ctrl SMC Enable"] = self.smc_cfg_enable
            p.extra_generation_params["CFG-Ctrl Lambda"] = self.smc_cfg_lambda
            p.extra_generation_params["CFG-Ctrl K"] = self.smc_cfg_K
            p.extra_generation_params["CFG-Ctrl Warmup Steps"] = self.no_cfg_warmup_steps
            logging.debug(f"CFG-Ctrl: Patch applied (SMC={self.smc_cfg_enable}, λ={self.smc_cfg_lambda}, K={self.smc_cfg_K})")

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        options = [
            xyz_grid.AxisOption(
                label="(CFG-Ctrl) Enabled",
                type=str,
                apply=partial(set_guidance_value_func, feature="cfg_ctrl", field="cfg_ctrl_enabled"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(
                label="(CFG-Ctrl) SMC Enable",
                type=str,
                apply=partial(set_guidance_value_func, feature="cfg_ctrl", field="smc_cfg_enable"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(
                label="(CFG-Ctrl) Lambda",
                type=float,
                apply=partial(set_guidance_value_func, feature="cfg_ctrl", field="smc_cfg_lambda"),
                choices=lambda: [0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            ),
            xyz_grid.AxisOption(
                label="(CFG-Ctrl) K (Switching Gain)",
                type=float,
                apply=partial(set_guidance_value_func, feature="cfg_ctrl", field="smc_cfg_K"),
                choices=lambda: [0.1, 0.2, 0.3, 0.5, 1.0],
            ),
            xyz_grid.AxisOption(
                label="(CFG-Ctrl) Warmup Steps",
                type=int,
                apply=partial(set_guidance_value_func, feature="cfg_ctrl", field="no_cfg_warmup_steps"),
                choices=lambda: [0, 1, 2, 3, 5, 10],
            ),
        ]
        xyz_grid.axis_options.extend(options)


register_processor(CFGCtrlProcessor)
