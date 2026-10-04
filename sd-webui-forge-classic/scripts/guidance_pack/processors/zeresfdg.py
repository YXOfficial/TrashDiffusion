import logging
import gradio as gr
from functools import partial
from ..base import GuidanceProcessor
from ..registry import register_processor
from core.forge_compat import get_unet, set_unet
from core.nodes.zeresfdg import ZeResFDGNode

class ZeResFDGProcessor(GuidanceProcessor):
    def __init__(self):
        self.node = ZeResFDGNode()
        self.zeresfdg_enabled = False
        self.zeresfdg_controller_enabled = True
        self.zeresfdg_w_low = 0.6
        self.zeresfdg_w_high = 1.3
        self.zeresfdg_alpha = 0.7
        self.zeresfdg_tau_lo = 0.45
        self.zeresfdg_tau_hi = 0.6
        self.zeresfdg_beta = 0.8

    def name(self) -> str:
        return "ZeResFDG"

    def create_ui(self):
        with gr.Tab(label="ZeResFDG"):
            gr.Markdown(
                "Enable ZeResFDG guidance with spectral EMA switching between CFGZeroFD and RescaleFDG modes."
            )
            zeresfdg_enabled = gr.Checkbox(label="Enable ZeResFDG", value=self.zeresfdg_enabled)
            zeresfdg_controller_enabled = gr.Checkbox(
                label="Enable Spectral Controller", value=self.zeresfdg_controller_enabled
            )
            zeresfdg_w_low = gr.Slider(
                label="λ_l (Low-Frequency Gain)", minimum=0.0, maximum=10.0, step=0.05, value=self.zeresfdg_w_low
            )
            zeresfdg_w_high = gr.Slider(
                label="λ_h (High-Frequency Gain)", minimum=0.0, maximum=10.0, step=0.05, value=self.zeresfdg_w_high
            )
            zeresfdg_alpha = gr.Slider(
                label="α (Rescale Blend)", minimum=0.0, maximum=1.0, step=0.01, value=self.zeresfdg_alpha
            )
            zeresfdg_tau_lo = gr.Slider(
                label="τ_lo (EMA Low Threshold)", minimum=0.0, maximum=1.0, step=0.01, value=self.zeresfdg_tau_lo
            )
            zeresfdg_tau_hi = gr.Slider(
                label="τ_hi (EMA High Threshold)", minimum=0.0, maximum=1.0, step=0.01, value=self.zeresfdg_tau_hi
            )
            zeresfdg_beta = gr.Slider(
                label="β (EMA Smoothing)", minimum=0.0, maximum=1.0, step=0.01, value=self.zeresfdg_beta
            )
        return [
            zeresfdg_enabled, zeresfdg_controller_enabled, zeresfdg_w_low, zeresfdg_w_high,
            zeresfdg_alpha, zeresfdg_tau_lo, zeresfdg_tau_hi, zeresfdg_beta
        ]

    def process(self, p, *args):
        (
            self.zeresfdg_enabled, self.zeresfdg_controller_enabled, self.zeresfdg_w_low,
            self.zeresfdg_w_high, self.zeresfdg_alpha, self.zeresfdg_tau_lo,
            self.zeresfdg_tau_hi, self.zeresfdg_beta
        ) = args

        xyz_settings = getattr(p, "_guidance_xyz", {})
        zeresfdg_xyz = xyz_settings.get("zeresfdg", {})
        if "zeresfdg_enabled" in zeresfdg_xyz:
            self.zeresfdg_enabled = str(zeresfdg_xyz["zeresfdg_enabled"]).lower() == "true"
        if "controller_enabled" in zeresfdg_xyz:
            self.zeresfdg_controller_enabled = str(zeresfdg_xyz["controller_enabled"]).lower() == "true"
        if "w_low" in zeresfdg_xyz:
            self.zeresfdg_w_low = float(zeresfdg_xyz["w_low"])
        if "w_high" in zeresfdg_xyz:
            self.zeresfdg_w_high = float(zeresfdg_xyz["w_high"])
        if "alpha" in zeresfdg_xyz:
            self.zeresfdg_alpha = float(zeresfdg_xyz["alpha"])
        if "tau_lo" in zeresfdg_xyz:
            self.zeresfdg_tau_lo = float(zeresfdg_xyz["tau_lo"])
        if "tau_hi" in zeresfdg_xyz:
            self.zeresfdg_tau_hi = float(zeresfdg_xyz["tau_hi"])
        if "beta" in zeresfdg_xyz:
            self.zeresfdg_beta = float(zeresfdg_xyz["beta"])

        if self.zeresfdg_enabled:
            patched_unet = self.node.patch(
                get_unet(p),
                zeresfdg_enabled=self.zeresfdg_enabled,
                w_low=self.zeresfdg_w_low,
                w_high=self.zeresfdg_w_high,
                alpha=self.zeresfdg_alpha,
                tau_lo=self.zeresfdg_tau_lo,
                tau_hi=self.zeresfdg_tau_hi,
                beta=self.zeresfdg_beta,
                controller_enabled=self.zeresfdg_controller_enabled,
            )[0]
            set_unet(p, patched_unet)
            p.extra_generation_params["ZeResFDG Enabled"] = self.zeresfdg_enabled
            p.extra_generation_params["ZeResFDG λ_l"] = self.zeresfdg_w_low
            p.extra_generation_params["ZeResFDG λ_h"] = self.zeresfdg_w_high
            p.extra_generation_params["ZeResFDG α"] = self.zeresfdg_alpha
            p.extra_generation_params["ZeResFDG τ_lo"] = self.zeresfdg_tau_lo
            p.extra_generation_params["ZeResFDG τ_hi"] = self.zeresfdg_tau_hi
            p.extra_generation_params["ZeResFDG β"] = self.zeresfdg_beta
            p.extra_generation_params["ZeResFDG Controller"] = self.zeresfdg_controller_enabled
            logging.debug("ZeResFDG: Patch applied.")

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        options = [
            xyz_grid.AxisOption(
                label="(ZeResFDG) Enabled",
                type=str,
                apply=partial(set_guidance_value_func, feature="zeresfdg", field="zeresfdg_enabled"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(
                label="(ZeResFDG) Controller",
                type=str,
                apply=partial(set_guidance_value_func, feature="zeresfdg", field="controller_enabled"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(
                label="(ZeResFDG) λ_l",
                type=float,
                apply=partial(set_guidance_value_func, feature="zeresfdg", field="w_low"),
            ),
            xyz_grid.AxisOption(
                label="(ZeResFDG) λ_h",
                type=float,
                apply=partial(set_guidance_value_func, feature="zeresfdg", field="w_high"),
            ),
            xyz_grid.AxisOption(
                label="(ZeResFDG) α",
                type=float,
                apply=partial(set_guidance_value_func, feature="zeresfdg", field="alpha"),
            ),
            xyz_grid.AxisOption(
                label="(ZeResFDG) τ_lo",
                type=float,
                apply=partial(set_guidance_value_func, feature="zeresfdg", field="tau_lo"),
            ),
            xyz_grid.AxisOption(
                label="(ZeResFDG) τ_hi",
                type=float,
                apply=partial(set_guidance_value_func, feature="zeresfdg", field="tau_hi"),
            ),
            xyz_grid.AxisOption(
                label="(ZeResFDG) β",
                type=float,
                apply=partial(set_guidance_value_func, feature="zeresfdg", field="beta"),
            ),
        ]
        xyz_grid.axis_options.extend(options)

register_processor(ZeResFDGProcessor)
