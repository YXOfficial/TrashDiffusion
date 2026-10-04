import logging
import gradio as gr
from functools import partial
from ..base import GuidanceProcessor
from ..registry import register_processor
from core.forge_compat import get_unet, set_unet
from core.nodes.s2 import S2GuidanceNode

class S2GuidanceProcessor(GuidanceProcessor):
    def __init__(self):
        self.node = S2GuidanceNode()
        self.s2_guidance_enabled = False
        self.s2_scale_omega = 0.25
        self.s2_drop_ratio = 0.1

    def name(self) -> str:
        return "S²-Guidance"

    def create_ui(self):
        with gr.Tab(label="S²-Guidance"):
            gr.Markdown("Enable and configure S²-Guidance. Increases generation time.")
            s2_guidance_enabled = gr.Checkbox(
                label="Enable S²-Guidance", value=self.s2_guidance_enabled
            )
            s2_scale_omega = gr.Slider(
                label="S² Scale (ω)", minimum=0.0, maximum=2.0, step=0.05, value=self.s2_scale_omega
            )
            s2_drop_ratio = gr.Slider(
                label="Block Drop Ratio",
                minimum=0.0,
                maximum=0.5,
                step=0.01,
                value=self.s2_drop_ratio,
                info="Ratio of UNet/DiT blocks to drop. Paper suggests ~0.1 (10%)",
            )
        return [s2_guidance_enabled, s2_scale_omega, s2_drop_ratio]

    def process(self, p, *args):
        self.s2_guidance_enabled, self.s2_scale_omega, self.s2_drop_ratio = args

        xyz_settings = getattr(p, "_guidance_xyz", {})
        s2_guidance_xyz = xyz_settings.get("s2_guidance", {})
        if "s2_enabled" in s2_guidance_xyz:
            self.s2_guidance_enabled = str(s2_guidance_xyz["s2_enabled"]).lower() == "true"
        if "s2_omega" in s2_guidance_xyz:
            self.s2_scale_omega = float(s2_guidance_xyz["s2_omega"])
        if "s2_drop" in s2_guidance_xyz:
            self.s2_drop_ratio = float(s2_guidance_xyz["s2_drop"])

        if self.s2_guidance_enabled:
            patched_unet = self.node.patch(
                get_unet(p),
                s2_guidance_enabled=self.s2_guidance_enabled,
                s2_scale_omega=self.s2_scale_omega,
                s2_drop_ratio=self.s2_drop_ratio,
            )[0]
            set_unet(p, patched_unet)
            p.extra_generation_params["S2-Guidance Enabled"] = self.s2_guidance_enabled
            p.extra_generation_params["S2-Guidance Omega"] = self.s2_scale_omega
            p.extra_generation_params["S2-Guidance Drop Ratio"] = self.s2_drop_ratio
            logging.debug("S2-Guidance: Patch applied.")

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        options = [
            xyz_grid.AxisOption(
                label="(S2-Guidance) Enabled",
                type=str,
                apply=partial(set_guidance_value_func, feature="s2_guidance", field="s2_enabled"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(
                label="(S2-Guidance) Omega",
                type=float,
                apply=partial(set_guidance_value_func, feature="s2_guidance", field="s2_omega"),
            ),
            xyz_grid.AxisOption(
                label="(S2-Guidance) Drop Ratio",
                type=float,
                apply=partial(set_guidance_value_func, feature="s2_guidance", field="s2_drop"),
            ),
        ]
        xyz_grid.axis_options.extend(options)

register_processor(S2GuidanceProcessor)
