import logging
import gradio as gr
from functools import partial
from ..base import GuidanceProcessor
from ..registry import register_processor
from core.forge_compat import get_unet, set_unet
from core.nodes.fdg import FDGNode

class FDGProcessor(GuidanceProcessor):
    def __init__(self):
        self.node = FDGNode()
        self.fdg_enabled = False
        self.fdg_w_low = 1.0
        self.fdg_w_high = 1.0
        self.fdg_levels = 3

    def name(self) -> str:
        return "FDG"

    def create_ui(self):
        with gr.Tab(label="FDG"):
            gr.Markdown("Configure Frequency-Decoupled Guidance.")
            fdg_enabled = gr.Checkbox(label="Enable FDG", value=self.fdg_enabled)
            fdg_w_low = gr.Slider(
                label="w_low (Low-Frequency Guidance)", minimum=0.0, maximum=10.0, step=0.1, value=self.fdg_w_low
            )
            fdg_w_high = gr.Slider(
                label="w_high (High-Frequency Guidance)", minimum=0.0, maximum=10.0, step=0.1, value=self.fdg_w_high
            )
            fdg_levels = gr.Slider(
                label="FDG Pyramid Levels", minimum=2, maximum=8, step=1, value=self.fdg_levels
            )
        return [fdg_enabled, fdg_w_low, fdg_w_high, fdg_levels]

    def process(self, p, *args):
        self.fdg_enabled, self.fdg_w_low, self.fdg_w_high, self.fdg_levels = args

        xyz_settings = getattr(p, "_guidance_xyz", {})
        fdg_xyz = xyz_settings.get("fdg", {})
        if "fdg_enabled" in fdg_xyz:
            self.fdg_enabled = str(fdg_xyz["fdg_enabled"]).lower() == "true"
        if "w_low" in fdg_xyz:
            self.fdg_w_low = float(fdg_xyz["w_low"])
        if "w_high" in fdg_xyz:
            self.fdg_w_high = float(fdg_xyz["w_high"])
        if "fdg_levels" in fdg_xyz:
            self.fdg_levels = int(fdg_xyz["fdg_levels"])

        if self.fdg_enabled:
            patched_unet = self.node.patch(
                get_unet(p),
                fdg_enabled=self.fdg_enabled,
                w_low=self.fdg_w_low,
                w_high=self.fdg_w_high,
                fdg_levels=int(self.fdg_levels),
            )[0]
            set_unet(p, patched_unet)
            p.extra_generation_params["FDG Enabled"] = self.fdg_enabled
            p.extra_generation_params["FDG w_low"] = self.fdg_w_low
            p.extra_generation_params["FDG w_high"] = self.fdg_w_high
            p.extra_generation_params["FDG Levels"] = int(self.fdg_levels)
            logging.debug(
                "FDG: Patch applied. Enabled=%s, w_low=%s, w_high=%s, levels=%s",
                self.fdg_enabled, self.fdg_w_low, self.fdg_w_high, self.fdg_levels
            )

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        options = [
            xyz_grid.AxisOption(
                label="(FDG) Enabled",
                type=str,
                apply=partial(set_guidance_value_func, feature="fdg", field="fdg_enabled"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(label="(FDG) w_low", type=float, apply=partial(set_guidance_value_func, feature="fdg", field="w_low")),
            xyz_grid.AxisOption(label="(FDG) w_high", type=float, apply=partial(set_guidance_value_func, feature="fdg", field="w_high")),
            xyz_grid.AxisOption(
                label="(FDG) Levels", type=int, apply=partial(set_guidance_value_func, feature="fdg", field="fdg_levels")
            ),
        ]
        xyz_grid.axis_options.extend(options)

register_processor(FDGProcessor)
