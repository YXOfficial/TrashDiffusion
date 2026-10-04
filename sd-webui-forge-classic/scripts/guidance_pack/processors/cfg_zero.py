import logging
import gradio as gr
from functools import partial
from ..base import GuidanceProcessor
from ..registry import register_processor
from core.forge_compat import get_unet, set_unet
from core.nodes.cfg_zero import CFGZeroNode

class CFGZeroProcessor(GuidanceProcessor):
    def __init__(self):
        self.node = CFGZeroNode()
        self.cfg_zero_enabled = False
        self.zero_init_first_step = False

    def name(self) -> str:
        return "CFG-Zero"

    def create_ui(self):
        with gr.Tab(label="CFG-Zero"):
            gr.Markdown("Enable CFG-Zero guidance adjustments.")
            cfg_zero_enabled = gr.Checkbox(label="Enable CFG-Zero", value=self.cfg_zero_enabled)
            zero_init_first_step = gr.Checkbox(
                label="Zero Init First Step (Experimental)", value=self.zero_init_first_step
            )
        return [cfg_zero_enabled, zero_init_first_step]

    def infotext_fields(self):
        return ["CFG-Zero Enabled", "CFG-Zero Init First Step"]

    def process(self, p, *args):
        # Unpack args
        self.cfg_zero_enabled, self.zero_init_first_step = args

        # Check XYZ
        xyz_settings = getattr(p, "_guidance_xyz", {})
        cfg_zero_xyz = xyz_settings.get("cfg_zero", {})
        if "cfg_zero_enabled" in cfg_zero_xyz:
            self.cfg_zero_enabled = str(cfg_zero_xyz["cfg_zero_enabled"]).lower() == "true"
        if "zero_init" in cfg_zero_xyz:
            self.zero_init_first_step = str(cfg_zero_xyz["zero_init"]).lower() == "true"

        # Apply
        if self.cfg_zero_enabled:
            patched_unet = self.node.patch(
                get_unet(p),
                cfg_zero_enabled=self.cfg_zero_enabled,
                zero_init_first_step=self.zero_init_first_step,
            )[0]
            set_unet(p, patched_unet)
            self.record_params(p, {
                "CFG-Zero Enabled": True,
                "CFG-Zero Init First Step": bool(self.zero_init_first_step),
            })
            logging.debug("CFG-Zero: Patch applied from Guidance Pack.")

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        options = [
            xyz_grid.AxisOption(
                label="(CFG-Zero) Enabled",
                type=str,
                apply=partial(set_guidance_value_func, feature="cfg_zero", field="cfg_zero_enabled"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(
                label="(CFG-Zero) Zero Init",
                type=str,
                apply=partial(set_guidance_value_func, feature="cfg_zero", field="zero_init"),
                choices=lambda: ["True", "False"],
            ),
        ]
        xyz_grid.axis_options.extend(options)

register_processor(CFGZeroProcessor)
