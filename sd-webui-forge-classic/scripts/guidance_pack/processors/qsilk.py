import logging
import gradio as gr
from functools import partial
from ..base import GuidanceProcessor
from ..registry import register_processor
from core.forge_compat import get_unet
from core.guidance import ensure_guidance_pipeline, make_qsilk_modifier

class QSilkProcessor(GuidanceProcessor):
    def __init__(self):
        self.qsilk_enabled = False
        self.micro_q_low = 0.001
        self.micro_q_high = 0.999
        self.micro_alpha = 2.0
        self.qsilk_vp_mix = 0.15
        self.qsilk_per_channel = False
        self.qsilk_late_weighting = True
        self.qsilk_use_aqclip = False
        self.qsilk_tile_size = 32
        self.qsilk_stride = 16
        self.qsilk_aqclip_alpha = 2.0
        self.qsilk_ema_beta = 0.8

    def name(self) -> str:
        return "QSilk"

    def create_ui(self):
        with gr.Tab(label="QSilk"):
            gr.Markdown("Stabilize denoised latents with Micrograin + AQClip-Lite.")
            qsilk_enabled = gr.Checkbox(label="Enable QSilk", value=self.qsilk_enabled)
            micro_q_low = gr.Slider(
                label="Micrograin q_low",
                minimum=0.0,
                maximum=0.01,
                step=0.0001,
                value=self.micro_q_low,
            )
            micro_q_high = gr.Slider(
                label="Micrograin q_high",
                minimum=0.95,
                maximum=1.0,
                step=0.0001,
                value=self.micro_q_high,
            )
            micro_alpha = gr.Slider(
                label="Micrograin α",
                minimum=0.1,
                maximum=10.0,
                step=0.1,
                value=self.micro_alpha,
            )
            qsilk_vp_mix = gr.Slider(
                label="Micrograin VP-Mix",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=self.qsilk_vp_mix,
                info="Blend toward variance-preserving remap (0=off, default 0.15).",
            )
            qsilk_per_channel = gr.Checkbox(
                label="Micrograin Per-Channel (slower)",
                value=self.qsilk_per_channel,
                info="Compute quantiles/VP stats per channel; can be slower.",
            )
            qsilk_late_weighting = gr.Checkbox(
                label="Late-Step Weighting",
                value=self.qsilk_late_weighting,
                info="Fade in micrograin over sampling steps to reduce early color feedback.",
            )
            qsilk_use_aqclip = gr.Checkbox(
                label="Enable AQClip-Lite", value=self.qsilk_use_aqclip
            )
            qsilk_tile_size = gr.Slider(
                label="AQClip Tile Size",
                minimum=8,
                maximum=128,
                step=8,
                value=self.qsilk_tile_size,
            )
            qsilk_stride = gr.Slider(
                label="AQClip Stride",
                minimum=4,
                maximum=128,
                step=4,
                value=self.qsilk_stride,
                info="Should be ≤ tile size to allow overlap",
            )
            qsilk_aqclip_alpha = gr.Slider(
                label="AQClip α",
                minimum=0.1,
                maximum=10.0,
                step=0.1,
                value=self.qsilk_aqclip_alpha,
            )
            qsilk_ema_beta = gr.Slider(
                label="AQClip EMA β",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=self.qsilk_ema_beta,
            )
        return [
            qsilk_enabled, micro_q_low, micro_q_high, micro_alpha, qsilk_vp_mix,
            qsilk_per_channel, qsilk_late_weighting, qsilk_use_aqclip,
            qsilk_tile_size, qsilk_stride, qsilk_aqclip_alpha, qsilk_ema_beta
        ]

    def process(self, p, *args):
        (
            self.qsilk_enabled, self.micro_q_low, self.micro_q_high, self.micro_alpha,
            self.qsilk_vp_mix, self.qsilk_per_channel, self.qsilk_late_weighting,
            self.qsilk_use_aqclip, self.qsilk_tile_size, self.qsilk_stride,
            self.qsilk_aqclip_alpha, self.qsilk_ema_beta
        ) = args

        xyz_settings = getattr(p, "_guidance_xyz", {})
        qsilk_xyz = xyz_settings.get("qsilk", {})
        if "qsilk_enabled" in qsilk_xyz:
            self.qsilk_enabled = str(qsilk_xyz["qsilk_enabled"]).lower() == "true"
        if "micro_q_low" in qsilk_xyz:
            self.micro_q_low = float(qsilk_xyz["micro_q_low"])
        if "micro_q_high" in qsilk_xyz:
            self.micro_q_high = float(qsilk_xyz["micro_q_high"])
        if "micro_alpha" in qsilk_xyz:
            self.micro_alpha = float(qsilk_xyz["micro_alpha"])
        if "micro_vp_mix" in qsilk_xyz:
            self.qsilk_vp_mix = float(qsilk_xyz["micro_vp_mix"])
        if "per_channel" in qsilk_xyz:
            self.qsilk_per_channel = str(qsilk_xyz["per_channel"]).lower() == "true"
        if "late_weighting" in qsilk_xyz:
            self.qsilk_late_weighting = str(qsilk_xyz["late_weighting"]).lower() == "true"
        if "use_aqclip" in qsilk_xyz:
            self.qsilk_use_aqclip = str(qsilk_xyz["use_aqclip"]).lower() == "true"
        if "tile_size" in qsilk_xyz:
            self.qsilk_tile_size = int(qsilk_xyz["tile_size"])
        if "stride" in qsilk_xyz:
            self.qsilk_stride = int(qsilk_xyz["stride"])
        if "aqclip_alpha" in qsilk_xyz:
            self.qsilk_aqclip_alpha = float(qsilk_xyz["aqclip_alpha"])
        if "ema_beta" in qsilk_xyz:
            self.qsilk_ema_beta = float(qsilk_xyz["ema_beta"])

        pipeline = None
        needs_qsilk_pipeline = self.qsilk_enabled or hasattr(get_unet(p), "_guidance_pipeline")
        if needs_qsilk_pipeline:
            pipeline = ensure_guidance_pipeline(get_unet(p))

        if self.qsilk_enabled and pipeline is not None:
            pipeline.add_modifier(
                "qsilk",
                make_qsilk_modifier(
                    micro_q_low=self.micro_q_low,
                    micro_q_high=self.micro_q_high,
                    micro_alpha=self.micro_alpha,
                    micro_vp_mix=self.qsilk_vp_mix,
                    per_channel=self.qsilk_per_channel,
                    enable_late_weighting=self.qsilk_late_weighting,
                    use_aqclip=self.qsilk_use_aqclip,
                    tile_size=int(self.qsilk_tile_size),
                    stride=int(self.qsilk_stride),
                    aqclip_alpha=self.qsilk_aqclip_alpha,
                    ema_beta=self.qsilk_ema_beta,
                ),
            )
            p.extra_generation_params["QSilk Enabled"] = self.qsilk_enabled
            p.extra_generation_params["QSilk micro_q_low"] = self.micro_q_low
            p.extra_generation_params["QSilk micro_q_high"] = self.micro_q_high
            p.extra_generation_params["QSilk micro_alpha"] = self.micro_alpha
            p.extra_generation_params["QSilk micro_vp_mix"] = self.qsilk_vp_mix
            p.extra_generation_params["QSilk per_channel"] = self.qsilk_per_channel
            p.extra_generation_params["QSilk late_weighting"] = self.qsilk_late_weighting
            p.extra_generation_params["QSilk use_aqclip"] = self.qsilk_use_aqclip
            p.extra_generation_params["QSilk tile_size"] = int(self.qsilk_tile_size)
            p.extra_generation_params["QSilk stride"] = int(self.qsilk_stride)
            p.extra_generation_params["QSilk aqclip_alpha"] = self.qsilk_aqclip_alpha
            p.extra_generation_params["QSilk ema_beta"] = self.qsilk_ema_beta
            logging.debug("QSilk: Applied.")
        elif pipeline is not None and "qsilk" in pipeline.modifiers:
            pipeline.modifiers.pop("qsilk", None)

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        options = [
            xyz_grid.AxisOption(
                label="(QSilk) Enabled",
                type=str,
                apply=partial(set_guidance_value_func, feature="qsilk", field="qsilk_enabled"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(
                label="(QSilk) micro_q_low",
                type=float,
                apply=partial(set_guidance_value_func, feature="qsilk", field="micro_q_low"),
            ),
            xyz_grid.AxisOption(
                label="(QSilk) micro_q_high",
                type=float,
                apply=partial(set_guidance_value_func, feature="qsilk", field="micro_q_high"),
            ),
            xyz_grid.AxisOption(
                label="(QSilk) micro_alpha",
                type=float,
                apply=partial(set_guidance_value_func, feature="qsilk", field="micro_alpha"),
            ),
            xyz_grid.AxisOption(
                label="(QSilk) micro_vp_mix",
                type=float,
                apply=partial(set_guidance_value_func, feature="qsilk", field="micro_vp_mix"),
            ),
            xyz_grid.AxisOption(
                label="(QSilk) per_channel",
                type=str,
                apply=partial(set_guidance_value_func, feature="qsilk", field="per_channel"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(
                label="(QSilk) late_weighting",
                type=str,
                apply=partial(set_guidance_value_func, feature="qsilk", field="late_weighting"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(
                label="(QSilk) use_aqclip",
                type=str,
                apply=partial(set_guidance_value_func, feature="qsilk", field="use_aqclip"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(
                label="(QSilk) tile_size",
                type=int,
                apply=partial(set_guidance_value_func, feature="qsilk", field="tile_size"),
            ),
            xyz_grid.AxisOption(
                label="(QSilk) stride",
                type=int,
                apply=partial(set_guidance_value_func, feature="qsilk", field="stride"),
            ),
            xyz_grid.AxisOption(
                label="(QSilk) aqclip_alpha",
                type=float,
                apply=partial(set_guidance_value_func, feature="qsilk", field="aqclip_alpha"),
            ),
            xyz_grid.AxisOption(
                label="(QSilk) ema_beta",
                type=float,
                apply=partial(set_guidance_value_func, feature="qsilk", field="ema_beta"),
            ),
        ]
        xyz_grid.axis_options.extend(options)

register_processor(QSilkProcessor)
