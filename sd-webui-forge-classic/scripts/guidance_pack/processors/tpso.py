import logging
import gradio as gr
from functools import partial
from ..base import GuidanceProcessor
from ..registry import register_processor
from core.forge_compat import get_unet, set_unet
from core.nodes.tpso import TPSONode

class TPSOProcessor(GuidanceProcessor):
    def __init__(self):
        self.node = TPSONode()
        self.tpso_enabled = False
        self.tpso_steps = 10
        self.tpso_lr = 0.01
        self.tpso_lambda = 0.5
        self.tpso_r = 0.4
        self.tpso_kappa = 0.8
        self.tpso_use_alpha = True

    def name(self) -> str:
        return "TPSO"

    def create_ui(self):
        with gr.Tab(label="TPSO"):
            gr.Markdown("Token-Prompt embedding Space Optimization (Diversity).")
            tpso_enabled = gr.Checkbox(label="Enable TPSO", value=self.tpso_enabled)
            tpso_steps = gr.Slider(
                label="Optimization Steps", minimum=1, maximum=50, step=1, value=self.tpso_steps
            )
            tpso_lr = gr.Slider(
                label="Learning Rate", minimum=0.001, maximum=0.1, step=0.001, value=self.tpso_lr
            )
            tpso_lambda = gr.Slider(
                label="Diversity Lambda", minimum=0.0, maximum=5.0, step=0.1, value=self.tpso_lambda
            )
            tpso_r = gr.Slider(
                label="Schedule Ratio (r)", minimum=0.0, maximum=1.0, step=0.05, value=self.tpso_r,
                info="Proportion of early steps to use optimized embeddings (0.4 = 40%)."
            )
            tpso_kappa = gr.Slider(
                label="Semantic Retention (Kappa)", minimum=0.5, maximum=1.0, step=0.01, value=self.tpso_kappa,
                info="Target cosine similarity. Higher = more faithful to prompt, Lower = more diverse."
            )
            tpso_use_alpha = gr.Checkbox(label="Use Alpha Scheduler (Interpolation)", value=self.tpso_use_alpha)
        return [tpso_enabled, tpso_steps, tpso_lr, tpso_lambda, tpso_r, tpso_kappa, tpso_use_alpha]

    def process(self, p, *args):
        self.tpso_enabled, self.tpso_steps, self.tpso_lr, self.tpso_lambda, self.tpso_r, self.tpso_kappa, self.tpso_use_alpha = args

        # Safe boolean conversion
        def to_bool(x):
            if isinstance(x, str):
                return x.lower() in ("true", "1", "yes", "on")
            return bool(x)

        self.tpso_enabled = to_bool(self.tpso_enabled)
        self.tpso_use_alpha = to_bool(self.tpso_use_alpha)

        xyz_settings = getattr(p, "_guidance_xyz", {})
        tpso_xyz = xyz_settings.get("tpso", {})
        if "tpso_enabled" in tpso_xyz:
            self.tpso_enabled = str(tpso_xyz["tpso_enabled"]).lower() == "true"
        if "tpso_steps" in tpso_xyz:
            self.tpso_steps = int(tpso_xyz["tpso_steps"])
        if "tpso_lr" in tpso_xyz:
            self.tpso_lr = float(tpso_xyz["tpso_lr"])
        if "tpso_lambda" in tpso_xyz:
            self.tpso_lambda = float(tpso_xyz["tpso_lambda"])
        if "tpso_r" in tpso_xyz:
            self.tpso_r = float(tpso_xyz["tpso_r"])
        if "tpso_use_alpha" in tpso_xyz:
            self.tpso_use_alpha = str(tpso_xyz["tpso_use_alpha"]).lower() == "true"

        if self.tpso_enabled:
            patched_unet = self.node.patch(
                get_unet(p),
                p=p,
                tpso_enabled=self.tpso_enabled,
                tpso_steps=self.tpso_steps,
                tpso_lr=self.tpso_lr,
                tpso_lambda=self.tpso_lambda,
                tpso_r=self.tpso_r,
                tpso_kappa=self.tpso_kappa,
                tpso_use_alpha=self.tpso_use_alpha,
            )[0]
            set_unet(p, patched_unet)
            p.extra_generation_params["TPSO Enabled"] = self.tpso_enabled
            p.extra_generation_params["TPSO Steps"] = self.tpso_steps
            p.extra_generation_params["TPSO LR"] = self.tpso_lr
            p.extra_generation_params["TPSO Lambda"] = self.tpso_lambda
            p.extra_generation_params["TPSO r"] = self.tpso_r
            p.extra_generation_params["TPSO Kappa"] = self.tpso_kappa
            p.extra_generation_params["TPSO Use Alpha"] = self.tpso_use_alpha
            logging.debug(f"TPSO: Patch applied. Alpha={self.tpso_use_alpha}")

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        options = [
            xyz_grid.AxisOption(
                label="(TPSO) Enabled",
                type=str,
                apply=partial(set_guidance_value_func, feature="tpso", field="tpso_enabled"),
                choices=lambda: ["True", "False"],
            ),
            xyz_grid.AxisOption(
                label="(TPSO) Steps",
                type=int,
                apply=partial(set_guidance_value_func, feature="tpso", field="tpso_steps"),
            ),
            xyz_grid.AxisOption(
                label="(TPSO) LR",
                type=float,
                apply=partial(set_guidance_value_func, feature="tpso", field="tpso_lr"),
            ),
            xyz_grid.AxisOption(
                label="(TPSO) Lambda",
                type=float,
                apply=partial(set_guidance_value_func, feature="tpso", field="tpso_lambda"),
            ),
            xyz_grid.AxisOption(
                label="(TPSO) r (Schedule)",
                type=float,
                apply=partial(set_guidance_value_func, feature="tpso", field="tpso_r"),
            ),
            xyz_grid.AxisOption(
                label="(TPSO) Kappa",
                type=float,
                apply=partial(set_guidance_value_func, feature="tpso", field="tpso_kappa"),
            ),
            xyz_grid.AxisOption(
                label="(TPSO) Use Alpha",
                type=str,
                apply=partial(set_guidance_value_func, feature="tpso", field="tpso_use_alpha"),
                choices=lambda: ["True", "False"],
            ),
        ]
        xyz_grid.axis_options.extend(options)

register_processor(TPSOProcessor)
