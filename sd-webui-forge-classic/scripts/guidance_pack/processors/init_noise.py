import torch
import gradio as gr
from ..base import GuidanceProcessor
from ..registry import register_processor
from core.forge_compat import get_unet


def _to_4d(x):
    B, C, T, H, W = x.shape
    return x.reshape(B * T, C, H, W)


def _from_4d(x4d, ref):
    B, C, T, H, W = ref.shape
    return x4d.reshape(B, C, T, H, W)


def _make_noise(shape, seed, device, dtype):
    gen = torch.Generator(device='cpu').manual_seed(seed + 1337)
    return torch.randn(shape, generator=gen, device='cpu', dtype=dtype).to(device)


def _get_freq_grid(H, W, device, dtype):
    fy = torch.fft.fftfreq(H, device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(W, device=device, dtype=dtype)
    fy2d, fx2d = torch.meshgrid(fy, fx, indexing='ij')
    return torch.sqrt(fy2d ** 2 + fx2d ** 2)


def _fft_filter(x, mask):
    is_wan = x.dim() == 5
    if is_wan:
        x4d = _to_4d(x).float()
        H, W = x4d.shape[-2], x4d.shape[-1]
        mask = mask.to(x4d.device, x4d.dtype)
        noise_f = torch.fft.rfft2(x4d, dim=(-2, -1), norm='ortho')
        filtered = noise_f * mask
        out4d = torch.fft.irfft2(filtered, dim=(-2, -1), s=(H, W), norm='ortho')
        return _from_4d(out4d.to(x.dtype), x)
    else:
        xf = x.float()
        H, W = xf.shape[-2], xf.shape[-1]
        mask = mask.to(xf.device, xf.dtype)
        noise_f = torch.fft.rfft2(xf, dim=(-2, -1), norm='ortho')
        filtered = noise_f * mask
        return torch.fft.irfft2(filtered, dim=(-2, -1), s=(H, W), norm='ortho').to(x.dtype)


class InitNoiseProcessor(GuidanceProcessor):
    def __init__(self):
        self.enabled = False
        self.mode = "lowpass"
        self.param = 0.0

    def name(self) -> str:
        return "Init Noise override"

    def create_ui(self):
        with gr.Tab(label="Init Noise"):
            gr.Markdown("Override initial latent noise with frequency-shaped noise.")
            enabled = gr.Checkbox(label="Enable", value=self.enabled)
            mode = gr.Radio(
                label="Noise mode",
                choices=["lowpass", "highpass", "pink"],
                value=self.mode,
            )
            param = gr.Slider(
                label="Sigma (cutoff frequency)",
                minimum=0.0,
                maximum=5.0,
                step=0.01,
                value=self.param,
            )
        return [enabled, mode, param]

    def infotext_fields(self):
        return ["Init Noise Enabled", "Init Noise", "Init Noise sigma"]

    def process(self, p, *args):
        self.enabled = args[0]
        self.record_params(p, {"Init Noise Enabled": bool(self.enabled)})
        if not self.enabled:
            return

        self.mode = args[1]
        self.param = args[2]

        xyz = getattr(p, "_guidance_xyz", {}).get("init_noise", {})
        if "mode" in xyz:
            self.mode = xyz["mode"]
        if "param" in xyz:
            self.param = float(xyz["param"])

        self.record_params(p, {"Init Noise": self.mode, "Init Noise sigma": float(self.param)})

        rng = getattr(p, "rng", None)
        if rng is None:
            return

        with torch.no_grad():
            model = get_unet(p).model.diffusion_model
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            B = len(rng.seeds) if hasattr(rng, 'seeds') else 1
            shape = rng.shape
            full_shape = (B, *shape)
            seed = rng.seeds[0] if hasattr(rng, 'seeds') else 42

            x = _make_noise(full_shape, seed, device, dtype)
            H, W = shape[-2], shape[-1]
            r = _get_freq_grid(H, W, device, dtype)

            sigma = max(abs(self.param), 0.1)

            if self.mode == "lowpass":
                mask = torch.exp(-(r ** 2) / (2 * sigma ** 2))
                x = _fft_filter(x, mask)
            elif self.mode == "highpass":
                mask = 1.0 - torch.exp(-(r ** 2) / (2 * sigma ** 2))
                x = _fft_filter(x, mask)
            elif self.mode == "pink":
                mask = torch.where(r > 0, r ** (-0.5 * sigma), torch.tensor(1.0, device=device, dtype=dtype))
                mask = mask.clamp(max=20.0)
                x = _fft_filter(x, mask)

            x = x * (x.std() + 1e-8).reciprocal()

        p.modified_noise = x

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        from functools import partial
        xyz_grid.axis_options.append(xyz_grid.AxisOption(
            label="Init Noise mode",
            type=str,
            apply=partial(set_guidance_value_func, feature="init_noise", field="mode"),
            choices=lambda: ["lowpass", "highpass", "pink"],
        ))
        xyz_grid.axis_options.append(xyz_grid.AxisOption(
            label="Init Noise sigma",
            type=float,
            apply=partial(set_guidance_value_func, feature="init_noise", field="param"),
        ))


register_processor(InitNoiseProcessor)
