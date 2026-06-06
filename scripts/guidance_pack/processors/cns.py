"""
CNS (Colored Noise Sampling) Guidance Processor for Forge.
Based on "Colored Noise Diffusion Sampling" (Davidson et al., 2026)
Ported from ComfyUI-CNS-Sampler-CHENGOU
"""

import os
import torch
import torch.nn.functional as F
import gradio as gr
from functools import partial
from ..base import GuidanceProcessor
from ..registry import register_processor

# ─────────────────────────────────────────────────────────────────────────────
# Core CNS Logic (STRICT - NO FALLBACK)
# ─────────────────────────────────────────────────────────────────────────────

def load_gamma_matrix(path):
    if not path or not os.path.exists(path):
        return None
    try:
        # Strict load of the official spectral progress matrix
        gm = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(gm, torch.Tensor):
            return gm
    except Exception as e:
        print(f"[CNS] Error loading REAL gamma matrix from {path}: {e}")
    return None

def compute_radial_freq_bins(height, width, num_bins=32, device="cpu"):
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    fy2d, fx2d = torch.meshgrid(fy, fx, indexing='ij')
    r = torch.sqrt(fy2d ** 2 + fx2d ** 2)
    r_max = r.max().item() + 1e-8
    bins = (r / r_max * (num_bins - 1)).long()
    return bins

def compute_beta_schedule(gamma_t, power_gamma=1.0, gamma_divider=1.0,
                           alpha_tilt=0.0, use_fnorm=False, num_bins=32):
    # γ(f, t) is frequency-dependent spectral progress
    gamma_t = (gamma_t / gamma_divider).clamp(0.0, 1.0)
    residual = (1.0 - gamma_t).clamp(min=1e-8) ** power_gamma

    if alpha_tilt != 0.0:
        freqs = torch.linspace(0.0, 1.0, num_bins, device=gamma_t.device)
        if use_fnorm:
            tilt = torch.exp(alpha_tilt * freqs)
        else:
            tilt = torch.ones(num_bins, device=gamma_t.device) * (1.0 + alpha_tilt)
        residual = residual * tilt

    mean_residual = residual.mean().clamp(min=1e-8)
    beta = torch.sqrt(residual / mean_residual)
    return beta

def apply_cns_to_noise(noise, beta, freq_bins, energy_scale=1.0):
    device = noise.device
    beta = beta.to(device)
    freq_bins = freq_bins.to(device)

    scale_map = beta[freq_bins] * energy_scale

    noise_f = torch.fft.fft2(noise, dim=(-2, -1))

    expand_dims = [1] * (noise.ndim - 2) + [scale_map.shape[0], scale_map.shape[1]]
    scale_map_expanded = scale_map.view(*expand_dims)

    noise_f_colored = noise_f * scale_map_expanded
    colored = torch.fft.ifft2(noise_f_colored, dim=(-2, -1)).real

    norm_dims = tuple(range(1, noise.ndim))
    orig_std = noise.std(dim=norm_dims, keepdim=True).clamp(min=1e-8)
    colored_std = colored.std(dim=norm_dims, keepdim=True).clamp(min=1e-8)
    colored = colored * (orig_std / colored_std)

    return colored

class CNSHandler:
    def __init__(self, params):
        self.params = params
        self.gamma_matrix = None
        self.freq_bins = None

    def apply(self, noise, i, sigmas):
        T = len(sigmas) - 1
        num_bins = self.params.get("num_freq_bins", 32)

        if self.gamma_matrix is None or self.gamma_matrix.shape[1] != T:
            # FORCE LOAD - NO FAKE APPROXIMATION ALLOWED
            path = self.params.get("gamma_matrix_pt", "")
            loaded_gm = load_gamma_matrix(path)
            
            if loaded_gm is None:
                # If no real matrix, CNS is DISABLED. We do NOT fallback to concept-swapping white noise progress.
                if i == 0:
                    print(f"[CNS] FAILED TO LOAD REAL GAMMA MATRIX FROM {path}. CNS DISABLED.")
                return noise
            
            # Rescale the REAL spectral matrix to current sampling steps and bins
            gm = loaded_gm.float()
            if gm.shape[1] != T:
                gm = F.interpolate(gm.unsqueeze(1), size=(T,), mode='linear', align_corners=True).squeeze(1)
            if gm.shape[0] != num_bins:
                gm = F.interpolate(gm.t().unsqueeze(1), size=(num_bins,), mode='linear', align_corners=True).squeeze(1).t()
            
            self.gamma_matrix = gm.to(noise.device)
            print(f"[CNS] Successfully loaded and interpolated REAL gamma matrix for {T} steps.")
            H, W = noise.shape[-2:]
            self.freq_bins = compute_radial_freq_bins(H, W, num_bins=num_bins, device=noise.device)

        t_norm = i / max(T - 1, 1)
        alpha_tilt_start = self.params.get("alpha_tilt_start", 0.15)
        alpha_tilt_end = self.params.get("alpha_tilt_end", -0.5)
        
        if self.params.get("alpha_exp_interp", True):
            sharpness = self.params.get("alpha_exp_sharpness", 0.75)
            w = (torch.tensor(t_norm) * sharpness).exp()
            w = (w - 1) / (torch.tensor(sharpness).exp() - 1 + 1e-8)
            w = w.item()
        else:
            w = t_norm
            
        alpha_t = alpha_tilt_start + w * (alpha_tilt_end - alpha_tilt_start)
        gamma_t = self.gamma_matrix[:, i]
        
        beta = compute_beta_schedule(
            gamma_t,
            power_gamma=self.params.get("power_gamma", 0.75),
            gamma_divider=self.params.get("gamma_divider", 1.73),
            alpha_tilt=alpha_t,
            use_fnorm=self.params.get("alpha_use_fnorm", True),
            num_bins=num_bins,
        )
        
        return apply_cns_to_noise(noise, beta, self.freq_bins, 
                                  energy_scale=self.params.get("energy_scale", 0.98))

# ─────────────────────────────────────────────────────────────────────────────
# Processor Implementation
# ─────────────────────────────────────────────────────────────────────────────

class CNSProcessor(GuidanceProcessor):
    def __init__(self):
        self.enabled = False
        
        # Determine path to the actual bundled matrix
        # Look for it in the CHENGOU directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        bundled_path = os.path.join(root_dir, "ComfyUI-CNS-Sampler-CHENGOU", "gamma_matrix_scaled.pt")
        
        if not os.path.exists(bundled_path):
            # Try sibling path if root structure is different
            bundled_path = os.path.join(os.path.dirname(root_dir), "ComfyUI-CNS-Sampler-CHENGOU", "gamma_matrix_scaled.pt")
            
        if not os.path.exists(bundled_path):
            bundled_path = ""

        self.params = {
            "s_churn": 0.5,
            "power_gamma": 0.75,
            "gamma_divider": 1.73,
            "energy_scale": 0.98,
            "alpha_tilt_start": 0.15,
            "alpha_tilt_end": -0.5,
            "alpha_use_fnorm": True,
            "alpha_exp_interp": True,
            "alpha_exp_sharpness": 0.75,
            "num_freq_bins": 32,
            "gamma_matrix_pt": bundled_path,
        }

    def name(self) -> str:
        return "CNS (Colored Noise Sampling)"

    def create_ui(self):
        with gr.Tab(label="CNS"):
            gr.Markdown("Implementation of Colored Noise Diffusion Sampling (Davidson et al., 2026)")
            enabled = gr.Checkbox(label="Enable CNS", value=self.enabled)
            
            gamma_matrix_pt = gr.Textbox(label="Gamma Matrix Path (.pt)", value=self.params["gamma_matrix_pt"], 
                                         placeholder="REQUIRED: Path to official precomputed gamma matrix")

            with gr.Row():
                s_churn = gr.Slider(label="SDE Noise Strength (s_churn)", minimum=0.0, maximum=2.0, step=0.01, value=self.params["s_churn"])
                power_gamma = gr.Slider(label="Power Gamma", minimum=0.1, maximum=3.0, step=0.05, value=self.params["power_gamma"])
            
            with gr.Row():
                gamma_divider = gr.Slider(label="Gamma Divider", minimum=0.1, maximum=50.0, step=0.01, value=self.params["gamma_divider"])
                energy_scale = gr.Slider(label="Energy Scale", minimum=0.5, maximum=1.5, step=0.005, value=self.params["energy_scale"])
            
            with gr.Row():
                alpha_tilt_start = gr.Slider(label="Alpha Tilt Start", minimum=-2.0, maximum=2.0, step=0.01, value=self.params["alpha_tilt_start"])
                alpha_tilt_end = gr.Slider(label="Alpha Tilt End", minimum=-2.0, maximum=2.0, step=0.01, value=self.params["alpha_tilt_end"])
            
            with gr.Row():
                alpha_use_fnorm = gr.Checkbox(label="Use F-Norm", value=self.params["alpha_use_fnorm"])
                alpha_exp_interp = gr.Checkbox(label="Exp Interpolation", value=self.params["alpha_exp_interp"])
                alpha_exp_sharpness = gr.Slider(label="Exp Sharpness", minimum=0.1, maximum=10.0, step=0.05, value=self.params["alpha_exp_sharpness"])
                num_freq_bins = gr.Slider(label="Frequency Bins", minimum=8, maximum=128, step=8, value=self.params["num_freq_bins"])
                
        return [enabled, gamma_matrix_pt, s_churn, power_gamma, gamma_divider, energy_scale, 
                alpha_tilt_start, alpha_tilt_end, alpha_use_fnorm, 
                alpha_exp_interp, alpha_exp_sharpness, num_freq_bins]

    def process(self, p, *args):
        self.enabled = args[0]
        if not self.enabled:
            return

        self.params = {
            "gamma_matrix_pt": args[1],
            "s_churn": args[2],
            "power_gamma": args[3],
            "gamma_divider": args[4],
            "energy_scale": args[5],
            "alpha_tilt_start": args[6],
            "alpha_tilt_end": args[7],
            "alpha_use_fnorm": args[8],
            "alpha_exp_interp": args[9],
            "alpha_exp_sharpness": args[10],
            "num_freq_bins": args[11],
        }

        # Handle XYZ
        xyz_settings = getattr(p, "_guidance_xyz", {})
        cns_xyz = xyz_settings.get("cns", {})
        for key in self.params:
            if key in cns_xyz:
                val = cns_xyz[key]
                if isinstance(self.params[key], bool):
                    self.params[key] = str(val).lower() == "true"
                elif isinstance(self.params[key], int):
                    self.params[key] = int(float(val))
                elif isinstance(self.params[key], float):
                    self.params[key] = float(val)

        # Register handler in model_options for custom_sampler.py
        handler = CNSHandler(self.params)
        
        unet_options = p.sd_model.forge_objects.unet.model_options
        if "transformer_options" not in unet_options:
            unet_options["transformer_options"] = {}
        
        unet_options["transformer_options"]["cns_handler"] = handler
        
        if not hasattr(p, "extra_generation_params"):
            p.extra_generation_params = {}
        
        p.extra_generation_params["CNS Enabled"] = True
        p.extra_generation_params["CNS Matrix"] = os.path.basename(self.params["gamma_matrix_pt"])
        
        p.s_churn = self.params["s_churn"]
        p.s_noise = 1.0

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        for key in ["s_churn", "power_gamma", "gamma_divider", "alpha_tilt_start"]:
            xyz_grid.axis_options.append(xyz_grid.AxisOption(
                label=f"CNS {key}",
                type=float,
                apply=partial(set_guidance_value_func, feature="cns", field=key)
            ))

register_processor(CNSProcessor)
