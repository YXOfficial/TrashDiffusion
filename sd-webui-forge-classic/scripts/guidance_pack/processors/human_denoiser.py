import numpy as np
import torch
import gradio as gr

from ..base import GuidanceProcessor
from ..registry import register_processor

from .human_denoiser_bridge import HumanDenoiserBridge


def _taesd_decode_to_np(latent):
    from modules import sd_vae_taesd, devices
    mdl = sd_vae_taesd.decoder_model()
    if mdl is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    param = next(mdl.parameters())
    img = mdl(latent.to(device=devices.device, dtype=param.dtype))
    img = img[0].float().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _kl_sigmas(n, sigma_min, sigma_max, device):
    alpha_min = torch.arctan(torch.tensor(sigma_min, device=device))
    alpha_max = torch.arctan(torch.tensor(sigma_max, device=device))
    step_indices = torch.arange(n + 1, device=device)
    return torch.tan(step_indices / max(n, 1) * alpha_min + (1.0 - step_indices / max(n, 1)) * alpha_max)


def _lambda(sigma_i, sigma_next):
    if sigma_i <= 0:
        return 1.0
    return max((sigma_i - sigma_next) / max(sigma_i, 1e-8), 1e-8)


class HumanDenoiserProcessor(GuidanceProcessor):
    def __init__(self):
        self.bridge = HumanDenoiserBridge()

    def name(self):
        return "Human Denoiser (Absolute Limit 2)"

    def create_ui(self):
        with gr.Tab(label="Human Denoiser"):
            status = gr.Textbox(label="Status", value="Idle", interactive=False)

            candidate_gallery = gr.Gallery(
                label="Chọn path (click ảnh để chọn, bấm SELECT để xác nhận)",
                columns=2,
                height=320,
                object_fit="contain",
                interactive=False,
                show_download_button=False,
                allow_preview=False,
            )

            with gr.Row():
                select_btn = gr.Button("SELECT", variant="primary")
                walk_steps = gr.Slider(label="Walk N steps", minimum=1, maximum=50, step=1, value=1)
                walk_btn = gr.Button("Walk by")
                stop_btn = gr.Button("Stop", variant="stop")
                refresh_btn = gr.Button("Refresh Preview")

            select_btn.click(fn=self._select, outputs=[status], queue=False)
            walk_btn.click(fn=self._walk, inputs=[walk_steps], outputs=[status], queue=False)
            stop_btn.click(fn=self._stop, outputs=[status], queue=False)
            refresh_btn.click(fn=self._refresh, outputs=[candidate_gallery, status], queue=False)
            candidate_gallery.select(fn=self._on_select, outputs=[status], queue=False)

        return [status, candidate_gallery, select_btn, walk_steps, walk_btn, stop_btn, refresh_btn]

    def _on_select(self, ev_data: gr.SelectData):
        info = self.bridge.get_latest_waiting()
        if info is None:
            return "No active generation."
        idx = int(ev_data.index)
        self.bridge.store_selection(idx)
        return f"Đã chọn path {idx}."

    def _select(self):
        info = self.bridge.get_latest_waiting()
        if info is None:
            return "No active generation."
        step = info.get("step", 0)
        idx = self.bridge.get_selection()
        if idx is None:
            idx = 0
        self.bridge.submit_choice(step, idx)
        print(f"[HumanDenoiser] Step {step + 1}: path {idx}")
        return f"Step {step + 1}: path {idx}."

    def _walk(self, steps):
        info = self.bridge.get_latest_waiting()
        if info is None:
            return "No active generation."
        step = info.get("step", 0)
        idx = self.bridge.get_selection()
        if idx is None:
            idx = 0
        self.bridge.set_walk(steps, idx)
        self.bridge.submit_choice(step, idx)
        print(f"[HumanDenoiser] Walk {steps} steps from step {step + 1}, path {idx}")
        return f"Walk {steps} steps, path {idx}."

    def _stop(self):
        from modules import shared
        shared.state.interrupt()
        self.bridge.signal_stop()
        return "Đã dừng."

    def _refresh(self):
        info = self.bridge.get_latest_waiting()
        if info is None:
            return [gr.update(), "Idle"]
        step = info.get("step", 0)
        total = info.get("total", 0)
        sigma = info.get("sigma", 0)
        paths = info.get("paths", [])
        n = info.get("n", 0)
        label = "Default" if n == 1 else "Default | KL Optimal"
        return [paths, f"Step {step + 1}/{total} | sigma: {sigma:.4f} | {label}"]

    def process(self, p, *args):
        self.bridge.cleanup()

        default_sigmas = None
        kl_sigmas = None

        def hook(x, D, sigma_val, step, total, extra_args):
            nonlocal default_sigmas, kl_sigmas

            if torch.is_tensor(sigma_val):
                sigma_f = float(sigma_val[0].item())
            else:
                sigma_f = float(sigma_val)

            if step == 0:
                from k_diffusion.sampling import _human_denoiser_sigmas
                hds = _human_denoiser_sigmas
                print(f"[HumanDenoiser] _human_denoiser_sigmas = {hds}")
                if hds is not None and len(hds) > 1:
                    sigma_min = float(hds[-2].item())
                    sigma_max = float(hds[0].item())
                    default_sigmas = [float(hds[i].item()) for i in range(len(hds))]
                    kl_sigmas = [float(_kl_sigmas(total, sigma_min, sigma_max, x.device)[i].item()) for i in range(total + 1)]
                    ld = _lambda(default_sigmas[0], default_sigmas[1])
                    lk = _lambda(kl_sigmas[0], kl_sigmas[1])
                    print(f"[HumanDenoiser] OK range [{sigma_max:.4f}..{sigma_min:.4f}] λ_def={ld:.6f} λ_kl={lk:.6f}")
                else:
                    print(f"[HumanDenoiser] FAIL hds={hds} type={type(hds).__name__} len={len(hds) if hds is not None else 'N/A'}")

            candidates = [D]
            generating = False

            if default_sigmas is not None and kl_sigmas is not None and step < len(default_sigmas) - 1 and step < len(kl_sigmas) - 1:
                generating = True
                print("[HumanDenoiser] Is generating biến thể...")
                lambda_default = _lambda(default_sigmas[step], default_sigmas[step + 1])
                lambda_kl = _lambda(kl_sigmas[step], kl_sigmas[step + 1])
                ratio = min(lambda_kl / max(lambda_default, 1e-8), 10.0)
                D_kl = x + ratio * (D - x)
                candidates.append(D_kl.to(dtype=D.dtype))

            should_walk, walk_path = self.bridge.consume_walk()
            if should_walk:
                print(f"[HumanDenoiser] Walk step {step}, path {walk_path}")
                return candidates[walk_path].to(device=x.device, dtype=D.dtype)

            previews = [_taesd_decode_to_np(c) for c in candidates]

            print(f"[HumanDenoiser] Step {step + 1}/{total}: waiting...")
            choice = self.bridge.wait_for_choice(step, previews, total, sigma_f)

            if choice == -1:
                print("[HumanDenoiser] interrupted")
                return D

            idx = max(0, min(choice, len(candidates) - 1))
            print(f"[HumanDenoiser] Step {step + 1}: path {idx}")
            return candidates[idx].to(device=x.device, dtype=D.dtype)

        import k_diffusion.sampling
        k_diffusion.sampling.set_human_denoiser_hook(hook)

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        pass


print("[HumanDenoiser] Processor loaded")
register_processor(HumanDenoiserProcessor)
