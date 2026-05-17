import torch
import logging
import gradio as gr
from ..base import GuidanceProcessor
from ..registry import register_processor

# Ground Truth from sd-webui-forge-classic/backend/sampling/sampling_function.py
COND = 0 
UNCOND = 1

def make_qwen_wrapper(strength, retention, ratio, amplification):
    """
    Tạo một wrapper closure để capture các tham số từ UI.
    Điều này tránh việc phải truy cập args['model'] (không tồn tại trong Forge).
    """
    def qwen_optimizer_wrapper(apply_model, args):
        # Forge Wrapper Args: {"input": x, "timestep": t, "c": c_dict, "cond_or_uncond": [...]}
        
        # current_sigma: 1.0 (start) -> 0.0 (end)
        current_sigma = args['timestep'].max().item()
        threshold = 1.0 - ratio
        
        if current_sigma > threshold:
            c_dict = args['c'].copy()
            context = c_dict.get('c_crossattn')
            
            if context is not None and isinstance(context, torch.Tensor):
                # cond_or_uncond: List các flag COND(0) hoặc UNCOND(1) cho mỗi sample trong batch
                cond_or_uncond = torch.tensor(args['cond_or_uncond'], device=context.device)
                mask = (cond_or_uncond == COND).view(-1, 1, 1).to(context.dtype)
                
                # Tạo nhiễu đã được chuẩn hóa (perturbation vector)
                noise = torch.randn_like(context)
                noise = torch.nn.functional.normalize(noise.flatten(1), p=2, dim=1).view_as(noise)
                
                # Tính scale dựa trên norm của embedding gốc để tránh làm sụp đổ phân phối
                c_norm = context.norm(p=2, dim=(1, 2), keepdim=True)
                p_scale = c_norm * strength * (1.0 - retention) * amplification
                
                # Chỉ cộng nhiễu vào các sample là Positive (COND)
                c_dict['c_crossattn'] = context + (noise * p_scale * mask)
                
                return apply_model(args['input'], args['timestep'], **c_dict)

        # Nếu không thỏa mãn điều kiện hoặc không có context, chạy như bình thường
        return apply_model(args['input'], args['timestep'], **args['c'])
    
    return qwen_optimizer_wrapper

class QwenOptimizerProcessor(GuidanceProcessor):
    def name(self) -> str:
        return "Qwen Optimizer"

    def create_ui(self):
        with gr.Tab(label="Qwen Optimizer"):
            gr.Markdown(
                "### Qwen Optimizer\n"
                "Explores semantic variations by perturbing Qwen embeddings during the early stages of diffusion.\n"
                "Specifically designed for Anima/Wan-based DiT models."
            )
            enabled = gr.Checkbox(label="Enable Qwen Optimizer", value=False)
            with gr.Row():
                strength = gr.Slider(label="Diversity Strength", minimum=0.0, maximum=5.0, step=0.1, value=1.0)
                retention = gr.Slider(label="Semantic Retention", minimum=0.0, maximum=1.0, step=0.01, value=0.8)
            with gr.Row():
                ratio = gr.Slider(label="Injection Ratio", minimum=0.0, maximum=1.0, step=0.01, value=0.4)
                amp = gr.Slider(label="Amplification", minimum=1.0, maximum=50.0, step=1.0, value=10.0)
        
        return [enabled, strength, retention, ratio, amp]

    def process(self, p, enabled, strength, retention, ratio, amp):
        if not enabled:
            return

        # Clone UNet để apply patch cho lượt generate này
        unet = p.sd_model.forge_objects.unet.clone()
        
        # Tạo wrapper với các tham số từ UI
        wrapper = make_qsilk_wrapper_like = make_qwen_wrapper(
            strength=float(strength),
            retention=float(retention),
            ratio=float(ratio),
            amplification=float(amp)
        )
        
        # Register wrapper vào patcher của Forge
        unet.set_model_unet_function_wrapper(wrapper)
        
        # Gán lại UNet đã patch cho process
        p.sd_model.forge_objects.unet = unet
        
        # Lưu params vào generation info
        p.extra_generation_params["QwenOpt Enabled"] = True
        p.extra_generation_params["QwenOpt Strength"] = strength
        p.extra_generation_params["QwenOpt Retention"] = retention
        p.extra_generation_params["QwenOpt Injection"] = ratio
        p.extra_generation_params["QwenOpt Amplification"] = amp
        
        logging.info(f"Qwen Optimizer: Active (Injection Ratio: {ratio})")

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        pass

register_processor(QwenOptimizerProcessor)
