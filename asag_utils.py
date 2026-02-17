import math
import torch
import torch.nn.functional as F
from typing import Any, Callable, Literal

def parse_unet_blocks(model, unet_block_list: str, attn: Literal["attn1", "attn2"] | None):
    from itertools import groupby
    output: list[tuple[str, int, int | None]] = []
    names: list[str] = []

    for name, module in model.model.diffusion_model.named_modules():
        if module.__class__.__name__ == "BasicTransformerBlock" and (attn is None or hasattr(module, attn)):
            parts = name.split(".")
            unet_part = parts[0]
            block_id = int(parts[1])
            if unet_part.startswith("input"):
                output.append((block_id, name, "input"))
            elif unet_part.startswith("middle"):
                output.append((block_id - 1, name, "middle"))
            elif unet_part.startswith("output"):
                output.append((block_id, name, "output"))

    # Logic đơn giản hóa để parse chuỗi d0,m0,u0...
    final_blocks = []
    user_inputs = [b.strip() for b in unet_block_list.split(",")]
    # Để đảm bảo chạy ổn định, ta sẽ map các block theo ID
    for user_input in user_inputs:
        if not user_input: continue
        prefix, idx = user_input[0], user_input[1:]
        part_map = {"d": "input", "m": "middle", "u": "output"}
        target_part = part_map.get(prefix)
        if target_part:
            final_blocks.append((target_part, int(idx), None))
    return final_blocks, []

def set_model_options_patch_replace(model_options, patch, name, block_name, number, transformer_index=None):
    to = model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if name not in to["patches_replace"]:
        to["patches_replace"][name] = {}
    else:
        to["patches_replace"][name] = to["patches_replace"][name].copy()

    block = (block_name, number, transformer_index) if transformer_index is not None else (block_name, number)
    to["patches_replace"][name][block] = patch
    model_options["transformer_options"] = to
    return model_options

def asag_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options, mask=None, *, sinkhorn_iters: int = 2):
    orig_dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    bh, n, d = q.shape
    heads_opt = extra_options.get("n_heads", 0)
    heads = heads_opt if heads_opt > 0 and bh % heads_opt == 0 else bh
    b = bh // heads
    q_ = q.view(b, heads, n, d)
    k_ = k.view(b, heads, n, d)
    v_ = v.view(b, heads, n, d)
    cost = torch.matmul(q_, k_.transpose(-1, -2))
    lambda_reg = 1.0 / math.sqrt(d)
    K = torch.exp(-lambda_reg * cost)
    u = torch.full((b, heads, n), 1.0 / n, device=q.device, dtype=q.dtype)
    v_vec = torch.full_like(u, 1.0 / n)
    for _ in range(sinkhorn_iters):
        Kv = torch.matmul(K, v_vec.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
        u = (1.0 / n) / Kv
        KTu = torch.matmul(K.transpose(-1, -2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
        v_vec = (1.0 / n) / KTu
    P = u.unsqueeze(-1) * K * v_vec.unsqueeze(-2)
    out = torch.matmul(P, v_).view(bh, n, d)
    return out.to(orig_dtype)

def rescale_guidance(guidance: torch.Tensor, cond_pred: torch.Tensor, cfg_result: torch.Tensor, rescale=0.0, rescale_mode="full"):
    if rescale == 0.0: return guidance
    guidance_result = cfg_result + guidance if rescale_mode == "full" else cond_pred + guidance
    std_cond = torch.std(cond_pred, dim=(1, 2, 3), keepdim=True)
    std_guidance = torch.std(guidance_result, dim=(1, 2, 3), keepdim=True)
    factor = rescale * (std_cond / (std_guidance + 1e-8)) + (1.0 - rescale)
    return guidance * factor

def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    x_kernel = (pdf / pdf.sum()).to(device=img.device, dtype=img.dtype)
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :]).expand(img.shape[-3], 1, kernel_size, kernel_size)
    padding = [kernel_size // 2] * 4
    return F.conv2d(F.pad(img, padding, mode="reflect"), kernel2d, groups=img.shape[-3])

def snf_guidance(t_guidance: torch.Tensor, s_guidance: torch.Tensor):
    b, c, h, w = t_guidance.shape
    t_omega = gaussian_blur_2d(torch.abs(t_guidance), 3, 1.0)
    s_omega = gaussian_blur_2d(torch.abs(s_guidance), 3, 1.0)
    t_softmax = torch.softmax(t_omega.reshape(b * c, h * w), dim=1).reshape(b, c, h, w)
    s_softmax = torch.softmax(s_omega.reshape(b * c, h * w), dim=1).reshape(b, c, h, w)
    argeps = torch.argmax(torch.stack([t_softmax, s_softmax], dim=0), dim=0, keepdim=True)
    return torch.gather(torch.stack([t_guidance, s_guidance], dim=0), dim=0, index=argeps).squeeze(0)
