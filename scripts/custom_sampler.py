import math
import numpy as np
import torch
from tqdm.auto import trange

# Import hàm bước giải bậc hai của thuật toán RES từ thư viện k-diffusion gốc
try:
    from k_diffusion.sampling import _refined_exp_sosu_step
except ImportError:
    raise ImportError(
        "Không tìm thấy hàm '_refined_exp_sosu_step' trong k_diffusion.sampling. "
        "Hãy đảm bảo Forge / WebUI của bạn đã được cập nhật hoặc thư viện k-diffusion có hỗ trợ RES."
    )

from modules import sd_samplers, sd_samplers_common, shared

try:
    from modules.sd_samplers_kdiffusion import KDiffusionSampler
except ImportError:
    from modules.sd_samplers import KDiffusionSampler


# ── 1. Orthonormal DCT Matrices (Float32 Precision) ───────────────────────────

def dct_type_2_matrix(num_n, num_k, norm="ortho", device=None, dtype=torch.float32):
    n = np.arange(0, num_n)[None, :]
    k = np.arange(0, num_k)[:, None]
    dct_matrix = np.cos(np.pi / num_n * (n + 0.5) * k)
    if norm == "ortho":
        orthogonal_reweigh = np.concatenate([
            np.ones(1) / np.sqrt(num_n),
            np.ones(num_k - 1) * np.sqrt(2 / num_n),
        ])
        dct_matrix = dct_matrix * orthogonal_reweigh[:, None]
    return torch.from_numpy(dct_matrix).to(device=device, dtype=dtype)


# ── 2. ConvDCT (Độc lập thiết bị & kiến trúc) ─────────────────────────────────

class ConvDCT:
    def __init__(self, block_size=8):
        self.block_size = block_size

    def forward(self, x):
        orig_dtype = x.dtype
        is_5d = (x.ndim == 5)
        if is_5d:
            B, C, T, H, W = x.shape
            x_4d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        else:
            B, C, H, W = x.shape
            x_4d = x

        x_4d = x_4d.to(torch.float32)
        device = x_4d.device

        dct_matrix = dct_type_2_matrix(self.block_size, self.block_size, norm="ortho", device=device,
                                       dtype=torch.float32)
        dct_kernel = dct_matrix[:, None, :, None] * dct_matrix[None, :, None, :]
        dct_kernel = dct_kernel.reshape(-1, 1, self.block_size, self.block_size)
        dct_kernel_expanded = dct_kernel.repeat(C, 1, 1, 1)

        convolved = torch.nn.functional.conv2d(
            x_4d,
            dct_kernel_expanded,
            stride=1,
            padding=0,
            groups=C
        )
        h, w = convolved.shape[-2:]
        convolved = convolved.view(-1, C, 64, h, w)

        if is_5d:
            convolved = convolved.view(B, T, C, 64, h, w).permute(0, 2, 3, 1, 4, 5)

        return convolved.to(orig_dtype)

    def inverse(self, x):
        orig_dtype = x.dtype
        is_5d = (x.ndim == 6)
        if is_5d:
            B, C, F, T, h, w = x.shape
            x_4d = x.permute(0, 3, 1, 2, 4, 5).reshape(B * T, C, F, h, w)
        else:
            B, C, F, h, w = x.shape
            x_4d = x

        x_4d = x_4d.to(torch.float32)
        device = x_4d.device

        dct_matrix = dct_type_2_matrix(self.block_size, self.block_size, norm="ortho", device=device,
                                       dtype=torch.float32)
        dct_kernel = dct_matrix[:, None, :, None] * dct_matrix[None, :, None, :]
        dct_kernel = dct_kernel.reshape(-1, 1, self.block_size, self.block_size)
        dct_kernel_expanded = dct_kernel.repeat(C, 1, 1, 1)

        x_reshaped = x_4d.view(-1, C * F, h, w)
        inverted = torch.nn.functional.conv_transpose2d(
            x_reshaped,
            dct_kernel_expanded,
            stride=1,
            padding=0,
            groups=C
        )

        H_new, W_new = inverted.shape[-2:]
        ones = torch.ones((1, C, H_new, W_new), device=device, dtype=torch.float32)
        fwd_ones = torch.nn.functional.conv2d(ones, dct_kernel_expanded, stride=1, padding=0, groups=C)
        normalization = torch.nn.functional.conv_transpose2d(fwd_ones, dct_kernel_expanded, stride=1, padding=0,
                                                             groups=C)
        normalization = torch.clamp(normalization, min=1e-6)

        inverted = inverted / normalization

        if is_5d:
            inverted = inverted.view(B, T, C, H_new, W_new).permute(0, 2, 1, 3, 4)

        return inverted.to(orig_dtype)


# ── 3. Bộ lọc hiệp biến cấu trúc thích ứng ───────────────────────────────────────

class CovarianceNoiseFilter:
    def __init__(self, block_size=8, var_cap=1e4):
        self.block_size = block_size
        self.var_cap = var_cap

    def apply_filter(self, dx_dnoise, eps):
        dct = ConvDCT(block_size=self.block_size)

        dx_dnoise_f = dct.forward(dx_dnoise).to(torch.float32)
        eps_f = dct.forward(eps).to(torch.float32)

        var_f = torch.mean(eps_f * dx_dnoise_f, dim=1, keepdim=True)
        eps2_f = torch.mean(eps_f ** 2, dim=1, keepdim=True)

        var_f = torch.clamp(var_f, min=0.0, max=self.var_cap)
        eps2_f = torch.clamp(eps2_f, min=1e-2)
        x_var_f = torch.sqrt(var_f / eps2_f)
        x_var_f = torch.clamp(x_var_f, max=3.0)

        eps_f = eps_f * x_var_f

        out = dct.inverse(eps_f)
        out = torch.clamp(out, -50.0, 50.0)

        return out.to(dx_dnoise.dtype)


# ── 4. Thuật toán RES Solver được nâng cấp (BẢN PURE ODE) ───────────────────────
@torch.no_grad()
def sample_refined_exp_s(
        model,
        x,
        sigmas,
        denoise_to_zero=True,
        extra_args=None,
        callback=None,
        disable=None,
        ita=None,
        c2=0.5,
        noise_sampler=None,
        simple_phi_calc=False,
        momentum=0.0,
):
    extra_args = {} if extra_args is None else extra_args

    # 1. Truy tìm CNS handler từ Forge
    cns_handler = None
    try:
        cns_handler = shared.sd_model.forge_objects.unet.model_options.get("transformer_options", {}).get("cns_handler",
                                                                                                          None)
    except Exception:
        pass

    # [KHÓA CHẶT ODE]: Vứt bỏ hoàn toàn s_churn trong vòng lặp. s_churn và ita cưỡng bức bằng 0.0
    ita = torch.zeros((1,), device=x.device)

    device = x.device
    sigmas = sigmas.to(device)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

    vel, vel_2 = None, None
    s_in = x.new_ones([x.shape[0]])

    # ── [THE PURE CNS-INIT RES-ODE] ──
    # Áp dụng đổi màu CNS lên chính hạt nhiễu khởi đầu x tại Step 0
    if cns_handler is not None:
        x = cns_handler.apply(x, 0, sigmas)
        print("\n[CNS-Sampler] >>> PURE ODE ACTIVE: Đã nhào nặn màu sắc CNS thành công vào Latent khởi đầu (Step 0).")
    else:
        print("\n[CNS-Sampler] >>> PURE ODE ACTIVE: Không tìm thấy CNS, chạy RES ODE nguyên bản.")

    for i in trange(len(sigmas) - (1 if denoise_to_zero else 2), disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        time = sigma / sigma_max

        # Giai đoạn sau hoàn toàn không có nhiễu, x_hat chính là x
        x_hat = x

        # Bước giải bậc hai RES thuần khiết không bị gián đoạn động lượng
        x_next, denoised, denoised2, vel, vel_2 = _refined_exp_sosu_step(
            model,
            x_hat,
            sigma,  # sigma_hat bằng đúng sigma vì ita = 0
            sigma_next,
            c2=c2,
            extra_args=extra_args,
            pbar=None,
            simple_phi_calc=simple_phi_calc,
            momentum=momentum,
            vel=vel,
            vel_2=vel_2,
            time=time
        )

        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigma,
                'sigma_hat': sigma,
                'denoised': denoised
            })

        x = x_next

    if denoise_to_zero:
        sigma = sigmas[-2]
        x_hat = x
        x_next = model(x_hat, sigma.to(x_hat.device).repeat(x_hat.size(0)), **extra_args)

        if callback is not None:
            callback({
                'x': x,
                'i': len(sigmas) - 2,
                'sigma': sigma,
                'sigma_hat': sigma,
                'denoised': x_next
            })

        x = x_next

    return x


# ── 5. Sampler Wrapper Interface ──────────────────────────────────────────────

@torch.no_grad()
def sample_res_solver(model, x, sigmas, extra_args=None, callback=None, disable=None,
                      noise_sampler_type="gaussian", noise_sampler=None, denoise_to_zero=True,
                      simple_phi_calc=False, c2=0.5, ita=None, momentum=0.0):
    # Khóa chặt ita bằng 0.0 để vô hiệu hóa hoàn toàn s_churn
    ita = torch.tensor([0.0])

    return sample_refined_exp_s(
        model, x, sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        noise_sampler=noise_sampler,
        denoise_to_zero=denoise_to_zero,
        simple_phi_calc=simple_phi_calc,
        c2=c2,
        ita=ita,
        momentum=momentum
    )


# ── 6. Đăng ký hệ thống WebUI / Forge ──────────────────────────────────────────

def add_custom_samplers():
    new_samplers_config = [
        ("RES Solver (Covariance-Aware)", sample_res_solver, ["res_cov_solver"], {}),
    ]

    if hasattr(sd_samplers, 'all_samplers'):
        existing_names = {x.name for x in sd_samplers.all_samplers}
    else:
        existing_names = set()

    samplers_data_to_add = []

    for label, func, aliases, options in new_samplers_config:
        if label not in existing_names:
            data = sd_samplers_common.SamplerData(
                label,
                lambda model, funcname=func: KDiffusionSampler(funcname, model),
                aliases,
                options
            )
            samplers_data_to_add.append(data)

    if samplers_data_to_add:
        sd_samplers.all_samplers.extend(samplers_data_to_add)
        sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
        sd_samplers.set_samplers()
        print(f"[RES-Cov] Registered 'RES Solver (Covariance-Aware)' custom sampler successfully.")


try:
    add_custom_samplers()
except ImportError:
    pass
except Exception as e:
    print(f"[RES-Cov] Error registering sampler: {e}")