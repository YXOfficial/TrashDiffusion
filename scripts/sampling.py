import torch
from tqdm.auto import trange


@torch.no_grad()
def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative"""
    return (x - denoised) / sigma


@torch.no_grad()
def sample_euler(
        model, x, sigmas, extra_args=None, callback=None, disable=None,
        s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
        **kwargs
):
    """
    Euler sampler integrated with Momentum (Algorithm 3) and Look-Ahead.
    Safe for K-Diffusion / EDM noise schedules (Anima/Cosmos).
    """
    # -----------------------------------------------------------------------
    # CẤU HÌNH THAM SỐ (ĐÃ CHUẨN HÓA CHO K-DIFFUSION)
    # -----------------------------------------------------------------------
    # Dùng Thuật toán 3 (Momentum) thay cho Thuật toán 2 (Look-Back) để tránh vỡ ảnh
    use_momentum = kwargs.get('use_momentum', True)
    beta_1 = kwargs.get('beta_1', 0.8)  # Mặc định từ bài báo (Appendix A) cực kỳ ổn định

    # Look-Ahead (Curvature Gate)
    use_lookahead = kwargs.get('use_lookahead', True)
    la_tau_curv = kwargs.get('la_tau_curv', 5.0)  # Nới lỏng ngưỡng cong để không cản trở model
    la_gamma = kwargs.get('la_gamma', 0.95)
    # -----------------------------------------------------------------------

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # Bộ nhớ cho Momentum và Look-Ahead
    m_k = torch.zeros_like(x)
    v_prev = None
    x_prev = None

    for i in trange(len(sigmas) - 1, disable=disable):
        # 1. Xử lý nhiễu Karras nguyên bản
        if s_churn > 0:
            gamma_s = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma_s + 1)
        else:
            gamma_s = 0
            sigma_hat = sigmas[i]

        if gamma_s > 0:
            eps_noise = torch.randn_like(x) * s_noise
            x = x + eps_noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        dt = sigmas[i + 1] - sigma_hat

        # ---------------------------------------------------------
        # 2. GỌI MÔ HÌNH (Sử dụng 'x' sạch, KHÔNG trộn nhiễu cũ)
        # ---------------------------------------------------------
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        # ---------------------------------------------------------
        # 3. [MOMENTUM / LOOK-BACK AN TOÀN] (Algorithm 3)
        # ---------------------------------------------------------
        if use_momentum:
            if i == 0:
                m_k = d.clone()
            else:
                m_k = beta_1 * m_k + (1 - beta_1) * d
            step_direction = m_k
        else:
            step_direction = d

        # ---------------------------------------------------------
        # 4. [LOOK-AHEAD] Cổng độ cong
        # ---------------------------------------------------------
        x_tilde = x + step_direction * dt

        if use_lookahead and v_prev is not None and x_prev is not None:
            # Ước tính độ cong dựa trên vận tốc
            diff_v = (d - v_prev).view(d.shape[0], -1)
            diff_x = (x - x_prev).view(x.shape[0], -1)

            kappa = torch.linalg.norm(diff_v, dim=1) / (torch.linalg.norm(diff_x, dim=1) + 1e-8)
            kappa = kappa.view(-1, *([1] * (x.ndim - 1)))

            x_next = torch.where(
                kappa <= la_tau_curv,
                x_tilde,
                x + la_gamma * (x_tilde - x)
            )
        else:
            x_next = x_tilde

        # Cập nhật state cho bước sau
        if use_lookahead:
            v_prev = d.clone()
            x_prev = x.clone()

        x = x_next

    return x