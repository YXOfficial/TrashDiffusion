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
        **kwargs  # <-- QUAN TRỌNG: Dùng **kwargs để hứng các tham số dư thừa từ WebUI truyền vào
):
    """
    Euler sampler integrated with Look-Ahead and Look-Back trajectory smoothing.
    Tương thích hoàn toàn với AUTOMATIC1111 / Forge KDiffusionSampler.
    """
    # -----------------------------------------------------------------------
    # CẤU HÌNH THAM SỐ TỪ BÀI BÁO (Bạn có thể chỉnh sửa trực tiếp ở đây)
    # -----------------------------------------------------------------------
    # Look-Back
    use_lookback = kwargs.get('use_lookback', True)
    lb_lambda = kwargs.get('lb_lambda', 0.1)  # Khuyến nghị: 0.1
    lb_gamma_max = kwargs.get('lb_gamma_max', 0.9)
    lb_beta = kwargs.get('lb_beta', 1.0)
    lb_xi_star = kwargs.get('lb_xi_star', 0.0)  # Điểm lật decay (0 hoặc 0.25)

    # Look-Ahead
    use_lookahead = kwargs.get('use_lookahead', True)
    la_tau_curv = kwargs.get('la_tau_curv', 1.0)  # Khuyến nghị: 1.0 đến 10.0
    la_gamma = kwargs.get('la_gamma', 0.9)  # Khuyến nghị: 0.9 hoặc 0.95
    la_eps = kwargs.get('la_eps', 1e-8)
    # -----------------------------------------------------------------------

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # Khởi tạo bộ nhớ cho thuật toán (Memory states)
    z_bar_prev = x.clone()
    v_prev = None
    x_prev = None

    for i in trange(len(sigmas) - 1, disable=disable):
        # 1. Stochasticity (Nhiễu Karras nguyên bản)
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
        # [LOOK-BACK] Tính toán nội suy trước khi gọi model
        # ---------------------------------------------------------
        if use_lookback:
            # SNR-Aware Decay Scheduling (Eq 12)
            xi_t = -2.0 * torch.log(sigma_hat.clamp(min=1e-8))
            gamma_t = lb_gamma_max * torch.sigmoid(lb_beta * (xi_t - lb_xi_star))
            # Reshape để nhân an toàn với tensor x[Batch, Channels, H, W]
            gamma_t = gamma_t.view(-1, *([1] * (x.ndim - 1)))

            # Trộn state hiện tại với EMA state (Eq 10)
            z_peek = (1 - lb_lambda) * x + lb_lambda * z_bar_prev
        else:
            z_peek = x

        # ---------------------------------------------------------
        # 2. GỌI MÔ HÌNH (Denoise)
        # ---------------------------------------------------------
        denoised = model(z_peek, sigma_hat * s_in, **extra_args)
        d = to_d(z_peek, sigma_hat, denoised)  # 'd' tương đương 'v' trong phương trình

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        # Cập nhật state Look-Back cho bước sau
        if use_lookback:
            z_bar_prev = gamma_t * z_bar_prev + (1 - gamma_t) * x

        # ---------------------------------------------------------
        # [LOOK-AHEAD] Cổng độ cong (Curvature Gate)
        # ---------------------------------------------------------
        x_tilde = x + d * dt  # Dự đoán Euler nguyên thủy

        if use_lookahead and v_prev is not None and x_prev is not None:
            # Ước tính độ cong
            diff_v = (d - v_prev).view(d.shape[0], -1)
            diff_x = (x - x_prev).view(x.shape[0], -1)

            kappa = torch.linalg.norm(diff_v, dim=1) / (torch.linalg.norm(diff_x, dim=1) + la_eps)
            kappa = kappa.view(-1, *([1] * (x.ndim - 1)))

            # Eq 8: Chấp nhận toàn bộ bước nếu độ cong nhỏ, ngược lại thì kìm hãm
            x_next = torch.where(
                kappa <= la_tau_curv,
                x_tilde,
                x + la_gamma * (x_tilde - x)
            )
        else:
            x_next = x_tilde

        # Cập nhật state Look-Ahead cho bước sau
        if use_lookahead:
            v_prev = d.clone()
            x_prev = x.clone()

        # Áp dụng thay đổi
        x = x_next

    return x