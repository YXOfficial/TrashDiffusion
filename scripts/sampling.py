import torch
from tqdm.auto import trange


@torch.no_grad()
def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative"""
    return (x - denoised) / sigma


@torch.no_grad()
def sample_euler_lookahead_back(
        model, x, sigmas, extra_args=None, callback=None, disable=None,
        s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,

        # --- THAM SỐ CHO LOOK-BACK (Algorithm 2) ---
        use_lookback=True,
        lb_lambda=0.1,  # lambda trong Eq 10
        lb_gamma_max=0.9,  # gamma_max trong Eq 12
        lb_beta=1.0,  # beta trong Eq 12 (độ dốc)
        lb_xi_star=0.0,  # xi* trong Eq 12 (điểm giữa SNR)

        # --- THAM SỐ CHO LOOK-AHEAD (Algorithm 1) ---
        use_lookahead=True,
        la_tau_curv=1.0,  # Ngưỡng độ cong tau_curv (khuyến nghị 1.0 - 10.0)
        la_gamma=0.9,  # Hệ số nội suy gamma khi độ cong cao (Eq 8)
        la_eps=1e-8
):
    """Implements Euler steps with Look-Ahead and Look-Back trajectory smoothing"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # Khởi tạo bộ nhớ cho Look-Back
    z_bar_prev = x.clone()

    # Khởi tạo bộ nhớ cho Look-Ahead
    v_prev = None
    x_prev = None

    for i in trange(len(sigmas) - 1, disable=disable):
        # 1. Xử lý nhiễu ngẫu nhiên (Stochasticity của Karras - giữ nguyên từ code gốc)
        if s_churn > 0:
            s_gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (s_gamma + 1)
        else:
            s_gamma = 0
            sigma_hat = sigmas[i]

        if s_gamma > 0:
            eps_noise = torch.randn_like(x) * s_noise
            x = x + eps_noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        dt = sigmas[i + 1] - sigma_hat

        # ---------------------------------------------------------
        # [LOOK-BACK] Tiền xử lý input trước khi gọi model
        # ---------------------------------------------------------
        if use_lookback:
            # Tính log-SNR (xi_t). Với lịch trình Karras, SNR tỷ lệ nghịch với sigma^2
            xi_t = -2.0 * torch.log(sigma_hat.clamp(min=1e-8))

            # Tính hệ số decay theo thuật toán 2 (Eq 12)
            gamma_t = lb_gamma_max * torch.sigmoid(lb_beta * (xi_t - lb_xi_star))
            gamma_t = gamma_t.view(-1, *([1] * (x.ndim - 1)))  # Reshape cho batch

            # Trộn latent hiện tại với lịch sử EMA (Eq 10)
            z_peek = (1 - lb_lambda) * x + lb_lambda * z_bar_prev
        else:
            z_peek = x

        # ---------------------------------------------------------
        # GỌI MODEL & TÍNH VẬN TỐC (VELOCITY)
        # ---------------------------------------------------------
        denoised = model(z_peek, sigma_hat * s_in, **extra_args)
        d = to_d(z_peek, sigma_hat, denoised)  # v_k trong bài báo

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        # Cập nhật bộ nhớ EMA cho Look-Back bước tiếp theo (Eq 9)
        if use_lookback:
            z_bar_prev = gamma_t * z_bar_prev + (1 - gamma_t) * x

        # ---------------------------------------------------------
        # [LOOK-AHEAD] Hậu xử lý bước nhảy Euler
        # ---------------------------------------------------------
        x_tilde = x + d * dt  # Dự đoán Euler tiêu chuẩn (Eq 5)

        if use_lookahead and v_prev is not None and x_prev is not None:
            # Tính độ cong quỹ đạo (Curvature) theo batch
            # Thay vì gọi model 2 lần, ta so sánh vận tốc hiện tại với vận tốc bước trước
            diff_v = (d - v_prev).view(d.shape[0], -1)
            diff_x = (x - x_prev).view(x.shape[0], -1)

            kappa = torch.linalg.norm(diff_v, dim=1) / (torch.linalg.norm(diff_x, dim=1) + la_eps)
            kappa = kappa.view(-1, *([1] * (x.ndim - 1)))  # Đưa về dạng [Batch, 1, 1, 1]

            # Cổng chọn lọc (Curvature Gate) - Eq 8
            # Nếu quỹ đạo thẳng (kappa nhỏ), đi hết 100% bước. Nếu cong (kappa lớn), hãm lại
            x_next = torch.where(
                kappa <= la_tau_curv,
                x_tilde,  # Chấp nhận full step
                x + la_gamma * (x_tilde - x)  # Nội suy (hãm lại)
            )
        else:
            x_next = x_tilde  # Ở bước đầu tiên chưa có lịch sử v_prev nên chạy bình thường

        # Cập nhật lịch sử cho Look-Ahead
        if use_lookahead:
            v_prev = d.clone()
            x_prev = x.clone()

        # Di chuyển tới bước tiếp theo
        x = x_next

    return x