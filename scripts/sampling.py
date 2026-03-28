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
    K-Diffusion Safe Euler with Look-Ahead Trajectory Smoothing.
    Tuyệt đối không dùng Momentum để tránh triệt tiêu CFG.
    """
    # -----------------------------------------------------------------------
    # CẤU HÌNH LOOK-AHEAD (Đã tinh chỉnh cho K-Diffusion)
    # -----------------------------------------------------------------------
    use_lookahead = kwargs.get('use_lookahead', True)
    la_gamma = kwargs.get('la_gamma', 0.95)  # Khi bẻ lái gắt, chỉ hãm phanh 5% (giảm độ dài step x_tilde)
    la_threshold = kwargs.get('la_threshold', 0.1)  # Ngưỡng Cosine Distance (0 là đi thẳng, 0.1 là bắt đầu lệch)
    # -----------------------------------------------------------------------

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # Chỉ lưu vector d của bước trước để đo góc lệch
    d_prev = None

    for i in trange(len(sigmas) - 1, disable=disable):
        # 1. Karras Stochasticity (Giữ nguyên gốc, không đụng chạm)
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

        # 2. Gọi model (Đảm bảo CFG hoạt động 100% công suất)
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        # Bước Euler tiêu chuẩn
        x_tilde = x + d * dt

        # ---------------------------------------------------------
        # 3. [LOOK-AHEAD] Phanh thông minh dựa trên góc lệch (Cosine)
        # ---------------------------------------------------------
        if use_lookahead and d_prev is not None:
            # Duỗi vector ra để tính góc lệch giữa 2 steps
            d_flat = d.view(d.shape[0], -1)
            d_prev_flat = d_prev.view(d_prev.shape[0], -1)

            # Tính Cosine Similarity (1 = đi thẳng tắp, -1 = quay đầu)
            cos_sim = torch.nn.functional.cosine_similarity(d_flat, d_prev_flat, dim=1)
            cos_dist = 1.0 - cos_sim  # Chuyển thành distance (0 = đi thẳng)
            cos_dist = cos_dist.view(-1, *([1] * (x.ndim - 1)))

            # Nếu quỹ đạo lệch gắt (> la_threshold), ta co ngắn bước nhảy lại (nhân với 0.95)
            # để tránh bị vỡ artifact ở rìa chi tiết. Nếu quỹ đạo thẳng, đi 100%.
            x_next = torch.where(
                cos_dist <= la_threshold,
                x_tilde,
                x + la_gamma * (x_tilde - x)
            )
        else:
            x_next = x_tilde

        # Lưu d hiện tại cho bước so sánh tiếp theo
        if use_lookahead:
            d_prev = d.clone()

        x = x_next

    return x