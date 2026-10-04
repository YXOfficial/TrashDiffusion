import torch
from tqdm.auto import trange


@torch.no_grad()
def to_d(x, sigma, denoised):
    return (x - denoised) / sigma


@torch.no_grad()
def sample_euler(
        model, x, sigmas, extra_args=None, callback=None, disable=None,
        s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
        **kwargs
):
    """
    K-Diffusion Safe Euler with Look-Ahead Trajectory Smoothing.
    """
    # LOOK-AHEAD CONFIGURATION
    use_lookahead = kwargs.get('use_lookahead', True)
    la_gamma = kwargs.get('la_gamma', 0.9)
    la_threshold = kwargs.get('la_threshold', 0.03)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    d_prev = None

    for i in trange(len(sigmas) - 1, disable=disable):
        # Karras Stochasticity
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

        # Model call (CFG preserved)
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        # Standard Euler step
        x_tilde = x + d * dt

        # LOOK-AHEAD: Adaptive braking based on trajectory curvature (Cosine Distance)
        if use_lookahead and d_prev is not None:
            d_flat = d.view(d.shape[0], -1)
            d_prev_flat = d_prev.view(d_prev.shape[0], -1)

            cos_sim = torch.nn.functional.cosine_similarity(d_flat, d_prev_flat, dim=1)
            cos_dist = 1.0 - cos_sim
            cos_dist = cos_dist.view(-1, *([1] * (x.ndim - 1)))

            # if i % 5 == 0:
            #     print(f"[Look-Ahead Debug] Step {i:02d} | Cos_Dist Mean: {cos_dist.mean().item():.5f} | Max: {cos_dist.max().item():.5f}")

            x_next = torch.where(
                cos_dist <= la_threshold,
                x_tilde,
                x + la_gamma * (x_tilde - x)
            )
        else:
            x_next = x_tilde

        if use_lookahead:
            d_prev = d.clone()

        x = x_next

    return x
