import torch
from tqdm.auto import trange


@torch.no_grad()
def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative"""
    return (x - denoised) / sigma


@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x
