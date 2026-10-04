"""Compatibility layer between :mod:`core` and sd-webui-forge-classic.

Everything that touches Forge internals lives here and ONLY here, resolved
lazily with fallbacks. Upstream may rename ``forge_objects``, move
``calc_cond_uncond_batch`` or restructure ``modules/devices`` -- adapters
must call these helpers instead of reaching into ``p.sd_model...`` directly,
so a restructure becomes a caught exception / warning, not a crash.

Nothing in this module is imported at ``core`` import time by
:mod:`core.guidance`; callers import it explicitly.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

GUIDANCE_PARAM_KEYS = [
    "CFG-Zero Enabled",
    "CFG-Zero Init First Step",
    "FDG Enabled",
    "FDG w_low",
    "FDG w_high",
    "FDG Levels",
    "ZeResFDG Enabled",
    "ZeResFDG λ_l",
    "ZeResFDG λ_h",
    "ZeResFDG α",
    "ZeResFDG τ_lo",
    "ZeResFDG τ_hi",
    "ZeResFDG β",
    "ZeResFDG Controller",
    "S2-Guidance Enabled",
    "S2-Guidance Omega",
    "S2-Guidance Drop Ratio",
    "QSilk Enabled",
    "QSilk micro_q_low",
    "QSilk micro_q_high",
    "QSilk micro_alpha",
    "QSilk use_aqclip",
    "QSilk tile_size",
    "QSilk stride",
    "QSilk aqclip_alpha",
    "QSilk ema_beta",
    "TPSO Enabled",
    "TPSO Steps",
    "TPSO LR",
    "TPSO Lambda",
    "TPSO r",
    "CFG-Ctrl Enabled",
    "CFG-Ctrl SMC Enable",
    "CFG-Ctrl Lambda",
    "CFG-Ctrl K",
    "CFG-Ctrl Warmup Steps",
]


# ---------------------------------------------------------------------------
# UNet patcher access (p.sd_model.forge_objects.unet and friends)
# ---------------------------------------------------------------------------

_UNET_ATTR_CANDIDATES = (
    # (holder getter, unet attr) tried in order
    ("forge_objects", "unet"),
    ("forge_objects_unet", None),  # hypothetical flattened future name
    ("unet", None),
)


def _get_sd_model(p: Any) -> Any | None:
    for attr in ("sd_model", "model", "sdmodel"):
        obj = getattr(p, attr, None)
        if obj is not None:
            return obj
    return None


def get_unet(p: Any) -> Any:
    """Return the UNet patcher for a Forge processing object ``p``.

    Raises :exc:`AttributeError` with a helpful message if none of the known
    layouts match (e.g. after an upstream restructure).
    """
    sd_model = _get_sd_model(p)
    if sd_model is None:
        raise AttributeError("get_unet: processing object has no sd_model/model attribute")
    # Current layout: p.sd_model.forge_objects.unet
    forge_objects = getattr(sd_model, "forge_objects", None)
    if forge_objects is not None and getattr(forge_objects, "unet", None) is not None:
        return forge_objects.unet
    # Fallbacks for renamed layouts.
    for holder_name, unet_attr in _UNET_ATTR_CANDIDATES[1:]:
        if unet_attr is None:
            candidate = getattr(sd_model, holder_name, None)
            if candidate is not None:
                return candidate
        else:
            holder = getattr(sd_model, holder_name, None)
            candidate = getattr(holder, unet_attr, None) if holder is not None else None
            if candidate is not None:
                return candidate
    raise AttributeError(
        "get_unet: could not locate UNet patcher on sd_model "
        f"(tried forge_objects.unet, attrs={[c[0] for c in _UNET_ATTR_CANDIDATES]}). "
        "Upstream may have restructured; please update core/forge_compat.py."
    )


def set_unet(p: Any, unet: Any) -> None:
    """Set the UNet patcher back on ``p`` (mirrors :func:`get_unet`)."""
    sd_model = _get_sd_model(p)
    if sd_model is None:
        raise AttributeError("set_unet: processing object has no sd_model/model attribute")
    forge_objects = getattr(sd_model, "forge_objects", None)
    if forge_objects is not None and hasattr(forge_objects, "unet"):
        forge_objects.unet = unet
        return
    for holder_name, unet_attr in _UNET_ATTR_CANDIDATES[1:]:
        if unet_attr is None:
            if hasattr(sd_model, holder_name):
                setattr(sd_model, holder_name, unet)
                return
        else:
            holder = getattr(sd_model, holder_name, None)
            if holder is not None and hasattr(holder, unet_attr):
                setattr(holder, unet_attr, unet)
                return
    raise AttributeError("set_unet: could not locate UNet patcher slot (see get_unet).")


def reset_unet_if_needed(p: Any) -> bool:
    """Restore the pristine UNet clone at the start of each generation.

    Returns True when a restore happened. Never raises: on failure it logs
    a warning and returns False so sampling can continue.
    """
    try:
        if not hasattr(p, "_guidance_original_unet"):
            p._guidance_original_unet = get_unet(p).clone()
        if getattr(p, "_guidance_unet_restored", False):
            return False
        set_unet(p, p._guidance_original_unet.clone())
        p._guidance_unet_restored = True
        return True
    except Exception as e:
        logging.warning("Guidance: reset_unet_if_needed failed (upstream layout changed?): %s", e)
        return False


# ---------------------------------------------------------------------------
# Generation params bookkeeping
# ---------------------------------------------------------------------------

def clear_generation_params(p: Any, keys: Iterable[str]) -> None:
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict):
        return
    for key in keys:
        params.pop(key, None)


def clear_generation_params_once(p: Any, keys: Iterable[str] = GUIDANCE_PARAM_KEYS) -> bool:
    if getattr(p, "_guidance_params_cleared", False):
        return False
    clear_generation_params(p, keys)
    try:
        p._guidance_params_cleared = True
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# Lazy Forge/Comfy backend resolution (never at import time)
# ---------------------------------------------------------------------------

def get_devices() -> tuple[Any, Any]:
    """Return ``(device, dtype_unet)`` without a hard ``modules`` dependency."""
    try:
        from modules import devices as _devices

        return _devices.device, _devices.dtype_unet
    except Exception:
        pass
    try:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float16
    except Exception:
        return "cuda", None


def resolve_cond_batch_fn() -> tuple[Callable | None, str | None]:
    """Return ``(calc_fn, backend_name)`` for ASAG-style second cond pass.

    Order: Forge-classic ``backend.*`` -> reForge ``ldm_patched`` ->
    ComfyUI ``comfy``. Returns ``(None, None)`` when nothing imports, so
    callers can skip gracefully instead of crashing at import time.
    """
    try:
        from backend.sampling.sampling_function import calc_cond_uncond_batch as fn

        return fn, "Forge"
    except Exception:
        pass
    try:
        import backend.sampling.sampling_function as sampling_fn

        fn = getattr(sampling_fn, "calc_cond_uncond_batch", None)
        if fn is not None:
            return fn, "Forge"
    except Exception:
        pass
    try:
        from ldm_patched.modules.samplers import calc_cond_uncond_batch as fn

        return fn, "reForge"
    except Exception:
        pass
    try:
        from comfy.samplers import calc_cond_batch as fn

        return fn, "ComfyUI"
    except Exception:
        pass
    return None, None
