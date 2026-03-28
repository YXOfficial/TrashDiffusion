import torch
import k_diffusion.sampling
from modules import sd_samplers, sd_samplers_common
try:
    from modules.sd_samplers_kdiffusion import KDiffusionSampler
except ImportError:
    from modules.sd_samplers import KDiffusionSampler

# ------------------------------------------------------------------------------
# 1. Import Euler sampler
# ------------------------------------------------------------------------------
try:
    from scripts.sampling import sample_euler
except ImportError:
    sample_euler = None
    print("[CrazyDiffusion] Warning: sample_euler not found in scripts.sampling")

# ------------------------------------------------------------------------------
# 2. Register the Sampler with SD WebUI
# ------------------------------------------------------------------------------
def add_custom_samplers():
    if sample_euler is None:
        return
        
    new_samplers_config = [
        ("Euler (Custom)", sample_euler, ["k_euler_custom"], {}),
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
        print(f"[CrazyDiffusion] Added {len(samplers_data_to_add)} custom sampler(s).")

try:
    add_custom_samplers()
except ImportError:
    pass
except Exception as e:
    print(f"[CrazyDiffusion] Error adding custom samplers: {e}")
