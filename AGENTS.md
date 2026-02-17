# Do not add/edit/run Test-reforge, read only as document
# Do not edit README.md

# ReForge

## Model Patching
- **Core Library:** `ldm_patched`
- **UNet Wrapper:** `model.set_model_unet_function_wrapper(wrapper_func)`
  - **Warning:** This overwrites `model_options['model_function_wrapper']`. Only one wrapper can exist unless manually chained.
  - **Signature:** `wrapper_func(apply_model, args)`
  - **Args:** `{"input": x, "timestep": t, "c": conditioning_dict, "cond_or_uncond": [0, 1, ...]}`

## Conditioning Structure
- **Conditioning (`c`):** A dictionary.
- **Key Keys:**
  - SD1.5: `crossattn` (usually)
  - SDXL: `c_crossattn` (usually), `y` (pooled)
- **Batching Identification:**
  - Do NOT guess based on batch size (e.g. `[target_bs:]`).
  - **USE:** `args['cond_or_uncond']` list.
  - **Constants:** `COND = 0`, `UNCOND = 1` (Found in `ldm_patched/modules/samplers.py`).

## Timesteps
- `args['timestep']` can be **Sigma** (e.g., 14.61 start for SDXL) or **DDPM Step** (999 start).
- **Strategy:** Detect max `t` at the first step to calculate thresholds dynamically (e.g. `threshold = max_t * (1.0 - ratio)`).
