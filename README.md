# TrashDiffusion

Personal guidance/sampling experiments, structured so upstream restructures
don't nuke everything. Single source of truth + thin adapters.

```text
TrashDiffusion/
    core/                      # PURE torch. No Forge/Comfy/Gradio imports.
        guidance.py            # GuidanceState/Pipeline + FDG/CFG-Zero/CFG-Ctrl...
        asag.py                # ASAG attention helpers
        forge_compat.py        # ONLY place that knows Forge layouts (fallbacks)
        _bootstrap.py          # locate repo root -> sys.path (no ../../../ hacks)
        nodes/                 # backend-agnostic node definitions (FDG, ...)
    sd-webui-forge-classic/    # Forge extension root (symlink/copy into extensions/)
        scripts/
            guidance_pack_script.py   # single AlwaysVisible entry
            guidance_pack/             # UI + processors (thin, call core.*)
            _legacy/                  # quarantined experiments, not autoloaded
    ComfyUI/                   # placeholder, intentionally empty for now
    tools/install.py           # one-shot installer (adapter + --share cloudflare + cloudflared)
    docs/                      # notes
```

## Install (Forge)

```bash
python tools/install.py --forge-root <path-to-sd-webui-forge-classic>
```

Details: `sd-webui-forge-classic/README.md`.

## ComfyUI

Not done — `ComfyUI/` is an empty placeholder. When built, thin wrappers around
`core.nodes`.

## Hard rules

1. `core/` must not import `modules`, `backend`, `comfy`, `gradio`,
   `ldm_patched` at top level. Forge access only via `core.forge_compat`
   (lazy + fallback).
2. No `sys.path.append("../../../")` — use
   `core._bootstrap.ensure_repo_on_path`.
3. No direct `p.sd_model.forge_objects.unet` in processors — use
   `get_unet(p)` / `set_unet(p, unet)`.
4. One broken processor must not kill the whole pack (per-module try/except
   at load time).

## Guidance / Sampling status

- [x] FDG, CFG-Zero, CFG-Ctrl, ASAG (+ init noise)
- [ ] Experimental: Anima Overdrive, VNA variants, Human Denoiser
