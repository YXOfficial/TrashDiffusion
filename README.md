# CrazyDiffusion

Personal guidance/sampling experiments, structured so upstream restructures don't
nuke everything. Single source of truth + thin adapters ("cứng như Linux").

```text
CrazyDiffusion/
    core/                      # PURE torch. No Forge/Comfy/Gradio imports.
        guidance.py            # GuidanceState/Pipeline + FDG/ZeResFDG/CFG-Zero/...
        asag.py                # ASAG attention helpers
        forge_compat.py        # ONLY place that knows Forge layouts (fallbacks)
        _bootstrap.py          # locate repo root -> sys.path (no ../../../ hacks)
        nodes/                 # backend-agnostic node definitions (FDG, ZeResFDG, ...)
    sd-webui-forge-classic/    # Forge extension root (symlink/copy into extensions/)
        scripts/
            guidance_pack_script.py   # single AlwaysVisible entry
            guidance_pack/             # UI + processors (thin, call core.*)
            _legacy/                  # quarantined experiments, not autoloaded
    ComfyUI/                   # placeholder, intentionally empty for now
    tools/install_forge.py     # symlink/copy installer
    docs/                      # notes
```

## Install (Forge)

```bash
python tools/install_forge.py --forge-root <path-to-sd-webui-forge-classic>
```

Chi tiết: `sd-webui-forge-classic/README.md`.

## ComfyUI

Chưa làm — `ComfyUI/` để trống giữ chỗ. Khi làm, wrapper mỏng quanh `core.nodes`.

## Quy tắc cứng

1. `core/` không import `modules`, `backend`, `comfy`, `gradio`, `ldm_patched`
   ở top-level. Forge access chỉ qua `core.forge_compat` (lazy + fallback).
2. Không `sys.path.append("../../../")` — dùng `core._bootstrap.ensure_repo_on_path`.
3. Không `p.sd_model.forge_objects.unet` trực tiếp trong processor — dùng
   `get_unet(p)` / `set_unet(p, unet)`.
4. Một processor hỏng không được giết cả pack (per-module try/except khi load).

## Guidance / Sampling status

- [x] ZeResFDG, CFG-Zero, FDG, ASAG (+ init noise)
- [ ] QSilk (issue: unknown), samplers (RES/CNS thử nghiệm trong `_legacy`)
