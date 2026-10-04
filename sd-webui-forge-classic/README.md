# TrashDiffusion — Forge Classic extension

Thin Forge adapter. All guidance logic lives in `../core/` (pure torch, no
Forge/Comfy/Gradio imports). This adapter only does UI + UNet patching via
`core.forge_compat` (multi-layout fallback), so when upstream
[sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic)
restructures `modules/` / `backend/`, things degrade to warnings instead of
import-time crashes.

## Install (don't clone the whole repo into extensions)

```bash
# Option 1: symlink (recommended, picks up updates on git pull)
ln -s /path/to/TrashDiffusion/sd-webui-forge-classic \
      /path/to/sd-webui-forge-classic/extensions/TrashDiffusion

# Windows (Admin PowerShell):
New-Item -ItemType SymbolicLink `
  -Path "<forge>\extensions\TrashDiffusion" `
  -Target "<repo>\sd-webui-forge-classic>"

# Option 2: copy
cp -r sd-webui-forge-classic <forge>/extensions/TrashDiffusion
```

Or run: `python tools/install.py --forge-root <path-to-forge>` (also applies `--share cloudflare` + cloudflared).
(default: symlink).

## Requirements

See `requirements.txt` (currently: `kornia`).

## Layout

```text
sd-webui-forge-classic/
    README.md            # this file
    requirements.txt
    scripts/
        guidance_pack_script.py   # single entry point (AlwaysVisible Script)
        guidance_pack/
            __init__.py           # bootstraps core/ onto sys.path
            base.py registry.py
            processors/           # one file per guidance method
        _legacy/                  # experimental samplers, NOT autoloaded by Forge
```
