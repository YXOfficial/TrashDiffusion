# CrazyDiffusion — Forge Classic extension

Thin Forge adapter. Mọi logic guidance nằm ở `../core/` (torch thuần, không import
Forge/Comfy/Gradio). Adapter này chỉ làm UI + patch UNet qua `core.forge_compat`
(multi-layout fallback), nên upstream
[sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic)
restructure `modules/` / `backend/` vẫn chỉ degrade thành warning thay vì crash import.

## Cài đặt (không clone cả repo vào extensions)

```bash
# Cách 1: symlink (khuyên dùng, nhận update khi git pull repo này)
ln -s /path/to/CrazyDiffusion/sd-webui-forge-classic \
      /path/to/sd-webui-forge-classic/extensions/CrazyDiffusion

# Windows (Admin PowerShell):
New-Item -ItemType SymbolicLink `
  -Path "<forge>\extensions\CrazyDiffusion" `
  -Target "<repo>\sd-webui-forge-classic>"

# Cách 2: copy
cp -r sd-webui-forge-classic <forge>/extensions/CrazyDiffusion
```

Hoặc chạy: `python tools/install_forge.py --forge-root <path-to-forge> [--copy]`
(mặc định symlink).

## Yêu cầu

Xem `requirements.txt` (hiện tại: `kornia`).

## Layout

```text
sd-webui-forge-classic/
    README.md            # file này
    requirements.txt
    scripts/
        guidance_pack_script.py   # entry point duy nhất (AlwaysVisible Script)
        guidance_pack/
            __init__.py           # bootstrap core/ lên sys.path
            base.py registry.py
            processors/           # mỗi guidance một file
        _legacy/                  # sampler thử nghiệm, Forge không autoload
```
