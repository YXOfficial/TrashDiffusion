# ComfyUI — placeholder

This directory reserves space for the ComfyUI adapter (custom nodes).
**Nothing implemented here yet.**

All guidance logic is pure torch in `../core/` (backend-agnostic), so building
the ComfyUI side only needs thin node wrappers importing from `core.nodes` —
no logic copied, no Forge imports.

Planned:

```text
ComfyUI/
    __init__.py          # NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
    nodes_*.py           # thin wrappers around core.nodes.*
```

Future install: symlink this directory into
`ComfyUI/custom_nodes/CrazyDiffusion-ComfyUI`.
