# ComfyUI — placeholder

Thư mục này giữ chỗ cho adapter ComfyUI (custom nodes), hiện **chưa làm gì cả**.

Toàn bộ logic guidance thuần torch nằm ở `../core/` (backend-agnostic), nên khi
làm ComfyUI chỉ cần viết node wrapper mỏng import từ `core.nodes` — không copy
logic, không dính import Forge.

Dự kiến:

```text
ComfyUI/
    __init__.py          # NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
    nodes_*.py           # wrapper mỏng quanh core.nodes.*
```

Cài đặt sau này: symlink thư mục này vào `ComfyUI/custom_nodes/CrazyDiffusion-ComfyUI`.
