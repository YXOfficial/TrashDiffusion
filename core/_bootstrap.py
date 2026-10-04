"""Find the repo root (the directory that contains ``core/``) and put it on sys.path.

Why this exists
---------------
Forge upstream (Haoming02/sd-webui-forge-classic) restructures ``modules/``,
``backend/`` and extension loading aggressively. Our extension must NOT depend
on:

* the current working directory,
* a hardcoded ``../../../`` relative depth,
* ``sys.path`` hacks scattered in every processor.

Instead, every Forge-side entry point calls :func:`ensure_repo_on_path`
first. It walks upwards from the calling file, finds the directory that
contains the ``core/`` package (our single source of truth), and inserts it
at ``sys.path[0]``. Symlinks (extension installed via symlink) are resolved.

Layout::

    CrazyDiffusion/
        core/                    <- pure torch, zero Forge/Comfy/Gradio imports
        sd-webui-forge-classic/  <- thin Forge adapter (this extension)
        ComfyUI/                 <- placeholder for now
"""

from __future__ import annotations

import sys
from pathlib import Path

_CORE_DIR_NAME = "core"
_MAX_SEARCH_DEPTH = 8


def find_repo_root(start: Path | str | None = None) -> Path | None:
    """Return the repo root (parent of ``core/``) or ``None`` if not found."""
    if start is None:
        start = Path(__file__).resolve()
    else:
        start = Path(start).resolve()
    if start.is_file():
        start = start.parent
    current = start
    for _ in range(_MAX_SEARCH_DEPTH):
        if (current / _CORE_DIR_NAME).is_dir():
            return current
        # Also accept the case where start is *inside* core/ itself.
        if current.name == _CORE_DIR_NAME and current.is_dir():
            return current.parent
        if current.parent == current:
            break
        current = current.parent
    return None


def ensure_repo_on_path(start: Path | str | None = None) -> Path | None:
    """Ensure repo root is on ``sys.path``. Returns root or ``None``."""
    root = find_repo_root(start)
    if root is None:
        return None
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
