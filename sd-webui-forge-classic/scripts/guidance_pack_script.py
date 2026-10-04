"""Guidance Pack — Forge Script entry point.

Thin adapter. All guidance math lives in ``core/``; all Forge-layout
knowledge lives in ``core/forge_compat.py``. This file only wires UI ->
processors and never reaches into ``p.sd_model...`` directly.

Install: symlink or copy the ``sd-webui-forge-classic/`` folder into
Forge's ``extensions/`` directory (see ../README.md). The bootstrap below
locates ``core/`` by searching upwards, so it works whether the extension
is a real checkout, a symlink, or a nested copy — no hardcoded ``../../../``.
"""

import logging
import sys
import traceback
from functools import partial
from pathlib import Path

# --- Bootstrap core/ on sys.path (robust to install layout) ---
try:
    from core._bootstrap import ensure_repo_on_path as _ensure
except ImportError:
    _ensure = None
    # Fallback: walk upwards looking for a dir containing "core/"
    _start = Path(__file__).resolve().parent
    for _cand in [_start, *_start.parents]:
        if (_cand / "core").is_dir():
            if str(_cand) not in sys.path:
                sys.path.insert(0, str(_cand))
            break
        if _cand.name == "core" and _cand.is_dir():
            if str(_cand.parent) not in sys.path:
                sys.path.insert(0, str(_cand.parent))
            break
else:
    _ensure(__file__)

import gradio as gr

# Forge imports are lazy-tolerant: this module must stay importable (for
# py_compile / tests) even when Forge restructures `modules/`.
try:
    from modules import scripts, script_callbacks
    _FORGE_AVAILABLE = True
except Exception as e:
    scripts = None  # type: ignore
    script_callbacks = None  # type: ignore
    _FORGE_AVAILABLE = False
    logging.warning("Guidance Pack: Forge `modules` not importable (%s). UI disabled.", e)

from core.forge_compat import (
    clear_generation_params_once,
    reset_unet_if_needed,
)

# Ensure the scripts directory is in path (common issue in some setups)
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import registry
from guidance_pack.registry import get_processors

# Dynamically import all processors. One broken processor must not kill the
# whole pack (upstream restructure often breaks a single `modules.*` import).
try:
    import pkgutil
    import importlib
    import guidance_pack.processors as _pkg

    _prefix = _pkg.__name__ + "."
    for _, _name, _is_pkg in pkgutil.iter_modules(_pkg.__path__, _prefix):
        if _is_pkg:
            continue
        _short = _name.rpartition(".")[2]
        if _short.startswith("_") or _short.startswith("test"):
            continue
        try:
            importlib.import_module(_name)
        except Exception as e:
            logging.error("Guidance Pack: skipping processor %s (%s)", _name, e)
except Exception as e:
    logging.error(f"Guidance Pack: Error loading processors: {e}")
    traceback.print_exc()


if _FORGE_AVAILABLE:

    class GuidancePackScript(scripts.Script):
        def __init__(self):
            super().__init__()
            self.processors = get_processors()
            self.arg_counts = []

        sorting_priority = 15.0

        def title(self):
            return "Guidance Pack"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, *args, **kwargs):
            self.arg_counts = []
            ui_components = []
            with gr.Accordion(open=False, label=self.title()):
                gr.Markdown("Unified UI for Guidance Methods.")
                with gr.Tabs():
                    for processor in self.processors:
                        try:
                            components = processor.create_ui()
                            ui_components.extend(components)
                            self.arg_counts.append(len(components))
                        except Exception as e:
                            logging.error(f"Guidance Pack: Error creating UI for {processor.name()}: {e}")
                            self.arg_counts.append(0)
            return ui_components

        def process_before_every_sampling(self, p, *args, **kwargs):
            # Global cleanup
            restored = reset_unet_if_needed(p)
            if restored:
                clear_generation_params_once(p)

            current_arg_index = 0
            for i, processor in enumerate(self.processors):
                if i < len(self.arg_counts):
                    count = self.arg_counts[i]
                    if count == 0:
                        continue

                    processor_args = args[current_arg_index : current_arg_index + count]
                    current_arg_index += count

                    try:
                        processor.process(p, *processor_args)
                    except Exception as e:
                         logging.error(f"Guidance Pack: Error processing {processor.name()}: {e}")
                         traceback.print_exc()
                else:
                     logging.warning(f"Guidance Pack: Argument count mismatch for {processor.name()}.")

            return

else:

    class GuidancePackScript:  # type: ignore
        """Placeholder when Forge is unavailable (import-safe)."""


def set_guidance_value(p, x, xs, *, feature: str, field: str):
    if not hasattr(p, "_guidance_xyz"):
        p._guidance_xyz = {}
    feature_map = p._guidance_xyz.setdefault(feature, {})
    feature_map[field] = str(x)


def make_guidance_axis_on_xyz_grid():
    if not _FORGE_AVAILABLE:
        return
    xyz_grid = None
    for script_data in scripts.scripts_data:
        if script_data.script_class.__module__ in ("xyz_grid.py", "xy_grid.py"):
            xyz_grid = script_data.module
            break
    if xyz_grid is None:
        return

    for processor in get_processors():
         try:
            processor.register_xyz(xyz_grid, set_guidance_value)
         except Exception as e:
            logging.error(f"Guidance Pack: Error registering XYZ for {processor.name()}: {e}")

    logging.info("Guidance Pack: XYZ Grid options registered.")


def on_guidance_pack_before_ui():
    try:
        make_guidance_axis_on_xyz_grid()
    except Exception:
        print(
            f"[-] Guidance Pack Script: Error setting up XYZ Grid options:\n{traceback.format_exc()}",
            file=sys.stderr,
        )


if _FORGE_AVAILABLE and script_callbacks is not None:
    try:
        script_callbacks.on_before_ui(on_guidance_pack_before_ui)
    except Exception as e:
        logging.warning("Guidance Pack: could not register on_before_ui (%s)", e)
