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

# Single log prefix for this extension. Forge's console mixes logs from every
# extension (Tag Autocomplete, other scripts, ...), so everything we emit at
# load time goes through TAG below — no per-processor print() allowed.
TAG = "[TrashDiffusion]"

# Import registry
from guidance_pack.registry import get_processors


def _failure_hint(exc: BaseException) -> str:
    """Short actionable hint for common import failures (else '')."""
    if isinstance(exc, ModuleNotFoundError):
        missing = str(exc).split("'")
        mod = missing[1] if len(missing) >= 2 else str(exc)
        if "ldm_patched" in mod:
            return (
                "hint: processor imports 'ldm_patched' directly; "
                "import via core.forge_compat or backend.sampling.sampling_function instead"
            )
        if mod.split(".")[0] in ("backend", "modules", "ldm", "comfy"):
            return (
                f"hint: Forge layout module {mod!r} not importable here; "
                "keep Forge imports lazy/tolerant (see core/forge_compat.py)"
            )
        return f"hint: missing third-party dependency {mod!r} (see requirements.txt)"
    if isinstance(exc, ImportError):
        return "hint: import failed — check Forge backend restructure / renamed symbol"
    return ""


def _exc_reason(exc: BaseException) -> str:
    """One-line reason: ExcType: message (at file:line)."""
    tb = traceback.extract_tb(exc.__traceback__)
    origin = f" (at {Path(tb[-1].filename).name}:{tb[-1].lineno})" if tb else ""
    msg = str(exc).strip() or "<no message>"
    return f"{type(exc).__name__}: {msg}{origin}"


def _refresh_extension_modules() -> None:
    """Drop our cached modules so Forge's Reload UI picks up `git pull` edits.

    Forge re-execs this entry file on reload, but `guidance_pack.*` would
    otherwise be served stale from sys.modules: processor code changes would
    not apply, and the load report would false-alarm every module as
    "imported OK but registered no processor". `core.*` is refreshed too,
    but only inside real Forge (harness/tests keep their stubs).
    """
    for name in [m for m in list(sys.modules)
                 if m == "guidance_pack" or m.startswith("guidance_pack.")]:
        del sys.modules[name]
    if _FORGE_AVAILABLE:
        for name in [m for m in list(sys.modules)
                     if m == "core" or m.startswith("core.")]:
            del sys.modules[name]


def _load_processors() -> dict:
    """Import every guidance_pack.processors.* module, report ONE summary.

    Returns a dict with keys: found / loaded / skipped / failed, where
    loaded = [(module, [processor names])], skipped = [(module, reason)],
    failed = [(module, reason, hint)].
    """
    global get_processors, clear_generation_params_once, reset_unet_if_needed
    import pkgutil
    import importlib

    _refresh_extension_modules()
    try:
        import guidance_pack.registry as _reg
        get_processors = _reg.get_processors
    except Exception as e:
        logging.error("%s registry re-import failed: %s", TAG, _exc_reason(e))
    if _FORGE_AVAILABLE:
        try:
            import core.forge_compat as _fc
            clear_generation_params_once = _fc.clear_generation_params_once
            reset_unet_if_needed = _fc.reset_unet_if_needed
        except Exception as e:
            logging.error("%s forge_compat re-import failed: %s", TAG, _exc_reason(e))

    import guidance_pack.processors as _pkg

    candidates = sorted(
        (m.name, m.name.rpartition(".")[2])
        for m in pkgutil.iter_modules(_pkg.__path__, _pkg.__name__ + ".")
        if not m.ispkg
    )
    loaded: list = []
    skipped: list = []
    failed: list = []

    for full_name, short in candidates:
        if short.startswith("_") or short.startswith("test"):
            skipped.append((full_name, "dev-only file, skipped by name prefix (test* / _*)"))
            continue
        before = len(get_processors())
        try:
            importlib.import_module(full_name)
        except Exception as exc:  # one broken processor must not kill the pack
            reason = _exc_reason(exc)
            hint = _failure_hint(exc)
            failed.append((full_name, reason, hint))
            logging.debug("%s processor import failed: %s\n%s", TAG, full_name, traceback.format_exc())
            continue
        new_processors = get_processors()[before:]
        if not new_processors:
            failed.append((
                full_name,
                "imported OK but registered no processor (missing register_processor() call?)",
                "",
            ))
            continue
        names: list = []
        for proc in new_processors:
            try:
                names.append(proc.name())
            except Exception as exc:
                names.append(f"<name() failed: {_exc_reason(exc)}>")
        loaded.append((full_name, names))

    total = len(candidates)
    lines = [
        f"{TAG} Processors: {total} found | {len(loaded)} loaded | "
        f"{len(skipped)} skipped | {len(failed)} failed"
    ]
    for mod, names in loaded:
        lines.append(f"{TAG}   loaded: {mod} -> {', '.join(names)}")
    for mod, reason in skipped:
        lines.append(f"{TAG}   skipped: {mod} — {reason}")
    for mod, reason, hint in failed:
        detail = f"{TAG}   failed: {mod} — {reason}"
        if hint:
            detail += f" — {hint}"
        lines.append(detail)
    summary = "\n".join(lines)
    # print(): Forge console always shows stdout; logging alone may be filtered.
    print(summary, flush=True)
    for line in lines:
        logging.info(line)
    return {"found": total, "loaded": loaded, "skipped": skipped, "failed": failed}


try:
    PROCESSOR_LOAD_REPORT = _load_processors()
except Exception as e:  # pkgutil itself broken — still exactly one TAG'd block
    PROCESSOR_LOAD_REPORT = {"found": 0, "loaded": [], "skipped": [], "failed": []}
    logging.error("%s processor discovery crashed: %s", TAG, _exc_reason(e))
    print(f"{TAG} Processors: discovery crashed — {_exc_reason(e)}", flush=True)


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
            # Fresh per ui() call (txt2img + img2img each call once): the
            # runner aggregates these into paste fields so PNG Info "Send to"
            # restores our components. Keys align with create_ui() order.
            self.infotext_fields = []
            self.paste_field_names = []
            ui_components = []
            with gr.Accordion(open=False, label=self.title()):
                gr.Markdown("Unified UI for Guidance Methods.")
                with gr.Tabs():
                    for processor in self.processors:
                        try:
                            components = processor.create_ui()
                            ui_components.extend(components)
                            self.arg_counts.append(len(components))
                            try:
                                keys = list(processor.infotext_fields())
                            except Exception:
                                keys = []
                            for comp, key in zip(components, keys):
                                self.infotext_fields.append((comp, key))
                                self.paste_field_names.append(key)
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
    except Exception as e:
        # Same TAG prefix as the load summary; full traceback at debug only.
        logging.error("%s XYZ Grid setup failed: %s", TAG, _exc_reason(e))
        logging.debug("%s XYZ traceback:\n%s", TAG, traceback.format_exc())


if _FORGE_AVAILABLE and script_callbacks is not None:
    try:
        script_callbacks.on_before_ui(on_guidance_pack_before_ui)
    except Exception as e:
        logging.warning("Guidance Pack: could not register on_before_ui (%s)", e)
