# Init file for guidance_pack package.
#
# Bootstraps the repo root (parent of ``core/``) onto sys.path so that
# ``from core...`` imports work no matter how Forge installed this extension
# (real checkout, symlink, or nested copy). Runs before any processor import.

import sys
from pathlib import Path

try:
    from core._bootstrap import ensure_repo_on_path

    ensure_repo_on_path(__file__)
except ImportError:
    _start = Path(__file__).resolve().parent
    for _cand in [_start, *_start.parents]:
        if (_cand / "core").is_dir():
            if str(_cand) not in sys.path:
                sys.path.insert(0, str(_cand))
            break
