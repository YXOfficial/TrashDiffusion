"""Install the Forge adapter (sd-webui-forge-classic/) into a Forge checkout.

Default is a symlink so `git pull` here is reflected instantly. Use --copy
for a snapshot install.

Usage:
    python tools/install_forge.py --forge-root D:/apps/sd-webui-forge-classic
    python tools/install_forge.py --forge-root D:/apps/sd-webui-forge-classic --copy
    python tools/install_forge.py --forge-root D:/apps/sd-webui-forge-classic --name CrazyDiffusion
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_SRC = REPO_ROOT / "sd-webui-forge-classic"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge-root", required=True, help="Path to sd-webui-forge-classic checkout")
    ap.add_argument("--name", default="CrazyDiffusion", help="Folder name under extensions/")
    ap.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = ap.parse_args()

    forge_root = Path(args.forge_root)
    dest = forge_root / "extensions" / args.name

    if not ADAPTER_SRC.is_dir():
        print(f"adapter not found: {ADAPTER_SRC}", file=sys.stderr)
        return 1
    if not (forge_root / "launch.py").exists() and not (forge_root / "webui.py").exists():
        print(f"warning: {forge_root} does not look like a Forge checkout (no launch.py/webui.py)")

    if dest.exists() or dest.is_symlink():
        print(f"target exists, removing: {dest}")
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    if args.copy:
        shutil.copytree(ADAPTER_SRC, dest)
        print(f"copied {ADAPTER_SRC} -> {dest}")
    else:
        try:
            dest.symlink_to(ADAPTER_SRC, target_is_directory=True)
        except OSError as e:
            print(f"symlink failed ({e}); retry as admin or use --copy", file=sys.stderr)
            return 1
        print(f"linked {dest} -> {ADAPTER_SRC}")

    # Core travels with the adapter path? No: core/ lives at repo root.
    # The adapter bootstraps it via core/_bootstrap.py as long as the link
    # target keeps the layout: <repo>/{core,sd-webui-forge-classic}.
    # Warn if the user copied only the inner folder somewhere detached.
    if args.copy:
        print("note: copied adapter resolves core/ via _bootstrap search; "
              "keep the repo layout intact for updates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
