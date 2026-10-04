"""TrashDiffusion one-shot installer for a Forge checkout (classic or neo).

    python tools/install.py --forge-root <path-to-forge>

Does everything, skipping what already exists:
  1. adapter   : link (default) or --copy sd-webui-forge-classic/ into
                 <forge>/extensions/TrashDiffusion
  2. share patch: --share [gradio|cloudflare] in modules/cmd_args.py +
                 cloudflare_share.py + 2-line webui.py hook
  3. cloudflared: standalone binary into ~/.cache/trashdiffusion/ (no root)

Opt-outs: --no-share, --no-cloudflared, --copy, --deb (dpkg flow, needs root),
--force (redo instead of skip), --revert (undo share patch + remove adapter).
Stdlib only.
"""

import argparse
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

TAG = "[TrashDiffusion]"
REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_SRC = REPO_ROOT / "sd-webui-forge-classic"
HOOK_SRC = REPO_ROOT / "tools" / "forge_patches" / "cloudflare_share.py"
RELEASE_BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download"
CACHE_DIR = Path.home() / ".cache" / "trashdiffusion"
BAK = ".trashdiffusion.bak"

SHARE_OLD = 'parser.add_argument("--share", action="store_true", help="use share=True for gradio and make the UI accessible through their site")'
SHARE_NEW = 'parser.add_argument("--share", nargs="?", const="gradio", default=None, choices=["gradio", "cloudflare"], help="share backend: bare --share (or --share gradio) = gradio tunnel; --share cloudflare = Cloudflare tunnel (TrashDiffusion)")'
HOOK_OLD = "    from modules.shared_cmd_options import cmd_opts\n\n    launch_api = cmd_opts.api"
HOOK_NEW = "    from modules.shared_cmd_options import cmd_opts\n\n    # TrashDiffusion: --share cloudflare (revert: tools/install.py --revert)\n    import cloudflare_share  # noqa: E402\n    cloudflare_share.maybe_hijack_share(cmd_opts, cmd_opts.port if cmd_opts.port else 7860)\n\n    launch_api = cmd_opts.api"


def log(msg):
    print(f"{TAG} {msg}", flush=True)


def step_adapter(forge_root, name, copy, force):
    dest = forge_root / "extensions" / name
    if dest.is_symlink() and dest.resolve() == ADAPTER_SRC.resolve():
        log("adapter: skip (linked)")
        return
    if dest.exists() or dest.is_symlink():
        if not force:
            log("adapter: skip (exists, --force to replace)")
            return
        dest.unlink() if dest.is_symlink() or dest.is_file() else shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(ADAPTER_SRC, dest)
    else:
        try:
            dest.symlink_to(ADAPTER_SRC, target_is_directory=True)
        except OSError as e:
            raise SystemExit(f"{TAG} symlink failed ({e}); retry as admin or use --copy")
    log(f"adapter: {'copied' if copy else 'linked'} -> {dest}")


def replace_once(path, old, new, what):
    text = path.read_text(encoding="utf-8")
    if new in text:
        log(f"{what}: skip (patched)")
        return
    if old not in text:
        log(f"{what}: skip (anchor mismatch, upstream changed?)")
        return
    bak = path.with_name(path.name + BAK)
    if not bak.exists():
        shutil.copy2(path, bak)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    log(f"{what}: patched")


def step_share(forge_root):
    cmd_args = forge_root / "modules" / "cmd_args.py"
    webui = forge_root / "webui.py"
    if not cmd_args.is_file() or not webui.is_file():
        log("share patch: skip (not a Forge checkout)")
        return
    replace_once(cmd_args, SHARE_OLD, SHARE_NEW, "share flag")
    hook_dst = forge_root / "cloudflare_share.py"
    if hook_dst.is_file() and hook_dst.read_bytes() == HOOK_SRC.read_bytes():
        log("hook module: skip (deployed)")
    else:
        if hook_dst.is_file():
            bak = hook_dst.with_name(hook_dst.name + BAK)
            if not bak.exists():
                shutil.copy2(hook_dst, bak)
        shutil.copy2(HOOK_SRC, hook_dst)
        log("hook module: deployed")
    replace_once(webui, HOOK_OLD, HOOK_NEW, "webui hook")


def cloudflared_asset():
    m = platform.machine().lower()
    arm = m in ("aarch64", "arm64")
    s = platform.system()
    if s == "Linux":
        return "cloudflared-linux-arm64" if arm else "cloudflared-linux-amd64"
    if s == "Windows":
        return "cloudflared-windows-arm64.exe" if arm else "cloudflared-windows-amd64.exe"
    if s == "Darwin":
        return "cloudflared-darwin-arm64.tgz" if arm else "cloudflared-darwin-amd64.tgz"
    raise SystemExit(f"{TAG} unsupported OS: {s}")


def step_cloudflared(deb):
    found = shutil.which("cloudflared")
    if found:
        log(f"cloudflared: skip ({found})")
        return
    if deb:
        name = cloudflared_asset() + ".deb"
        deb_path = Path.home() / name
        log(f"cloudflared: downloading {name}")
        with urllib.request.urlopen(f"{RELEASE_BASE}/{name}") as r, open(deb_path, "wb") as f:
            shutil.copyfileobj(r, f)
        if subprocess.run(["dpkg", "-i", str(deb_path)]).returncode != 0:
            raise SystemExit(f"{TAG} dpkg failed; try sudo")
        log("cloudflared: installed via dpkg")
        return
    name = cloudflared_asset()
    dest = CACHE_DIR / ("cloudflared.exe" if platform.system() == "Windows" else "cloudflared")
    if dest.exists():
        log(f"cloudflared: skip ({dest})")
        return
    log(f"cloudflared: downloading {name}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if name.endswith(".tgz"):
        import tarfile
        tgz = CACHE_DIR / name
        with urllib.request.urlopen(f"{RELEASE_BASE}/{name}") as r, open(tgz, "wb") as f:
            shutil.copyfileobj(r, f)
        with tarfile.open(tgz) as tf:
            member = next(m for m in tf.getmembers() if m.isfile())
            member.name = dest.name
            tf.extract(member, CACHE_DIR)
    else:
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(f"{RELEASE_BASE}/{name}") as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dest)
    if platform.system() != "Windows":
        dest.chmod(0o755)
    log(f"cloudflared: cached ({dest})")


def revert(forge_root, name):
    for rel in ("modules/cmd_args.py", "webui.py"):
        p, bak = forge_root / rel, forge_root / (rel + BAK)
        if bak.is_file():
            shutil.copy2(bak, p)
            bak.unlink()
            log(f"reverted: {rel}")
    hook = forge_root / "cloudflare_share.py"
    hbak = forge_root / ("cloudflare_share.py" + BAK)
    if hbak.is_file():
        shutil.copy2(hbak, hook)
        hbak.unlink()
        log("reverted: cloudflare_share.py")
    elif hook.is_file() and hook.read_bytes() == HOOK_SRC.read_bytes():
        hook.unlink()
        log("removed: cloudflare_share.py")
    dest = forge_root / "extensions" / name
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
        log(f"removed: {dest}")
    elif dest.is_dir():
        shutil.rmtree(dest)
        log(f"removed: {dest}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TrashDiffusion one-shot Forge installer.")
    ap.add_argument("--forge-root", required=True)
    ap.add_argument("--name", default="TrashDiffusion")
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-share", action="store_true")
    ap.add_argument("--no-cloudflared", action="store_true")
    ap.add_argument("--deb", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args(argv)
    forge_root = Path(a.forge_root)
    if a.revert:
        revert(forge_root, a.name)
        return 0
    if not ADAPTER_SRC.is_dir():
        raise SystemExit(f"{TAG} adapter missing: {ADAPTER_SRC}")
    if not (forge_root / "launch.py").exists():
        log("warning: no launch.py, not a Forge checkout?")
    step_adapter(forge_root, a.name, a.copy, a.force)
    if not a.no_share:
        step_share(forge_root)
    if not a.no_cloudflared:
        step_cloudflared(a.deb)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
