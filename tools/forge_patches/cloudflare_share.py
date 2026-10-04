"""Force `--share cloudflare` to mean Cloudflare tunnel (TrashDiffusion).

Deployed as `<forge>/cloudflare_share.py` by tools/install.py
together with a `--share [gradio|cloudflare]` flag in
`<forge>/modules/cmd_args.py`. Hooked from webui.py::webui_worker.

`--share` semantics after patching:
- absent               -> local only, untouched.
- `--share` / `--share gradio` -> upstream gradio tunnel, no Cloudflare.
- `--share cloudflare` -> spawn
      cloudflared tunnel --url http://localhost:<port> --protocol http2
  in a background thread, announce the public *.trycloudflare.com URL, and
  flip cmd_opts.share off so Gradio never opens its own tunnel.

Self-contained: stdlib only, zero TrashDiffusion imports (it runs inside the
Forge checkout). Binary lookup order: CLOUDFLARED_BIN env -> PATH ->
download standalone binary into ~/.cache/trashdiffusion/.

Env overrides:
    CLOUDFLARED_BIN            path to binary (skip lookup/download)
    CLOUDFLARED_PROTOCOL      default: http2 (survives Colab UDP blocks)
    CLOUDFLARED_DEB=1         Linux: .deb + dpkg flow (needs root)
    CLOUDFLARED_NO_DOWNLOAD=1 fail instead of downloading
"""

import atexit
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

TAG = "[TrashDiffusion::cloudflare-share]"
RELEASE_BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download"
CACHE_DIR = Path.home() / ".cache" / "trashdiffusion"
URL_RE = re.compile(r"https://[\w-]+\.trycloudflare\.com")

_proc = None


def _log(msg):
    print(f"{TAG} {msg}", flush=True)


def _asset_name(use_deb=False):
    system = platform.system()
    machine = platform.machine().lower()
    arm = machine in ("aarch64", "arm64")
    if use_deb:
        if system != "Linux":
            raise SystemExit(f"{TAG} .deb flow is Linux-only (current: {system})")
        return "cloudflared-linux-arm64.deb" if arm else "cloudflared-linux-amd64.deb"
    if system == "Linux":
        return "cloudflared-linux-arm64" if arm else "cloudflared-linux-amd64"
    if system == "Windows":
        return "cloudflared-windows-arm64.exe" if arm else "cloudflared-windows-amd64.exe"
    if system == "Darwin":
        return "cloudflared-darwin-arm64.tgz" if arm else "cloudflared-darwin-amd64.tgz"
    raise SystemExit(f"{TAG} unsupported OS: {system} ({machine})")


def _ensure_binary():
    explicit = os.environ.get("CLOUDFLARED_BIN")
    if explicit:
        if not Path(explicit).exists():
            raise SystemExit(f"{TAG} CLOUDFLARED_BIN not found: {explicit}")
        return explicit
    found = shutil.which("cloudflared")
    if found:
        return found
    if os.environ.get("CLOUDFLARED_NO_DOWNLOAD") == "1":
        raise SystemExit(
            f"{TAG} cloudflared not on PATH and download disabled; install manually:\n"
            f"  wget -P ~ {RELEASE_BASE}/cloudflared-linux-amd64.deb\n"
            f"  dpkg -i ~/cloudflared-linux-amd64.deb"
        )
    if os.environ.get("CLOUDFLARED_DEB") == "1":
        name = _asset_name(use_deb=True)
        deb = Path.home() / name
        _log(f"downloading {RELEASE_BASE}/{name}")
        with urllib.request.urlopen(f"{RELEASE_BASE}/{name}") as r, open(deb, "wb") as f:
            shutil.copyfileobj(r, f)
        rc = subprocess.run(["dpkg", "-i", str(deb)]).returncode
        if rc != 0:
            raise SystemExit(f"{TAG} dpkg failed (rc={rc}); try sudo or drop CLOUDFLARED_DEB")
        found = shutil.which("cloudflared")
        if not found:
            raise SystemExit(f"{TAG} dpkg claimed success but cloudflared not on PATH")
        return found
    name = _asset_name()
    dest = CACHE_DIR / ("cloudflared.exe" if platform.system() == "Windows" else "cloudflared")
    if not dest.exists():
        _log(f"downloading {RELEASE_BASE}/{name} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
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
    return str(dest)


def _watch(proc, port):
    announced = False
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if not announced:
            m = URL_RE.search(line)
            if m:
                announced = True
                print(f"{TAG} tunnel UP: WebUI public at {m.group(0)}", flush=True)
                print(f"{TAG} local stays at http://localhost:{port}", flush=True)
    proc.wait()


def maybe_hijack_share(cmd_opts, port):
    """Handle `--share [backend]`. Returns tunnel Popen or None.

    - None/False    -> untouched (no sharing at all).
    - True/"gradio" -> normalized to True; upstream gradio tunnel.
    - "cloudflare"  -> start cloudflared, force cmd_opts.share = False.
    """
    global _proc
    backend = getattr(cmd_opts, "share", None)
    if backend in (None, False):
        return None
    if backend is True or backend == "gradio":
        cmd_opts.share = True  # normalize for launch(share=...)
        return None
    if backend != "cloudflare":
        raise SystemExit(f"{TAG} unknown --share backend: {backend!r} (use: gradio | cloudflare)")
    if _proc is not None and _proc.poll() is None:
        cmd_opts.share = False
        return _proc

    binary = _ensure_binary()
    protocol = os.environ.get("CLOUDFLARED_PROTOCOL", "http2")
    cmd = [binary, "tunnel", "--url", f"http://localhost:{port}",
           "--protocol", protocol]
    _log("--share cloudflare: forcing Cloudflare tunnel, gradio share OFF")
    _log(f"starting: {' '.join(cmd)}")
    try:
        _proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
    except OSError as e:
        raise SystemExit(f"{TAG} failed to launch cloudflared: {e}")
    atexit.register(_proc.terminate)
    threading.Thread(target=_watch, args=(_proc, port), daemon=True).start()
    cmd_opts.share = False  # gradio must NOT open its own tunnel
    _log("gradio share disabled; public access comes from the tunnel above")
    return _proc
