"""
Launch the modern web app with uvicorn, optionally exposing a public URL.

Local development::

    python launch.py                 # serves on http://localhost:8000

Public share (Colab or a quick demo for someone else)::

    python launch.py --share         # prints a public *.trycloudflare.com URL

``--share`` downloads the ``cloudflared`` binary on first use (Linux x86-64)
and opens a Cloudflare Quick Tunnel — no account, no token, no signup. If a
``NGROK_AUTHTOKEN`` environment variable is set and ``pyngrok`` is installed,
ngrok is used instead.

The tunnel is the recommended way to publish from Colab, where the trained
checkpoints already live on the mounted Drive. For a longer-lived deployment,
see the Dockerfile and README in this directory.
"""
from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOUDFLARED = HERE / "cloudflared"
CF_DOWNLOAD = ("https://github.com/cloudflare/cloudflared/releases/latest/"
               "download/cloudflared-linux-amd64")
_TRYCF = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")


def _ensure_cloudflared() -> Path:
    """Download the cloudflared binary next to this script if missing."""
    if CLOUDFLARED.exists():
        return CLOUDFLARED
    print("[launch] downloading cloudflared…", flush=True)
    urllib.request.urlretrieve(CF_DOWNLOAD, CLOUDFLARED)
    CLOUDFLARED.chmod(CLOUDFLARED.stat().st_mode | stat.S_IEXEC)
    return CLOUDFLARED


def _start_cloudflared(port: int):
    """Open a Cloudflare Quick Tunnel; return ``(process, public_url|None)``.

    cloudflared writes to a log file (not a pipe) so its buffer can never
    fill and stall the tunnel while we poll the file for the public URL.
    """
    binary = _ensure_cloudflared()
    log_path = HERE / "cloudflared.log"
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        [str(binary), "tunnel", "--no-autoupdate", "--url",
         f"http://localhost:{port}"],
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 40
    while time.time() < deadline:
        time.sleep(0.5)
        match = _TRYCF.search(log_path.read_text(errors="ignore"))
        if match:
            return proc, match.group(0)
    return proc, None


def _start_ngrok(port: int):
    """Open an ngrok tunnel if pyngrok is available; return the URL or None."""
    try:
        from pyngrok import ngrok
    except ImportError:
        return None
    token = os.environ.get("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)
    return str(ngrok.connect(port).public_url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--share", action="store_true",
                        help="Expose a public URL via cloudflared (or ngrok).")
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on code changes (development).")
    args = parser.parse_args()

    tunnel_proc = None
    if args.share:
        public = _start_ngrok(args.port) if os.environ.get("NGROK_AUTHTOKEN") else None
        if not public:
            tunnel_proc, public = _start_cloudflared(args.port)
        banner = public or "(tunnel URL not detected yet — check cloudflared.log)"
        print("\n" + "=" * 64)
        print(f"  Public URL:  {banner}")
        print("=" * 64 + "\n", flush=True)

    sys.path.insert(0, str(HERE))
    import uvicorn
    try:
        uvicorn.run("server:app", host=args.host, port=args.port,
                    reload=args.reload, log_level="info")
    finally:
        if tunnel_proc is not None:
            tunnel_proc.terminate()


if __name__ == "__main__":
    main()
