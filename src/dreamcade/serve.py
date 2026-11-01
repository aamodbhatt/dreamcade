"""Static file server for the browser player (standard library only).

It sends the cross-origin isolation headers (COOP/COEP) so onnxruntime-web may use
multi-threaded WebAssembly; a plain ``python -m http.server`` works too, single-threaded.
"""

from __future__ import annotations

import functools
import http.server
import webbrowser
from pathlib import Path


class IsolatedHandler(http.server.SimpleHTTPRequestHandler):
    """Adds COOP/COEP and correct MIME types for ES modules and models."""

    extensions_map = {  # noqa: RUF012 - class attribute required by the base class
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".wasm": "application/wasm",
    }

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        """Quiet: the player polls nothing, so request logs are noise."""


def serve(web: Path, host: str, port: int, open_browser: bool = False) -> None:
    """Serve ``web`` until interrupted."""
    if not (web / "index.html").exists():
        msg = f"{web} has no index.html; pass --web pointing at the repository's web/ folder"
        raise SystemExit(msg)
    handler = functools.partial(IsolatedHandler, directory=str(web))
    with http.server.ThreadingHTTPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/"
        print(f"dreamcade is running at {url}  (Ctrl+C to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
