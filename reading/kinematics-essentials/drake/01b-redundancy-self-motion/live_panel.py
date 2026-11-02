"""Serve an embedded Meshcat page and immutable snapshots from its owning thread.

HTTP threads never call Drake. The demo publishes Python values after publishing its
scene, and the browser reads those values from /state. No build step or CDN is needed.
"""
import html
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class LivePanel:
    def __init__(self, meshcat_url, port=0):
        source = Path(__file__).parent / "panel" / "index.html"
        # Use the same loopback hostname in parent and iframe. The ports may differ.
        meshcat_url = meshcat_url.replace("://localhost:", "://127.0.0.1:")
        self.page = source.read_text().replace("__MESHCAT_URL__", html.escape(meshcat_url, quote=True)).encode()
        self._lock = threading.Lock()
        self._payload = None
        self._sequence = 0
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                path = urlsplit(self.path).path
                if path in ("/", "/index.html"):
                    body, kind, code = panel.page, "text/html; charset=utf-8", 200
                elif path == "/state":
                    with panel._lock:
                        body = panel._payload
                    code = 200 if body is not None else 503
                    body = body or b'{"message":"Waiting for the first robot state."}'
                    kind = "application/json; charset=utf-8"
                elif path == "/favicon.ico":
                    body, kind, code = b"", "image/x-icon", 204
                else:
                    body, kind, code = b"Not found", "text/plain; charset=utf-8", 404
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def publish(self, snapshot):
        self._sequence += 1
        state = {**snapshot, "sequence": self._sequence, "updated_at": time.time()}
        payload = json.dumps(state, allow_nan=False).encode()
        with self._lock:
            self._payload = payload

    def start(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}/"

    def close(self):
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=2)
        self.server.server_close()
