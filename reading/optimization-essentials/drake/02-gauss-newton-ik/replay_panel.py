"""Serve the replay's web interface: an embedded Meshcat view beside readable explanations, readouts, plots and
the replay controls (the pattern of drake/01b-redundancy-self-motion/live_panel.py).

HTTP threads never call Drake. The demo publishes its state after drawing (/state), serves the whole record once
(/record), and reads the page's requests (/action?do=...) from a queue in its own loop. No build step or CDN.
"""
import html
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


class ReplayPanel:
    def __init__(self, meshcat_url, record, port=0, page=None):
        source = Path(page) if page is not None else Path(__file__).parent / "panel" / "index.html"   # another demo may bring its own page
        meshcat_url = meshcat_url.replace("://localhost:", "://127.0.0.1:")   # same loopback host in parent and iframe
        self.page = source.read_text().replace("__MESHCAT_URL__", html.escape(meshcat_url, quote=True)).encode()
        self.record = json.dumps(record, allow_nan=False).encode()
        self._lock = threading.Lock()
        self._payload = None
        self._sequence = 0
        self._action_sequence = 0
        self.actions = deque()                      # (id, request) pairs, drained by the demo's loop
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                parts = urlsplit(self.path)
                path = parts.path
                if path in ("/", "/index.html"):
                    body, kind, code = panel.page, "text/html; charset=utf-8", 200
                elif path == "/record":
                    body, kind, code = panel.record, "application/json; charset=utf-8", 200
                elif path == "/state":
                    with panel._lock:
                        body = panel._payload
                    code = 200 if body is not None else 503
                    body = body or b'{"message":"Waiting for the first state."}'
                    kind = "application/json; charset=utf-8"
                elif path == "/action":
                    do = parse_qs(parts.query).get("do", [""])[0]
                    action_id = None
                    if do:
                        with panel._lock:
                            panel._action_sequence += 1
                            action_id = panel._action_sequence
                            panel.actions.append((action_id, do))
                    body, kind, code = json.dumps({"queued": do, "action_id": action_id}).encode(), "application/json; charset=utf-8", 200
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

    def pop_actions(self):
        with self._lock:
            out = list(self.actions)
            self.actions.clear()
        return out

    def start(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}/"

    def close(self):
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=2)
        self.server.server_close()
