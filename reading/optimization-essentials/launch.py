#!/usr/bin/env python3
"""Open the packaged reading/apps, or run one of its optional Drake demos."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import webbrowser

ROOT = Path(__file__).resolve().parent
DEMOS = {
    "gn-ik-replay": ("02-gauss-newton-ik/gn_ik_panda.py", ["--replay", "--app"]),
    "ik-nlp": ("03-ik-as-nlp/ik_nlp_panda.py", ["--app"]),
    "constrained-ik-replay": ("04-constrained-ik-al/constrained_ik_panda.py", ["--replay", "--app"]),
    "ik-nlp-al-replay": ("05-ik-nlp-al/ik_nlp_al_panda.py", ["--replay", "--app"]),
}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", nargs="?", choices=DEMOS)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--app-port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.demo:
        script, flags = DEMOS[args.demo]
        command = [sys.executable, str(ROOT / "drake" / script), *flags]
        if "--app" in flags:
            command += ["--app-port", str(args.app_port)]
        raise SystemExit(subprocess.call(command, cwd=ROOT))
    handler = partial(Handler, directory=str(ROOT))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        address = f"http://127.0.0.1:{server.server_port}/README.html"
        print(f"Open {address}", flush=True)
        print("Keep this terminal open. Press Ctrl+C to stop.", flush=True)
        if not args.no_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
