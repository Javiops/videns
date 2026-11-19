"""Local draft tool: http://127.0.0.1:8030 (stdlib only, localhost only; also the API of videns.gg/draft.html)."""
from __future__ import annotations

import json
import mimetypes
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from draft.engine import DraftEngine  # noqa: E402

TOOL = ROOT / "tool"
PORT = 8030
ENGINE = DraftEngine(ROOT)
LOCK = threading.Lock()
META = json.dumps(ENGINE.meta()).encode()
# The public site (videns.gg/draft.html, GitHub Pages) calls this API from the
# browser, so its origin and local static servers may read it. No cookies.
ORIGINS = re.compile(r"^(https://(www\.)?videns\.gg|https://javiops\.github\.io|https?://(localhost|127\.0\.0\.1)(:\d+)?|null)$")


def ids(values, n=None):
    out = [int(v or 0) for v in (values or [])]
    if n is not None:
        out = (out + [0] * n)[:n]
    return out


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Vary", "Origin")
        origin = self.headers.get("Origin")
        if origin and ORIGINS.match(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            # Chrome asks before a public page reaches a loopback server
            if self.headers.get("Access-Control-Request-Private-Network") == "true":
                self.send_header("Access-Control-Allow-Private-Network", "true")

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, (TOOL / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/meta":
            return self._send(200, META)
        if path.startswith("/static/"):
            f = (TOOL / "static" / path[len("/static/"):]).resolve()
            if f.is_file() and f.is_relative_to((TOOL / "static").resolve()):
                return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            blue, red, bans = ids(body.get("blue"), 5), ids(body.get("red"), 5), ids(body.get("bans"))
            with LOCK:
                if self.path == "/api/state":
                    out = ENGINE.state(blue, red)
                elif self.path == "/api/recommend":
                    side, role = int(body["side"]), int(body["role"])
                    if side not in (0, 1) or not 0 <= role < 5:
                        raise ValueError("side/role out of range")
                    out = ENGINE.recommend(blue, red, bans, side, role, pool=ids(body.get("pool")) or None,
                                           minute=int(body.get("minute", 12)))
                else:
                    return self._send(404, b'{"error":"not found"}')
            self._send(200, json.dumps(out).encode())
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            self._send(400, json.dumps({"error": str(e)}).encode())

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"draft tool on http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()
