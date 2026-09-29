# -*- coding: utf-8 -*-
"""Serve the review console against the real Python engine.

The page ships a JavaScript port of the engine (verified against Python by
eval/verify_web_port.js), so it works as a static page too.  This server adds
two things the static page cannot give you:

  * answers from the Python engine itself, live from data/kb/ with no export
    step, so you can edit the knowledge base and refresh;
  * /api/ask, if you want to drive the engine from curl or another front end.

Run:  python web/server.py            then open http://127.0.0.1:8000
      python web/server.py --port 9000 --host 0.0.0.0

Standard library only -- no Flask, no extra dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agrichat import AgriEngine  # noqa: E402

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}

ENGINE = None


class Handler(BaseHTTPRequestHandler):
    server_version = "AgriChatBot"

    def log_message(self, fmt, *args):
        if self.path.startswith("/api/"):
            sys.stderr.write("%s %s\n" % (self.command, self.path))

    # -- helpers -----------------------------------------------------------

    def _send(self, status, body, content_type):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload, ensure_ascii=False), CONTENT_TYPES[".json"])

    # -- routes ------------------------------------------------------------

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json(200, {"ok": True, "intents": len(ENGINE.entries)})
        if path in ("/", "/index.html"):
            path = "/index.html"
        return self._serve_static(path)

    def do_POST(self):
        if urlparse(self.path).path != "/api/ask":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            return self._json(400, {"error": "invalid JSON body"})

        question = (payload.get("q") or "").strip()
        if not question:
            return self._json(400, {"error": "missing 'q'"})

        result = ENGINE.answer(question)
        ranked = ENGINE.rank(question)
        return self._json(200, {
            "question": question,
            "reply": ENGINE.reply(question),
            "intent": result.entry.id,
            "score": result.score,
            "margin": result.margin,
            "confident": result.confident,
            "crops": sorted(result.crops),
            "ranked": [{"id": e.id, "score": s} for e, s in ranked[:5]],
        })

    def _serve_static(self, path):
        # Only files that actually live in web/ are servable.
        name = os.path.normpath(path.lstrip("/"))
        full = os.path.join(WEB_DIR, name)
        if not os.path.abspath(full).startswith(WEB_DIR + os.sep) or not os.path.isfile(full):
            return self._send(404, "Not found\n", "text/plain; charset=utf-8")
        ext = os.path.splitext(full)[1]
        with open(full, "rb") as fh:
            body = fh.read()
        return self._send(200, body, CONTENT_TYPES.get(ext, "application/octet-stream"))


def main() -> int:
    global ENGINE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    bundle = os.path.join(WEB_DIR, "kb-bundle.js")
    if not os.path.isfile(bundle):
        print("kb-bundle.js is missing -- run: python -m agrichat.export_web", file=sys.stderr)
        return 1

    ENGINE = AgriEngine()
    print(f"AgriChatBot: {len(ENGINE.entries)} intents loaded")
    print(f"  http://{args.host}:{args.port}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
