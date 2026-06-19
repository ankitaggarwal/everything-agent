"""A tiny live window into the agent.

Serves docs/diagram.html and a Server-Sent-Events stream of the real pipeline
stages (the same ones events.py prints). Open it on the LAN -- e.g.
http://<robot-ip>:8080 -- and the diagram moves with what Reachy actually hears
and says, stamped to the millisecond.

No dependencies: stdlib http.server in a background thread. Every emit() in the
running loop is broadcast to every connected browser.
"""
from __future__ import annotations

import collections
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import events

_DIAGRAM = Path(__file__).resolve().parent.parent / "docs" / "diagram.html"

_clients: "set[queue.Queue]" = set()
_recent: "collections.deque[str]" = collections.deque(maxlen=40)  # replay on (re)connect
_lock = threading.Lock()
_started = False


def _broadcast(stage: str, data: dict) -> None:
    """events.py listener -> fan a JSON line out to every open browser."""
    msg = json.dumps({"stage": stage, "text": data.get("text", ""),
                      "ms": data.get("ms"),  # this stage's latency, for the profiler
                      "t": round(time.time() * 1000)})
    with _lock:
        _recent.append(msg)
        dead = []
        for q in _clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _clients.discard(q)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # keep journald clean
        pass

    def _html(self) -> None:
        body = _DIAGRAM.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q: queue.Queue = queue.Queue(maxsize=256)
        with _lock:
            _clients.add(q)
            backlog = list(_recent)
        try:
            # replay recent history first (so a reconnect after a restart still shows
            # the turn that just happened) -- as a separate event type the page renders
            # into the console without animating stale phases.
            for msg in backlog:
                self.wfile.write(f"event: replay\ndata: {msg}\n\n".encode())
            # then an immediate live sign-of-life
            hello = json.dumps({"stage": "listening", "text": "● connected",
                                "t": round(time.time() * 1000)})
            self.wfile.write(f"data: {hello}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=15)
                    self.wfile.write(f"data: {msg}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")  # comment -> EventSource ignores
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _lock:
                _clients.discard(q)

    def do_GET(self):
        try:
            if self.path == "/events":
                self._events()
            elif self.path in ("/", "/index.html", "/diagram.html"):
                self._html()
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass


def start(port: int = 8080) -> int | None:
    """Launch the server once, in a daemon thread. Returns the port, or None on failure."""
    global _started
    if _started:
        return port
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", int(port)), _Handler)
    except OSError as e:
        print(f"[webview] could not bind :{port} ({e}) -- live diagram disabled", flush=True)
        return None
    srv.daemon_threads = True
    events.add_listener(_broadcast)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _started = True
    print(f"[webview] live diagram at http://0.0.0.0:{port}  (open it on the LAN)", flush=True)
    return port
