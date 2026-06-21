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
_CONFIG_LOCAL = Path(__file__).resolve().parent.parent / "config.local.yaml"

# Keys the config tab can manage: flat UI field -> (section, key) in config.local.yaml.
# NOTE: this is an UNAUTHENTICATED LAN endpoint -- fine for a personal robot on a home
# network, but it can read (masked) and write secrets. Values are never returned in full.
_CONFIG_FIELDS = {
    "gemini_api_key": ("gemini", "api_key"),
    "cartesia_api_key": ("cartesia", "api_key"),
    "mem0_api_key": ("mem0", "api_key"),
    "upstash_vector_rest_url": ("upstash", "vector_rest_url"),
    "upstash_vector_rest_token": ("upstash", "vector_rest_token"),
    "upstash_redis_rest_url": ("upstash", "redis_rest_url"),
    "upstash_redis_rest_token": ("upstash", "redis_rest_token"),
}


def _read_local() -> dict:
    if not _CONFIG_LOCAL.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(_CONFIG_LOCAL.read_text()) or {}
    except Exception:
        return {}


def _mask(value: str) -> str:
    if not value:
        return ""
    v = str(value)
    return ("•" * 4 + v[-4:]) if len(v) > 4 else "•" * len(v)


def _config_status() -> dict:
    """Per field: whether it's set + a masked hint. Never the real value."""
    local = _read_local()
    out = {}
    for flat, (sec, key) in _CONFIG_FIELDS.items():
        val = (local.get(sec) or {}).get(key, "")
        out[flat] = {"set": bool(val), "hint": _mask(val)}
    return out


def _save_config(updates: dict) -> list:
    """Merge non-empty updates into config.local.yaml. Returns the fields actually written."""
    import yaml
    local = _read_local()
    written = []
    for flat, value in (updates or {}).items():
        if flat not in _CONFIG_FIELDS or not str(value).strip():
            continue  # unknown field or blank -> leave existing value untouched
        sec, key = _CONFIG_FIELDS[flat]
        local.setdefault(sec, {})[key] = str(value).strip()
        written.append(flat)
    if written:
        _CONFIG_LOCAL.write_text(yaml.safe_dump(local, default_flow_style=False, sort_keys=False))
    return written

_clients: "set[queue.Queue]" = set()
_recent: "collections.deque[str]" = collections.deque(maxlen=40)  # replay on (re)connect
_lock = threading.Lock()
_started = False
_trigger = None  # callback(name, **kwargs) -> run a tool remotely (GET /tool?name=dance&move=zoo)


def register_trigger(fn) -> None:
    """Let the agent expose its tools so they can be fired from the browser/curl."""
    global _trigger
    _trigger = fn


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

    def _tool(self) -> None:
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        name = (q.get("name") or [""])[0]
        kwargs = {k: v[0] for k, v in q.items() if k != "name"}
        try:
            result = str(_trigger(name, **kwargs)) if (_trigger and name) else "no trigger/name"
        except Exception as e:
            result = f"error: {e}"
        body = result.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode() or "{}")
        except Exception:
            return {}

    def do_GET(self):
        try:
            if self.path == "/events":
                self._events()
            elif self.path.startswith("/tool"):
                self._tool()
            elif self.path == "/config":
                self._json(_config_status())          # masked status only
            elif self.path in ("/", "/index.html", "/diagram.html"):
                self._html()
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            if self.path == "/config":
                written = _save_config(self._read_body())
                self._json({"ok": True, "written": written})
            elif self.path == "/restart":
                # Apply new keys: ask the daemon to restart this app (server-side, no CORS).
                def _do():
                    import urllib.request
                    try:
                        urllib.request.urlopen(
                            "http://localhost:8000/api/apps/restart-current-app",
                            data=b"", timeout=8)
                    except Exception as e:
                        print(f"[webview] restart failed: {e}", flush=True)
                self._json({"ok": True, "restarting": True})
                threading.Timer(0.4, _do).start()
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
