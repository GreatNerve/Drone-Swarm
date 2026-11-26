"""Local viewer server (ported from swarmflow.server): localhost only, a fixed set of static files, JSON API.

GET  /api/terrain   static heightfield and fixed obstacles (sent once)
GET  /api/frame     the current frame of the scene source
GET  /api/trace     every Layer 3 decision made so far
POST /api/reset     {"seed", "drones", "formation"} -> restart the scene
POST /api/pause     {"paused": bool}
"""
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ..config import FORMATIONS, DemoConfig, ViewerConfig
from ..io.files import dumps

ASSETS = Path(__file__).parent / "web"
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8"),
          "/vendor/three.module.min.js": ("vendor/three.module.min.js", "text/javascript; charset=utf-8")}


class Clock:
    """Scene time that can be paused and restarted."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.origin, self.paused_at = time.monotonic(), None

    def now(self):
        return (self.paused_at or time.monotonic()) - self.origin

    def pause(self, paused):
        if paused and self.paused_at is None:
            self.paused_at = time.monotonic()
        elif not paused and self.paused_at is not None:
            self.origin += time.monotonic() - self.paused_at
            self.paused_at = None


def make_server(scene, config=None, scene_factory=None):
    """scene: an object with terrain(), frame(t) and trace(t, limit). scene_factory(seed, drones, formation)
    builds a new one for /api/reset (default: the Phase 0 demo)."""
    config = config or ViewerConfig()
    if scene_factory is None:
        from .demo import DemoScene

        def scene_factory(seed, drones, formation):
            return DemoScene(DemoConfig(drones=drones, formation=formation), seed=seed)

    state = {"scene": scene, "clock": Clock(), "seed": getattr(scene, "seed", None)}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def respond(self, value, status=200, content_type="application/json"):
            body = dumps(value).encode() if content_type == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            with lock:
                scene, now = state["scene"], state["clock"].now()
                if route == "/api/terrain":
                    self.respond({**scene.terrain(), "seed": state["seed"]})
                    return
                if route == "/api/frame":
                    self.respond({**scene.frame(now), "paused": state["clock"].paused_at is not None})
                    return
                if route == "/api/trace":
                    self.respond({"time": round(now, 3), "trace": scene.trace(now)})
                    return
            if route not in STATIC:
                self.respond({"error": "Not found"}, 404)
                return
            name, mime = STATIC[route]
            self.respond((ASSETS / name).read_bytes(), content_type=mime)

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{self.headers.get('Host')}":
                    self.respond({"error": "Cross-origin requests are not accepted"}, 403)
                    return
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                route = urlparse(self.path).path
                with lock:
                    if route == "/api/reset":
                        current = state["scene"]
                        seed = int(data.get("seed", state["seed"] if state["seed"] is not None else 1000))
                        drones = int(data.get("drones", getattr(getattr(current, "config", None), "drones", 8)))
                        formation = data.get("formation", getattr(getattr(current, "config", None), "formation",
                                                                  "DIAMOND"))
                        if seed < 0 or formation not in FORMATIONS:
                            raise ValueError(f"seed must be >= 0 and formation one of {FORMATIONS}")
                        state.update(scene=scene_factory(seed, drones, formation), seed=seed)
                        state["clock"].reset()
                        self.respond({"ok": True, "seed": seed})
                    elif route == "/api/pause":
                        state["clock"].pause(bool(data.get("paused", True)))
                        self.respond({"ok": True, "paused": state["clock"].paused_at is not None})
                    else:
                        self.respond({"error": "Not found"}, 404)
            except (ValueError, TypeError) as error:
                self.respond({"error": str(error)}, 400)

    return ThreadingHTTPServer((config.host, config.port), Handler)


def serve(scene, config=None, open_browser=True):
    server = make_server(scene, config)
    url = f"http://{server.server_address[0]}:{server.server_port}/"
    print(f"Swarm-V2 viewer: {url}  (Ctrl+C to stop)", flush=True)
    print("Phase 0 demo: scripted poses, no physics; the frame schema is the one the simulator will use.", flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
