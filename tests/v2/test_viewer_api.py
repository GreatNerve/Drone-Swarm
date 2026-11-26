import json
import re
import threading
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from swarmv2.config import ViewerConfig
from swarmv2.viewer.demo import DemoScene
from swarmv2.viewer.scene import validate_frame
from swarmv2.viewer.server import ASSETS, make_server

urlopen = build_opener(ProxyHandler({})).open


@pytest.fixture()
def base():
    server = make_server(DemoScene(seed=1000), ViewerConfig(port=0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def get(url):
    return json.load(urlopen(url))


def post(url, body, headers=None):
    request = Request(url, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", **(headers or {})})
    return json.load(urlopen(request))


def test_static_files_are_served_without_any_external_source(base):
    for path, kind in (("/", "text/html"), ("/app.js", "text/javascript"), ("/style.css", "text/css"),
                       ("/vendor/three.module.min.js", "text/javascript")):
        response = urlopen(base + path)
        assert response.status == 200 and response.headers["Content-Type"].startswith(kind)
    page = (ASSETS / "index.html").read_text(encoding="utf-8")
    script = (ASSETS / "app.js").read_text(encoding="utf-8")
    assert not re.search(r"(src|href)=\"https?://", page)
    assert re.findall(r"^import .* from '([^']+)'", script, re.MULTILINE) == ["/vendor/three.module.min.js"]
    with pytest.raises(HTTPError) as missing:
        urlopen(base + "/../pyproject.toml")
    assert missing.value.code == 404


def test_terrain_frame_and_trace_endpoints(base):
    terrain = get(base + "/api/terrain")
    assert terrain["cells"] == len(terrain["heights"]) and terrain["seed"] == 1000
    frame = validate_frame(get(base + "/api/frame"))
    assert len(frame["drones"]) == 8 and len(frame["wind"]) == 49 and frame["decision"] is not None
    assert frame["paused"] is False
    assert get(base + "/api/trace")["trace"][0]["index"] == 0


def test_reset_pause_and_rejected_requests(base):
    before = get(base + "/api/terrain")["heights"]
    assert post(base + "/api/reset", {"seed": 1234, "formation": "LINE"})["seed"] == 1234
    assert get(base + "/api/terrain")["heights"] != before
    post(base + "/api/pause", {"paused": True})
    first = get(base + "/api/frame")
    time.sleep(0.15)
    second = get(base + "/api/frame")
    assert first["paused"] and first["time"] == second["time"]
    post(base + "/api/pause", {"paused": False})
    for body, headers, code in (({"seed": 1}, {"Origin": "http://evil.example"}, 403),
                                ({"formation": "CIRCLE"}, None, 400), ({"seed": -5}, None, 400)):
        with pytest.raises(HTTPError) as error:
            post(base + "/api/reset", body, headers)
        assert error.value.code == code
