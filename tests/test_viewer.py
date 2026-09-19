import matplotlib
import numpy as np

matplotlib.use("Agg")

from swarm.config import load, repo_root
from swarm.viewer.python_view import PythonView, _load_obj


def test_shared_quadrotor_body_loads():
    verts, faces = _load_obj(repo_root() / "assets" / "drone" / "quadrotor.obj")
    assert verts.shape[1] == 3
    assert verts.shape[0] > 20
    assert faces.shape[0] > 20
    assert faces.min() >= 0
    assert faces.max() < len(verts)


def test_python_view_draws_scaled_body():
    cfg = load()
    view = PythonView(cfg, demo=True, headless=True)
    assert view._verts.shape[0] > 20
    assert view._vis_arm >= float(cfg["collision_cm"]) * 0.9
    n = int(cfg["num_drones"])
    snap = {
        "pos": np.zeros((n, 3)) + np.array([0.0, 0.0, 200.0]),
        "rot": np.repeat(np.eye(3)[None], n, axis=0),
        "shared_target": np.array([0.0, 0.0, 200.0]),
        "slots": None,
        "test_id": 1,
        "geometry_kind": None,
    }
    view.draw(snap, {"step": 1, "total": 10, "mode": "baseline", "policy": "baseline"}, force=True)
    view.close()
