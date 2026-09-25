import numpy as np

from swarm.config import load
from swarm.gates.pass_gate import evaluate_episode


def _frame(pos, vel=None, rot=None, aim=None):
    n = pos.shape[0]
    if vel is None:
        vel = np.zeros_like(pos)
    if rot is None:
        rot = np.repeat(np.eye(3)[None, ...], n, axis=0)
    if aim is None:
        aim = np.zeros_like(pos)
    return {
        "pos": pos,
        "vel": vel,
        "rot": rot,
        "aim": aim,
        "shared_target": np.asarray(aim[0], dtype=float).copy(),
    }


def test_gate_pass_and_fail_on_synthetic():
    cfg = load()
    n = int(cfg["num_drones"])
    hz = int(cfg["control_hz"])
    hold = int(float(cfg["hold_s"]) * hz)
    pos = np.array(
        [[15.0, 15.0, 200.0], [15.0, -15.0, 200.0], [-15.0, 15.0, 200.0], [-15.0, -15.0, 200.0]],
        dtype=float,
    )[:n]
    aim = np.tile(np.array([0.0, 0.0, 200.0]), (n, 1))
    hist = [_frame(pos, aim=aim) for _ in range(hold + 10)]
    ok = evaluate_episode(hist, cfg, 1, collided=False, out_of_bounds=False)
    assert ok["passed"]
    bad = evaluate_episode(hist, cfg, 1, collided=True, out_of_bounds=False)
    assert not bad["passed"]
    far = pos.copy()
    far += 300.0
    hist_far = [_frame(far, aim=aim) for _ in range(hold + 10)]
    fail = evaluate_episode(hist_far, cfg, 1, collided=False, out_of_bounds=False)
    assert not fail["passed"]
    short = evaluate_episode(hist[:3], cfg, 1, collided=True, out_of_bounds=False)
    assert not short["passed"]
    assert short["error"] is not None


def test_gate_test1_scores_ring_not_point():
    """Four drones cannot occupy the shared target; the score is centimetres off the ring."""
    cfg = load()
    n = int(cfg["num_drones"])
    hz = int(cfg["control_hz"])
    hold = int(float(cfg["hold_s"]) * hz)
    st = float(cfg["test1_dist_cm"]) * 0.75
    z = 200.0
    on_ring = np.array(
        [[st, 0.0, z], [0.0, st, z], [-st, 0.0, z], [0.0, -st, z]],
        dtype=float,
    )[:n]
    on_point = np.tile(np.array([0.0, 0.0, z]), (n, 1))
    aim = np.tile(np.array([0.0, 0.0, z]), (n, 1))
    ring = evaluate_episode(
        [_frame(on_ring, aim=aim) for _ in range(hold + 10)],
        cfg,
        1,
        collided=False,
        out_of_bounds=False,
    )
    stacked = evaluate_episode(
        [_frame(on_point, aim=aim) for _ in range(hold + 10)],
        cfg,
        1,
        collided=False,
        out_of_bounds=False,
    )
    assert ring["passed"]
    assert ring["error"] < stacked["error"]
    assert not stacked["passed"]
    assert not stacked["space_ok"]
