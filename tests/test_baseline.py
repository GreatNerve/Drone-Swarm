import numpy as np

from swarm.baseline.pid import GeometricPid
from swarm.config import load
from swarm.env.swarm_env import SwarmEnv


def test_baseline_cmds_in_range():
    cfg = load()
    env = SwarmEnv(cfg, test_id=1, seed=0)
    obs = env.reset()
    pid = GeometricPid.from_cfg(cfg, test_id=1)
    act = pid.step(obs)
    assert act.shape == (int(cfg["num_drones"]), 4)
    assert np.all(act >= -1.0) and np.all(act <= 1.0)
    step = env.step(act)
    assert step.obs.shape == obs.shape
    assert not np.isnan(step.rewards).any()


def test_pid_holds_standoff_around_shared_target():
    cfg = load()
    env = SwarmEnv(cfg, test_id=1, seed=0, eval_mode=True)
    pid = GeometricPid.from_cfg(cfg, test_id=1)
    obs = env.reset()
    for _ in range(400):
        obs = env.step(pid.step(obs)).obs
    pos = env.positions()
    tgt = env.task_state.shared_target
    from swarm.gates.pass_gate import ring_error, standoff_cm

    dist = ring_error(pos, tgt, standoff_cm(cfg))
    assert not env.out_of_bounds
    assert not env.floor_hit
    assert dist.mean() < 55.0
    assert np.all(dist < 90.0)
    pair = env._pair_distances(pos).min()
    assert pair >= float(cfg["collision_cm"]) * 0.9


def test_pid_holds_geometry_slots():
    cfg = load()
    from swarm.gates.pass_gate import evaluate_episode

    n = int(float(cfg["episode_s"]) * float(cfg["control_hz"]))
    for kind in ("square", "circle"):
        env = SwarmEnv(cfg, test_id=2, seed=0, eval_mode=True)
        env.set_geometry_kind(kind)
        pid = GeometricPid.from_cfg(cfg, test_id=2)
        obs = env.reset()
        for _ in range(n):
            obs = env.step(pid.step(obs)).obs
        gate = evaluate_episode(env.history, cfg, 2, env.collided, env.out_of_bounds)
        assert gate["passed"], (kind, gate)


def test_pid_holds_after_live_rectangle_switch():
    cfg = load()
    from swarm.gates.pass_gate import evaluate_episode

    env = SwarmEnv(cfg, test_id=2, seed=0, eval_mode=True)
    pid = GeometricPid.from_cfg(cfg, test_id=2)
    obs = env.reset()
    n = int(float(cfg["episode_s"]) * float(cfg["control_hz"]))
    switch = n // 3
    for i in range(n):
        if i == switch:
            env.set_geometry_kind("rectangle")
        obs = env.step(pid.step(obs)).obs
    gate = evaluate_episode(env.history, cfg, 2, env.collided, env.out_of_bounds)
    assert gate["passed"], gate


def test_pid_holds_moving_target_ring():
    cfg = load()
    from swarm.gates.pass_gate import evaluate_episode

    env = SwarmEnv(cfg, test_id=3, seed=0, eval_mode=True)
    pid = GeometricPid.from_cfg(cfg, test_id=3)
    obs = env.reset()
    n = int(float(cfg["episode_s"]) * float(cfg["control_hz"]))
    for _ in range(n):
        obs = env.step(pid.step(obs)).obs
    gate = evaluate_episode(env.history, cfg, 3, env.collided, env.out_of_bounds)
    assert gate["passed"], gate


def test_pid_splits_altitude_on_xy_conflict():
    cfg = load()
    env = SwarmEnv(cfg, test_id=2, seed=0, eval_mode=True)
    pid = GeometricPid.from_cfg(cfg, test_id=2)
    env.reset()
    env.drones[0].pos[:] = [-25.0, 0.0, 200.0]
    env.drones[1].pos[:] = [25.0, 0.0, 200.0]
    env.drones[0].vel[:] = [80.0, 0.0, 0.0]
    env.drones[1].vel[:] = [-80.0, 0.0, 0.0]
    for _ in range(80):
        env.step(pid.step(env._obs()))
    dz = abs(float(env.positions()[0, 2] - env.positions()[1, 2]))
    assert dz > 8.0
    assert not env.floor_hit


def test_pid_close_xy_does_not_yeet():
    # 1/r² repulsion blew up at ~2 cm XY; keep them inside the arena.
    cfg = load()
    env = SwarmEnv(cfg, test_id=2, seed=0, eval_mode=True)
    pid = GeometricPid.from_cfg(cfg, test_id=2)
    env.reset()
    env.drones[0].pos[:] = [0.0, 0.0, 200.0]
    env.drones[1].pos[:] = [2.0, 0.0, 212.0]
    for _ in range(120):
        act = pid.step(env._obs())
        assert np.all(np.isfinite(act))
        env.step(act)
    assert not env.out_of_bounds
    assert np.all(np.abs(env.positions()[:, :2]) < 300.0)
