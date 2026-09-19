import numpy as np
import pytest

from swarm.config import load
from swarm.env.swarm_env import SwarmEnv
from swarm.tasks.test_geometry import GeometryTask, slot_positions
from swarm.tasks.test_static_target import StaticTargetTask
from swarm.tasks.test_moving_target import MovingTargetTask


def test_shared_target_and_slots():
    cfg = load()
    n = int(cfg["num_drones"])
    t1 = StaticTargetTask(n, cfg["shared_target"]).reset(np.random.default_rng(0))
    assert t1.slot_positions is None
    assert np.allclose(t1.aim_points, cfg["shared_target"])
    slots = slot_positions("square", n, float(cfg["geometry_size_cm"]), 200.0)
    assert slots.shape == (n, 3)
    t3 = MovingTargetTask(n, cfg["shared_target"])
    t3.reset(np.random.default_rng(0))
    a = t3.step(0.0).shared_target
    b = t3.step(1.0).shared_target
    assert not np.allclose(a, b)
    assert abs(float(b[2] - a[2])) > 1.0


def test_joystick_overrides_scripted_target():
    t3 = MovingTargetTask(4, [0.0, 0.0, 200.0])
    t3.reset(np.random.default_rng(0))
    t3.set_joystick_target([120.0, -40.0, 200.0])
    held = t3.step(3.0).shared_target
    assert np.allclose(held, [120.0, -40.0, 200.0])


def test_geometry_kind_changes_slots_mid_run():
    g = GeometryTask(4, "square", 160.0)
    g.reset(np.random.default_rng(0))
    square = g.step(0.0).slot_positions.copy()
    g.set_kind("circle")
    circle = g.step(0.0).slot_positions
    assert g.kind == "circle"
    assert not np.allclose(square, circle)
    with pytest.raises(ValueError):
        g.set_kind("triangle")


def test_env_applies_live_geometry():
    cfg = load()
    env = SwarmEnv(cfg, test_id=2, seed=0, eval_mode=True)
    env.reset()
    env.set_geometry_kind("rectangle")
    assert env.task.kind == "rectangle"
    assert env.task_state is not None
    assert env.task_state.slot_positions is not None
    env.set_joystick_target([50.0, 0.0, 200.0])
    center = env.task_state.slot_positions.mean(axis=0)
    assert np.allclose(center[:2], [50.0, 0.0], atol=1.0)


def test_geometry_origin_follows_joystick():
    g = GeometryTask(4, "square", 160.0)
    g.reset(np.random.default_rng(0))
    before = g.step(0.0).slot_positions.copy()
    g.set_joystick_target([80.0, -40.0, 210.0])
    after = g.step(0.0).slot_positions
    assert np.allclose(after.mean(axis=0), [80.0, -40.0, 210.0], atol=1.0)
    assert not np.allclose(before, after)

