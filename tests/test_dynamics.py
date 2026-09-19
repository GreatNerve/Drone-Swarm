import numpy as np

from swarm.config import load
from swarm.dynamics.quadrotor import G, Quadrotor, allocate, mix_matrix


def test_hover_thrust_balances_gravity():
    cfg = load()
    q = Quadrotor(cfg["mass_kg"], cfg["arm_cm"], cfg["max_thrust"])
    q.reset([0.0, 0.0, 200.0], 0.0, [0.0, 0.0, 0.0])
    hover = q.hover_thrust
    assert abs(4 * hover - q.mass_kg * G) < 1e-6
    action = 2.0 * (hover / q.max_thrust) - 1.0
    z0 = q.pos[2]
    for _ in range(50):
        q.step([action] * 4, 0.01)
    assert not np.isnan(q.pos).any()
    assert q.pos[2] > z0 - 40.0


def test_allocate_matches_mixer():
    cfg = load()
    arm = float(cfg["arm_cm"])
    yaw = float(cfg["yaw_coeff"])
    hover = float(cfg["mass_kg"]) * G
    thrust = allocate(hover, np.zeros(3), arm, yaw, float(cfg["max_thrust"]))
    wrench = mix_matrix(arm, yaw) @ thrust
    assert abs(wrench[0] - hover) < 1e-6
    assert np.allclose(wrench[1:], 0.0, atol=1e-6)
    equal = np.full(4, hover / 4.0)
    tau = mix_matrix(arm, yaw) @ equal
    assert abs(tau[0] - hover) < 1e-6
    assert np.allclose(tau[1:], 0.0, atol=1e-6)
