import numpy as np

from swarm.config import load
from swarm.driver.attitude import AttitudeDriver
from swarm.dynamics.quadrotor import G, Quadrotor


def test_zero_cmd_is_hover_thrust():
    cfg = load()
    q = Quadrotor(cfg["mass_kg"], cfg["arm_cm"], cfg["max_thrust"])
    q.reset([0.0, 0.0, 200.0], 0.0, [0.0, 0.0, 0.0])
    driver = AttitudeDriver.from_cfg(cfg)
    motors = driver.motors(q.rot, q.omega, q.vel, np.zeros(4))
    thrust = q.motors_from_action(motors)
    assert np.all(np.abs(motors) <= 1.0)
    assert abs(float(thrust.sum()) - q.mass_kg * G) < 5.0


def test_decode_cmd_units():
    cfg = load()
    driver = AttitudeDriver.from_cfg(cfg)
    roll, pitch, yaw_rate, climb = driver.decode([1.0, -1.0, 0.5, 1.0])
    assert abs(roll - driver.max_tilt) < 1e-9
    assert abs(pitch + driver.max_tilt) < 1e-9
    assert abs(yaw_rate - 0.5 * driver.max_yaw_rate) < 1e-9
    assert abs(climb - driver.max_climb) < 1e-9
