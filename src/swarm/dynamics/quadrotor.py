from __future__ import annotations

import numpy as np

G = 981.0


def hover_action(mass_kg: float, max_thrust: float) -> float:
    hover = (float(mass_kg) * G) / 4.0
    return float(np.clip(2.0 * hover / float(max_thrust) - 1.0, -0.999, 0.999))


def _skew(omega: np.ndarray) -> np.ndarray:
    x, y, z = omega
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _orthonormalize(rot: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rot)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def mix_matrix(arm_cm: float, yaw_coeff: float) -> np.ndarray:
    """Rows: total thrust, roll, pitch, yaw. Cols: FR, RR, RL, FL."""
    l = arm_cm / np.sqrt(2.0)
    k = yaw_coeff
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [l, -l, -l, l],
            [l, l, -l, -l],
            [-k, k, k, -k],
        ]
    )


def allocate(force_z: float, tau: np.ndarray, arm_cm: float, yaw_coeff: float, max_thrust: float) -> np.ndarray:
    m = mix_matrix(arm_cm, yaw_coeff)
    wrench = np.array([force_z, tau[0], tau[1], tau[2]])
    thrust = np.linalg.lstsq(m, wrench, rcond=None)[0]
    return np.clip(thrust, 0.0, max_thrust)


def _exp_so3(omega: np.ndarray, dt: float) -> np.ndarray:
    angle = float(np.linalg.norm(omega) * dt)
    if angle < 1e-8:
        return np.eye(3) + _skew(omega) * dt
    axis = omega / (np.linalg.norm(omega) + 1e-12)
    k = _skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


class Quadrotor:
    """X-config rigid body. Body +z is thrust. Motors: 0 FR, 1 RR, 2 RL, 3 FL."""

    def __init__(
        self,
        mass_kg: float,
        arm_cm: float,
        max_thrust: float,
        motor_lag: bool = False,
        yaw_coeff: float = 6.0,
        inertia_diag: np.ndarray | None = None,
    ):
        self.mass_kg = float(mass_kg)
        self.arm_cm = float(arm_cm)
        self.max_thrust = float(max_thrust)
        self.motor_lag = motor_lag
        self.yaw_coeff = float(yaw_coeff)
        diag = [23.0, 23.0, 40.0] if inertia_diag is None else inertia_diag
        self.inertia = np.diag(np.asarray(diag, dtype=float))
        self.inertia_inv = np.linalg.inv(self.inertia)
        self._thrust_filt = np.zeros(4)
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.rot = np.eye(3)
        self.omega = np.zeros(3)

    @property
    def hover_thrust(self) -> float:
        return (self.mass_kg * G) / 4.0

    def reset(self, pos: np.ndarray, yaw: float, vel: np.ndarray) -> None:
        self.pos = np.asarray(pos, dtype=float).copy()
        self.vel = np.asarray(vel, dtype=float).copy()
        c, s = np.cos(yaw), np.sin(yaw)
        self.rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.omega = np.zeros(3)
        self._thrust_filt[:] = self.hover_thrust

    def motors_from_action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return 0.5 * (a + 1.0) * self.max_thrust

    def action_from_motors(self, thrust: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * (np.asarray(thrust) / self.max_thrust) - 1.0, -1.0, 1.0)

    def step(self, action: np.ndarray, dt: float) -> None:
        thrust = self.motors_from_action(action)
        if self.motor_lag:
            self._thrust_filt = 0.6 * self._thrust_filt + 0.4 * thrust
            thrust = self._thrust_filt
        thrust = np.clip(thrust, 0.0, self.max_thrust)
        wrench = mix_matrix(self.arm_cm, self.yaw_coeff) @ thrust
        acc = (self.rot @ np.array([0.0, 0.0, wrench[0]])) / self.mass_kg + np.array([0.0, 0.0, -G])
        self.vel = np.clip(self.vel + acc * dt, -800.0, 800.0)
        self.pos = self.pos + self.vel * dt
        tau = wrench[1:]
        omega_dot = self.inertia_inv @ (tau - np.cross(self.omega, self.inertia @ self.omega))
        self.omega = np.clip(self.omega + omega_dot * dt, -20.0, 20.0)
        self.rot = _orthonormalize(self.rot @ _exp_so3(self.omega, dt))
