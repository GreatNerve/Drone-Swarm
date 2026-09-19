from __future__ import annotations

import numpy as np

from swarm.driver.attitude import cmd_scales
from swarm.dynamics.quadrotor import G


def _ring_error(rel: np.ndarray, standoff_cm: float) -> np.ndarray:
    """Track a ring of radius standoff around the aim, not the point itself."""
    if standoff_cm <= 1e-6:
        return rel
    xy = rel[:2]
    dist = float(np.linalg.norm(xy))
    if dist < 1e-2:
        return np.array([standoff_cm, 0.0, rel[2]])
    n = xy / dist
    return np.array([(dist - standoff_cm) * n[0], (dist - standoff_cm) * n[1], rel[2]])


class GeometricPid:
    """obs → flight cmd [roll, pitch, yaw_rate, climb]. Driver owns motors."""

    def __init__(
        self,
        collision_cm: float = 20.0,
        standoff_cm: float = 0.0,
        max_tilt: float | None = None,
        max_climb: float | None = None,
    ):
        self.collision_cm = collision_cm
        self.standoff_cm = standoff_cm
        tilt, climb, _yaw = cmd_scales()
        self.max_tilt = tilt if max_tilt is None else float(max_tilt)
        self.max_climb = climb if max_climb is None else float(max_climb)
        self.kp = 3.5
        self.kd = 2.4
        self.k_rep = 800.0
        self.k_alt = 400.0
        self.body_z = 0.6 * collision_cm
        self._tilt_acc = G * np.tan(self.max_tilt)
        self.dt = 0.01
        self._prev_rel: np.ndarray | None = None

    @classmethod
    def from_cfg(cls, cfg: dict, test_id: int = 1) -> "GeometricPid":
        standoff = 0.0
        tid = int(test_id)
        if tid == 1:
            standoff = float(cfg["test1_dist_cm"]) * 0.75
        elif tid == 3:
            # moving target piles the pack; a wider ring keeps Chebyshev > body
            standoff = 50.0
        tilt, climb, _yaw = cmd_scales(cfg)
        pid = cls(float(cfg["collision_cm"]), standoff, tilt, climb)
        pid.dt = 1.0 / float(cfg["control_hz"])
        if tid == 3:
            pid.k_alt = 800.0
        return pid

    def step(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        if obs.ndim == 1:
            obs = obs[None, :]
        out = np.zeros((obs.shape[0], 4), dtype=np.float32)
        rels = obs[:, 0:3]
        vels = obs[:, 3:6]
        if self._prev_rel is not None and self._prev_rel.shape == rels.shape:
            aim_vel = vels - (rels - self._prev_rel) / self.dt
        else:
            aim_vel = np.zeros_like(rels)
        self._prev_rel = rels.copy()
        for i, row in enumerate(obs):
            rel = row[0:3]
            vel = row[3:6]
            err = _ring_error(rel, self.standoff_cm)
            acc = -self.kp * err - self.kd * (vel - aim_vel[i])
            # leave tilt budget for repulsion so chase does not squeeze drones together
            track = 0.8 * self._tilt_acc
            acc[0] = float(np.clip(acc[0], -track, track))
            acc[1] = float(np.clip(acc[1], -track, track))
            acc = acc + self._repulsion(row[18:])
            acc[0] = float(np.clip(acc[0], -self._tilt_acc, self._tilt_acc))
            acc[1] = float(np.clip(acc[1], -self._tilt_acc, self._tilt_acc))
            # _body_z: +pitch → +x accel, +roll → −y accel
            roll = float(np.clip(np.arctan2(-acc[1], G) / self.max_tilt, -1.0, 1.0))
            pitch = float(np.clip(np.arctan2(acc[0], G) / self.max_tilt, -1.0, 1.0))
            vz = float(
                np.clip(-self.kp * err[2] - self.kd * vel[2] + acc[2], -self.max_climb, self.max_climb)
            )
            out[i] = np.array([roll, pitch, 0.0, vz / self.max_climb], dtype=np.float32)
        return out

    def _repulsion(self, packed: np.ndarray) -> np.ndarray:
        if packed.size < 6:
            return np.zeros(3)
        nei = packed.reshape(-1, 6)
        acc = np.zeros(3)
        r_safe = max(3.0 * self.collision_cm, 45.0)
        if self.standoff_cm > 1e-6:
            r_safe = max(2.5 * self.collision_cm, self.standoff_cm)
        r_alt = 4.0 * self.collision_cm if self.standoff_cm <= 1e-6 else 1.6 * self.collision_cm
        for row in nei:
            d = row[:3]
            xy = float(max(abs(d[0]), abs(d[1])))
            if xy < r_safe and abs(d[2]) < self.body_z * 2.5:
                # 1/r² with xy≈0 yeets a drone into the arena wall
                scale = (r_safe - xy) / (xy * xy + 400.0)
                acc[:2] -= self.k_rep * scale * d[:2]
            if 1e-2 < xy < r_alt:
                # local rule, no ids: I climb if the neighbor is in -x
                acc[2] += self.k_alt * (1.0 if d[0] < 0.0 else -1.0) * (1.0 - xy / r_alt)
        return acc
