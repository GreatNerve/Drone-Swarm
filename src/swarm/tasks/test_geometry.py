from __future__ import annotations

import numpy as np

from swarm.tasks.base import Task, TaskState


def slot_positions(kind: str, n: int, size_cm: float, z: float) -> np.ndarray:
    half = size_cm / 2.0
    if kind == "square":
        corners = np.array(
            [[half, half, z], [half, -half, z], [-half, -half, z], [-half, half, z]],
            dtype=float,
        )
        if n <= 4:
            return corners[:n]
        extra = _circle(n - 4, half * 1.4, z)
        return np.vstack([corners, extra])
    if kind == "rectangle":
        hx, hy = half * 1.3, half * 0.7
        corners = np.array(
            [[hx, hy, z], [hx, -hy, z], [-hx, -hy, z], [-hx, hy, z]],
            dtype=float,
        )
        return corners[:n] if n <= 4 else np.vstack([corners, _circle(n - 4, half, z)])
    if kind == "circle":
        return _circle(n, half, z)
    raise ValueError(f"unknown geometry_kind {kind}")


def _circle(n: int, radius: float, z: float) -> np.ndarray:
    ang = 2.0 * np.pi * np.arange(n) / max(n, 1)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang), np.full(n, z)], axis=1)


class GeometryTask(Task):
    def __init__(self, num_drones: int, kind: str, size_cm: float, z: float = 200.0):
        self.num_drones = num_drones
        self.size_cm = size_cm
        self.z = z
        self.kind = kind
        self._slots = slot_positions(kind, num_drones, size_cm, z)

    def set_kind(self, kind: str) -> None:
        k = str(kind).strip().lower()
        self._slots = slot_positions(k, self.num_drones, self.size_cm, self.z)
        self.kind = k

    def reset(self, rng: np.random.Generator) -> TaskState:
        self._slots = slot_positions(self.kind, self.num_drones, self.size_cm, self.z)
        return self.step(0.0)

    def step(self, t: float) -> TaskState:
        center = self._slots.mean(axis=0)
        return TaskState(center, self._slots.copy(), self._slots.copy())
