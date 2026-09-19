from __future__ import annotations

import numpy as np

from swarm.tasks.base import Task, TaskState


class MovingTargetTask(Task):
    def __init__(self, num_drones: int, center: np.ndarray, radius_cm: float = 200.0, period_s: float = 12.0):
        self.num_drones = num_drones
        self.center = np.asarray(center, dtype=float)
        self.radius_cm = radius_cm
        self.period_s = period_s
        self._joystick: np.ndarray | None = None

    def set_joystick_target(self, xyz: np.ndarray | None) -> None:
        self._joystick = None if xyz is None else np.asarray(xyz, dtype=float)

    def reset(self, rng: np.random.Generator) -> TaskState:
        self._joystick = None
        return self.step(0.0)

    def step(self, t: float) -> TaskState:
        if self._joystick is not None:
            target = self._joystick.copy()
        else:
            # 3D Lissajous chase path (same shared target for every drone)
            ax = ay = self.radius_cm
            az = 40.0
            wx = 2.0 * np.pi / self.period_s
            wy = 2.0 * np.pi / (self.period_s * 1.5)
            wz = 2.0 * np.pi / (self.period_s * 0.85)
            target = self.center + np.array(
                [
                    ax * np.sin(wx * t),
                    ay * np.sin(wy * t + 0.6),
                    az * np.sin(wz * t),
                ]
            )
        aim = np.repeat(target[None, :], self.num_drones, axis=0)
        return TaskState(target, None, aim)
