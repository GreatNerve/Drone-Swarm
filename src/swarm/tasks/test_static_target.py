from __future__ import annotations

import numpy as np

from swarm.tasks.base import Task, TaskState


class StaticTargetTask(Task):
    def __init__(self, num_drones: int, shared_target: np.ndarray):
        self.num_drones = num_drones
        self._target = np.asarray(shared_target, dtype=float)
        self._joystick: np.ndarray | None = None

    def set_joystick_target(self, xyz: np.ndarray | None) -> None:
        self._joystick = None if xyz is None else np.asarray(xyz, dtype=float)

    def reset(self, rng: np.random.Generator) -> TaskState:
        self._joystick = None
        return self.step(0.0)

    def step(self, t: float) -> TaskState:
        target = self._target if self._joystick is None else self._joystick
        aim = np.repeat(np.asarray(target, dtype=float)[None, :], self.num_drones, axis=0)
        return TaskState(aim[0].copy(), None, aim)
