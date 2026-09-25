from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TaskState:
    shared_target: np.ndarray
    slot_positions: np.ndarray | None
    aim_points: np.ndarray


class Task:
    def reset(self, rng: np.random.Generator) -> TaskState:
        raise NotImplementedError

    def step(self, t: float) -> TaskState:
        raise NotImplementedError
