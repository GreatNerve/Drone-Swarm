"""Opt-in safety filter around any controller (shielding; Alshiekh et al., AAAI 2018).

Each drone's commanded motion is rolled out for `horizon` seconds with the same first-order flight
response the predictive teacher plans with, against every sensed neighbour and object moving at its
current velocity. If that prediction comes within `margin` metres of contact, the drone flies the
predictive baseline's command instead for at least `hold` seconds. Everything else is the wrapped
controller's own command. Results with the filter are reported separately from the policy alone, with
the share of drone decisions it took over.
"""
import numpy as np

from .baseline import PredictiveController
from .geometry import limit_norm

CEILING_M = 10.0


class SafetyFilter:
    def __init__(self, controller, config, margin=0.1, horizon=1.0, hold=0.5, teacher=None):
        if margin < 0 or horizon <= 0 or hold < 0:
            raise ValueError("Safety filter margin and hold must be nonnegative and horizon positive")
        self.controller = controller
        self.config = config
        self.margin, self.horizon, self.hold = float(margin), float(horizon), float(hold)
        self.teacher = teacher or PredictiveController(config)
        self.times = np.arange(0.1, horizon + 1e-9, 0.1)
        self.hold_steps = max(1, int(round(hold / config.dt)))
        self.reset()

    def reset(self):
        self.remaining = {}
        self.interventions = 0
        self.decisions = 0

    def describe(self):
        return {"margin_m": self.margin, "horizon_s": self.horizon, "hold_s": self.hold,
                "teacher": "predictive baseline (soft / hard margin 0.35 / 0.08 m)"}

    def episode_stats(self):
        return {"filter_interventions": self.interventions, "filter_decisions": self.decisions}

    def predicted_gap(self, observation, action):
        """Smallest predicted surface gap (m) over the horizon if this drone flies `action`."""
        c = self.config
        own = observation["self"]
        velocity = own[3:6].astype(float)
        command = limit_norm(np.asarray(action, dtype=float) * c.max_speed, c.max_speed)
        v, p = velocity.copy(), np.zeros(3)
        positions = np.zeros((len(self.times), 3))
        for k in range(len(self.times)):
            v = v + limit_norm((command - v) / c.response_time, c.max_accel) * 0.1
            p = p + v * 0.1
            positions[k] = p
        altitude = float(own[15]) + positions[:, 2]
        gap = min(float((altitude - c.radius).min()), float((CEILING_M - c.radius - altitude).min()))
        for kind in ("neighbors", "obstacles"):
            rows = observation[kind]
            rows = rows[rows[:, -1] > 0.5]
            if not len(rows):
                continue
            other = rows[:, None, :3] + (rows[:, None, 3:6] + velocity) * self.times[None, :, None]
            gaps = np.linalg.norm(positions[None] - other, axis=-1) - rows[:, 6:7] - c.radius
            gap = min(gap, float(gaps.min()))
        return gap

    def __call__(self, observations):
        actions = self.controller(observations)
        result = {}
        for agent, observation in observations.items():
            self.decisions += 1
            remaining = self.remaining.get(agent, 0)
            if remaining == 0 and self.predicted_gap(observation, actions[agent]) < self.margin:
                remaining = self.hold_steps
            if remaining > 0:
                result[agent] = self.teacher.action(observation)
                self.remaining[agent] = remaining - 1
                self.interventions += 1
            else:
                result[agent] = actions[agent]
        return result
