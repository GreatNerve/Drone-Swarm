from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.driver import AttitudeDriver
from swarm.dynamics.quadrotor import Quadrotor
from swarm.obs.local_obs import build_obs
from swarm.tasks import make_task


def body_z_m(collision_xy: float) -> float:
    # ponytail: Half-height of the square body. |dz| above this is a pass-through, not a hit.
    return 0.6 * float(collision_xy)


def body_overlap(pos: np.ndarray, collision_cm: float, body_z: float) -> np.ndarray:
    # ponytail: (N,N) True where square footprints overlap at the same altitude. Diagonal False.
    p = np.asarray(pos, dtype=float)
    d = p[:, None, :] - p[None, :, :]
    xy = np.maximum(np.abs(d[..., 0]), np.abs(d[..., 1]))
    z = np.abs(d[..., 2])
    overlap = (xy < float(collision_cm)) & (z < float(body_z))
    np.fill_diagonal(overlap, False)
    return overlap


def distance_cost(dist_cm: np.ndarray) -> np.ndarray:
    excess = np.maximum(np.asarray(dist_cm, dtype=float) - 10.0, 0.0)
    e = excess / 10.0
    return e + e * e


@dataclass
class StepResult:
    obs: np.ndarray
    rewards: np.ndarray
    done: bool
    info: dict


class SwarmEnv:
    def __init__(
        self,
        cfg: dict,
        test_id: int = 1,
        seed: int = 0,
        eval_mode: bool = False,
    ):
        self.cfg = cfg
        self.test_id = int(test_id)
        self.seed = int(seed)
        self.eval_mode = bool(eval_mode)
        self.__post_init__()

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.n = int(self.cfg["num_drones"])
        self.k = int(self.cfg["neighbor_k"])
        self.hz = float(self.cfg["control_hz"])
        self.dt = 1.0 / self.hz
        self.control_dt = self.dt
        self.episode_s = float(self.cfg["episode_s"])
        self.episode_steps = int(self.episode_s * self.hz)
        self.arena_cm = np.asarray(self.cfg["arena_cm"], dtype=float)
        self.collision_cm = float(self.cfg["collision_cm"])
        self.body_z = body_z_m(self.collision_cm)
        self.task = make_task(self.test_id, self.cfg)
        self.driver = AttitudeDriver.from_cfg(self.cfg)
        self.drones = [
            Quadrotor(
                mass_kg=float(self.cfg.get("mass_kg", 0.5)),
                arm_cm=float(self.cfg.get("arm_cm", 12.5)),
                max_thrust=float(self.cfg.get("max_thrust", 400.0)),
                motor_lag=bool(self.cfg.get("motor_lag", False)),
                yaw_coeff=float(self.cfg.get("yaw_coeff", 6.0)),
                inertia_diag=self.cfg.get("inertia_diag"),
            )
            for _ in range(self.n)
        ]
        self.step_count = 0
        self.sim_time = 0.0
        self.collided = False
        self.out_of_bounds = False
        self.floor_hit = False
        self.history: list[dict] = []
        self.prev_dist_cm: np.ndarray | None = None
        self.reset()

    def obs_size(self) -> int:
        return 18 + 6 * self.k

    def positions(self) -> np.ndarray:
        return np.stack([d.pos for d in self.drones])

    def velocities(self) -> np.ndarray:
        return np.stack([d.vel for d in self.drones])

    def rotations(self) -> np.ndarray:
        return np.stack([d.rot for d in self.drones])

    def omegas(self) -> np.ndarray:
        return np.stack([d.omega for d in self.drones])

    def reset(self) -> np.ndarray:
        self.task_state = self.task.reset(self.rng)
        self.step_count = 0
        self.sim_time = 0.0
        self.collided = False
        self.out_of_bounds = False
        self.floor_hit = False
        self.history = []
        self.prev_dist_cm = None

        if self.test_id in (1, 3):
            tgt = np.asarray(self.task_state.shared_target, dtype=float)
            for i, drone in enumerate(self.drones):
                ang = 2.0 * np.pi * i / self.n + float(self.rng.uniform(-0.1, 0.1))
                pos = np.array(
                    [
                        tgt[0] + 50.0 * np.cos(ang),
                        tgt[1] + 50.0 * np.sin(ang),
                        tgt[2] + float(self.rng.uniform(-5.0, 5.0)),
                    ]
                )
                vel = np.asarray(self.rng.normal(0.0, 5.0, size=3), dtype=float)
                drone.reset(pos, 0.0, vel)
        else:
            slots = np.asarray(self.task_state.slot_positions, dtype=float)
            for i, drone in enumerate(self.drones):
                offset = self.rng.uniform(-22.0, 22.0, size=3)
                pos = slots[i] + offset
                vel = np.asarray(self.rng.normal(0.0, 5.0, size=3), dtype=float)
                drone.reset(pos, 0.0, vel)

        self._clip_arena()
        return self._obs()

    def set_joystick_target(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=float)
        setter = getattr(self.task, "set_joystick_target", None)
        if setter is not None:
            setter(target)
            self.task_state = self.task.step(self.sim_time)

    def set_geometry_kind(self, kind: str) -> None:
        setter = getattr(self.task, "set_kind", None)
        if setter is not None:
            setter(str(kind))
            self.task_state = self.task.step(self.sim_time)

    def set_test_id(self, test_id: int) -> np.ndarray:
        if test_id not in (1, 2, 3):
            raise ValueError(f"test_id must be 1, 2, or 3; got {test_id}")
        self.test_id = int(test_id)
        self.task = make_task(self.test_id, self.cfg)
        return self.reset()

    def step(self, cmds: np.ndarray) -> StepResult:
        cmds = np.asarray(cmds, dtype=float).reshape(self.n, 4)
        for i, drone in enumerate(self.drones):
            action = self.driver.motors(drone.rot, drone.omega, drone.vel, cmds[i])
            drone.step(action, self.dt)
        self.sim_time += self.dt
        self.step_count += 1

        self.task_state = self.task.step(self.sim_time)
        self._clip_arena()
        pos = self.positions()

        pair_hit, floor_hit = self._collisions(pos)
        if pair_hit:
            self.collided = True
            overlap = body_overlap(pos, self.collision_cm, self.body_z)
            self._separate(pos, overlap)
        if floor_hit:
            self.floor_hit = True
            self._bounce_floor()

        self.out_of_bounds = self._bounds(pos)

        rewards = self._rewards(pos, cmds, self.collided, self.out_of_bounds)

        self.history.append(
            {
                "pos": pos.copy(),
                "vel": self.velocities().copy(),
                "rot": self.rotations().copy(),
                "aim": self.task_state.aim_points.copy(),
                "shared_target": np.asarray(
                    self.task_state.shared_target
                    if self.task_state.shared_target is not None
                    else self.task_state.aim_points[0],
                    dtype=float,
                ).copy(),
            }
        )

        if self.eval_mode:
            done = self.step_count >= self.episode_steps
        else:
            done = (self.step_count >= self.episode_steps) or self.collided or self.out_of_bounds

        info = {
            "collided": self.collided,
            "out_of_bounds": self.out_of_bounds,
            "floor_hit": self.floor_hit,
        }
        return StepResult(obs=self._obs(), rewards=rewards, done=done, info=info)

    def snapshot(self) -> dict:
        tgt = self.task_state.shared_target
        if tgt is None:
            tgt = self.task_state.aim_points[0] if len(self.task_state.aim_points) > 0 else np.zeros(3)
        return {
            "pos": self.positions(),
            "vel": self.velocities(),
            "rot": self.rotations(),
            "shared_target": np.asarray(tgt, dtype=float),
            "slots": getattr(self.task_state, "slot_positions", None),
            "test_id": self.test_id,
            "geometry_kind": getattr(self.task, "kind", None) or "",
        }

    def _collisions(self, pos: np.ndarray) -> tuple[bool, bool]:
        overlap = body_overlap(pos, self.collision_cm, self.body_z)
        pair = bool(np.any(overlap))
        floor = bool(np.any(pos[:, 2] <= 5.0))
        return pair, floor

    def _separate(self, pos: np.ndarray, overlap: np.ndarray) -> None:
        pairs = np.argwhere(overlap)
        for i, j in pairs:
            if i >= j:
                continue
            delta = self.drones[i].pos - self.drones[j].pos
            dist = float(np.linalg.norm(delta))
            if dist < 1e-6:
                n = np.array([1.0, 0.0, 0.0])
            else:
                n = delta / dist
            rel_vel = self.drones[i].vel - self.drones[j].vel
            closing = float(np.dot(rel_vel, n))
            if closing < 0:
                impulse = -closing * 0.5
                self.drones[i].vel += impulse * n
                self.drones[j].vel -= impulse * n

    def _bounce_floor(self) -> None:
        for drone in self.drones:
            if drone.pos[2] < 5.0:
                drone.pos[2] = 5.0
                if drone.vel[2] < 0:
                    drone.vel[2] = -0.5 * drone.vel[2]

    def _clip_arena(self) -> None:
        half_x = float(self.arena_cm[0]) * 0.5
        half_y = float(self.arena_cm[1]) * 0.5
        max_z = float(self.arena_cm[2])
        for drone in self.drones:
            drone.pos[0] = np.clip(drone.pos[0], -half_x, half_x)
            drone.pos[1] = np.clip(drone.pos[1], -half_y, half_y)
            drone.pos[2] = np.clip(drone.pos[2], 0.0, max_z)

    def _obs(self) -> np.ndarray:
        obs = build_obs(
            self.positions(),
            self.velocities(),
            self.rotations(),
            self.omegas(),
            self.task_state.aim_points,
            self.k,
        )
        if self.cfg.get("sensor_noise", False):
            obs = (obs + self.rng.normal(0.0, 0.01, size=obs.shape)).astype(np.float32)
        return obs.astype(np.float32)

    def _bounds(self, pos: np.ndarray) -> bool:
        half_x = float(self.arena_cm[0]) * 0.5
        half_y = float(self.arena_cm[1]) * 0.5
        max_z = float(self.arena_cm[2])
        out_x = np.any(np.abs(pos[:, 0]) >= half_x)
        out_y = np.any(np.abs(pos[:, 1]) >= half_y)
        out_z = np.any((pos[:, 2] < 0.0) | (pos[:, 2] > max_z))
        return bool(out_x or out_y or out_z)

    def _rewards(
        self,
        pos: np.ndarray,
        act: np.ndarray,
        collided: bool,
        out_of_bounds: bool,
    ) -> np.ndarray:
        train_cfg = self.cfg.get("train", {}) if isinstance(self.cfg.get("train"), dict) else self.cfg
        w_dist = float(train_cfg.get("w_distance", 1.0))
        w_coll = float(train_cfg.get("w_collision", 8.0))
        w_prox = float(train_cfg.get("w_proximity", 2.0))
        w_om = float(train_cfg.get("w_omega", 0.02))
        w_th = float(train_cfg.get("w_thrust", 0.01))
        w_ti = float(train_cfg.get("w_tilt", 0.05))

        if self.test_id in (1, 3):
            tgt = np.asarray(
                self.task_state.shared_target
                if self.task_state.shared_target is not None
                else self.task_state.aim_points[0],
                dtype=float,
            )
            standoff = float(self.cfg["test1_dist_cm"]) * 0.75
            rel = pos - tgt
            xy_dist = np.linalg.norm(rel[:, :2], axis=1)
            dist = np.sqrt((xy_dist - standoff) ** 2 + rel[:, 2] ** 2)
        else:
            dist = np.linalg.norm(pos - self.task_state.aim_points, axis=1)

        c_dist = distance_cost(dist)

        c_recede = np.zeros(self.n)
        for i in range(self.n):
            aim_i = self.task_state.aim_points[i]
            delta = aim_i - pos[i]
            norm_d = float(np.linalg.norm(delta))
            if norm_d > 1e-6:
                unit = delta / norm_d
                v_towards = float(np.dot(self.drones[i].vel, unit))
                if v_towards < 0:
                    c_recede[i] = -v_towards * 0.01

        fade = np.clip(1.0 - (dist - 10.0) / 10.0, 0.0, 1.0)
        omegas = self.omegas()
        rot = self.rotations()
        c_om = w_om * np.linalg.norm(omegas, axis=1) * fade
        c_act = w_th * np.mean(act ** 2, axis=1) * fade
        tilt_rad = np.arccos(np.clip(rot[:, 2, 2], -1.0, 1.0))
        c_tilt = w_ti * tilt_rad * fade

        overlap = body_overlap(pos, self.collision_cm, self.body_z)
        c_coll = np.zeros(self.n)
        for i in range(self.n):
            if np.any(overlap[i]):
                c_coll[i] = w_coll

        c_prox = np.zeros(self.n)
        pair_dist = self._pair_distances(pos)
        min_p = np.min(pair_dist, axis=1)
        safe_r = 2.0 * self.collision_cm
        close = min_p < safe_r
        c_prox[close] = w_prox * (1.0 - min_p[close] / safe_r)

        r = -(w_dist * c_dist + c_recede + c_coll + c_prox + c_om + c_act + c_tilt)
        if collided:
            r -= w_coll
        if out_of_bounds:
            r -= 10.0

        self.prev_dist_cm = dist.copy()
        return r.astype(np.float32)

    def _pair_distances(self, pos: np.ndarray) -> np.ndarray:
        p = np.asarray(pos, dtype=float)
        d = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(dist, np.inf)
        return dist
