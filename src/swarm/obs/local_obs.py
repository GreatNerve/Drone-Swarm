from __future__ import annotations

import numpy as np


def self_obs(pos: np.ndarray, vel: np.ndarray, rot: np.ndarray, omega: np.ndarray, aim: np.ndarray) -> np.ndarray:
    return np.concatenate([pos - aim, vel, rot.reshape(-1), omega]).astype(np.float32)


def neighbor_set(
    positions: np.ndarray,
    velocities: np.ndarray,
    drone_i: int,
    k: int,
) -> np.ndarray:
    n = positions.shape[0]
    out = np.zeros((k, 6), dtype=np.float32)
    if n <= 1 or k <= 0:
        return out.reshape(-1)
    others = [j for j in range(n) if j != drone_i]
    rel_pos = positions[others] - positions[drone_i]
    rel_vel = velocities[others] - velocities[drone_i]
    dist = np.linalg.norm(rel_pos, axis=1)
    dist = np.maximum(dist, 1e-3)
    unit = rel_pos / dist[:, None]
    score = dist + np.sum(unit * rel_vel, axis=1)
    order = np.argsort(score)[:k]
    packed = np.concatenate([rel_pos[order], rel_vel[order]], axis=1)
    out[: packed.shape[0]] = packed
    return out.reshape(-1)


def build_obs(
    positions: np.ndarray,
    velocities: np.ndarray,
    rotations: np.ndarray,
    omegas: np.ndarray,
    aim_points: np.ndarray,
    neighbor_k: int,
) -> np.ndarray:
    n = positions.shape[0]
    rows = [
        np.concatenate(
            [
                self_obs(positions[i], velocities[i], rotations[i], omegas[i], aim_points[i]),
                neighbor_set(positions, velocities, i, neighbor_k),
            ]
        )
        for i in range(n)
    ]
    return np.stack(rows, axis=0)
