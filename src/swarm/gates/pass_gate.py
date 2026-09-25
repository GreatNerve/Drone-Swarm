from __future__ import annotations

import numpy as np

from swarm.env.swarm_env import body_overlap, body_z_m


def standoff_cm(cfg: dict) -> float:
    return float(cfg["test1_dist_cm"]) * 0.75


def ring_error(pos: np.ndarray, target: np.ndarray, radius: float) -> np.ndarray:
    """Centimetres off the standoff ring around a shared target (not the point)."""
    rel = np.asarray(pos, dtype=float) - np.asarray(target, dtype=float)
    xy = np.linalg.norm(rel[..., :2], axis=-1)
    z = rel[..., 2]
    return np.sqrt((xy - float(radius)) ** 2 + z * z)


def _tilt_deg(rot: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(rot[:, 2, 2], -1.0, 1.0)))


def _space_ok(pos: np.ndarray, cfg: dict) -> bool:
    xy = float(cfg["collision_cm"])
    return not bool(np.any(body_overlap(pos, xy, body_z_m(xy))))


def evaluate_episode(history: list[dict], cfg: dict, test_id: int, collided: bool, out_of_bounds: bool) -> dict:
    hz = float(cfg["control_hz"])
    hold_n = int(float(cfg["hold_s"]) * hz)
    failed = collided or out_of_bounds
    if not history:
        return {"passed": False, "reason": "empty", "error": 1000.0, "test_id": test_id}
    window = history[-hold_n:] if len(history) >= hold_n else history
    pos = np.stack([h["pos"] for h in window])
    vel = np.stack([h["vel"] for h in window])
    rot = np.stack([h["rot"] for h in window])
    aim = np.stack([h["aim"] for h in window])
    if test_id in (1, 3):
        tgt = np.stack(
            [np.asarray(h.get("shared_target", h["aim"][0]), dtype=float) for h in window]
        )
        dist = ring_error(pos, tgt[:, None, :], standoff_cm(cfg))
    else:
        dist = np.linalg.norm(pos - aim, axis=2)
    speed = np.linalg.norm(vel, axis=2)
    tilt = np.stack([_tilt_deg(r) for r in rot])
    space_ok = all(_space_ok(h["pos"], cfg) for h in window)
    mean_dist = float(dist.mean())
    short = len(history) < hold_n
    if failed:
        return {"passed": False, "reason": "collision_or_bounds", "error": mean_dist, "test_id": test_id}
    speed_ok = bool(np.all(speed < float(cfg["speed_max_cmps"])) or test_id == 3)
    tilt_ok = bool(np.all(tilt < float(cfg["tilt_max_deg"])))
    if test_id == 1:
        dist_ok = bool(np.all(dist < float(cfg["test1_dist_cm"])))
        passed = dist_ok and speed_ok and tilt_ok and space_ok
        cap = cfg["test1_dist_cm"]
    elif test_id == 2:
        dist_ok = bool(np.all(dist < float(cfg["test2_slot_cm"])))
        passed = dist_ok and speed_ok and tilt_ok and space_ok
        cap = cfg["test2_slot_cm"]
    else:
        dist_ok = bool(np.all(dist < float(cfg["test3_lag_cm"])))
        passed = dist_ok and tilt_ok and space_ok
        cap = cfg["test3_lag_cm"]
        speed_ok = True
    if short:
        passed = False
    return {
        "passed": passed,
        "reason": "short_episode" if short else ("ok" if passed else "hold_failed"),
        "error": mean_dist,
        "dist_ok": dist_ok,
        "speed_ok": speed_ok,
        "tilt_ok": tilt_ok,
        "space_ok": space_ok,
        "cap": cap,
        "test_id": test_id,
    }


def combined_score(results: list[dict]) -> float:
    parts = []
    for r in results:
        err = 1000.0 if r.get("error") is None else float(r["error"])
        parts.append((1.0 if r.get("passed") else 0.0) - 0.001 * err)
    return float(sum(parts) / max(len(parts), 1))
