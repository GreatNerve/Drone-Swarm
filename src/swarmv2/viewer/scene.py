"""Frame and terrain schema shared by every scene source (the Phase 0 demo now, the simulator from Phase 1).

World frame: x east, y north, z up (metres). Quaternions are unit, ordered [w, x, y, z] (Hamilton),
rotating body axes (x forward, y left, z up) into the world frame.
"""
import math

import numpy as np

SCHEMA_VERSION = 1
OBSTACLE_KINDS = ("pillar", "box", "sphere", "tree", "gate")
# Required fields per obstacle kind. Every kind reduces to basic collision primitives
# (collision_primitives); styles such as a building's facade are presentation-only hints for the GUI.
OBSTACLE_FIELDS = {"pillar": ("x", "y", "base", "height", "radius"),
                   "box": ("x", "y", "base", "sx", "sy", "sz", "yaw"),
                   "sphere": ("p", "radius"),
                   "tree": ("x", "y", "base", "height", "radius"),
                   "gate": ("x", "y", "base", "width", "height", "post", "yaw")}


def quat_from_euler(roll, pitch, yaw):
    """ZYX (yaw-pitch-roll) Euler angles in radians to [w, x, y, z]; broadcasts over arrays."""
    cr, sr = np.cos(np.asarray(roll) / 2), np.sin(np.asarray(roll) / 2)
    cp, sp = np.cos(np.asarray(pitch) / 2), np.sin(np.asarray(pitch) / 2)
    cy, sy = np.cos(np.asarray(yaw) / 2), np.sin(np.asarray(yaw) / 2)
    return np.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), axis=-1)


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_rotate(q, v):
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4)."""
    q, v = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _round(values, digits=4):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def make_frame(time_s, positions, quaternions, slots, target, target_velocity, wind_samples, obstacles=(),
               decision=None, metrics=None):
    """One viewer frame. wind_samples: list of (position, velocity) pairs."""
    return {"schema": SCHEMA_VERSION, "time": round(float(time_s), 3),
            "drones": [{"p": _round(p), "q": _round(q, 6)} for p, q in zip(positions, quaternions)],
            "slots": _round(slots), "target": {"p": _round(target), "v": _round(target_velocity)},
            "wind": [{"p": _round(p, 2), "v": _round(v, 3)} for p, v in wind_samples],
            "obstacles": list(obstacles), "decision": decision, "metrics": metrics or {}}


def make_terrain(heights, size_m, obstacles=(), regions=()):
    """Static terrain: a square heightfield heights[iy][ix] centred on the origin, plus fixed obstacles."""
    heights = np.asarray(heights, dtype=float)
    return {"schema": SCHEMA_VERSION, "size": float(size_m), "cells": int(heights.shape[0]),
            "heights": np.round(heights, 3).tolist(), "obstacles": list(obstacles), "regions": list(regions)}


def validate_frame(frame):
    """Raise ValueError if a frame breaks the schema (used by tests and by the server in debug)."""
    if frame.get("schema") != SCHEMA_VERSION:
        raise ValueError("unknown frame schema")
    for drone in frame["drones"]:
        p, q = drone["p"], drone["q"]
        if len(p) != 3 or len(q) != 4 or not all(math.isfinite(x) for x in p + q):
            raise ValueError("drone pose must be finite p[3] and q[4]")
        if abs(math.sqrt(sum(x * x for x in q)) - 1.0) > 1e-4:
            raise ValueError("drone quaternion is not unit length")
    if len(frame["slots"]) != len(frame["drones"]):
        raise ValueError("one slot per drone")
    for sample in frame["wind"]:
        if len(sample["p"]) != 3 or len(sample["v"]) != 3:
            raise ValueError("wind sample must be p[3], v[3]")
    for obstacle in frame["obstacles"]:
        if obstacle.get("kind") not in OBSTACLE_KINDS:
            raise ValueError(f"obstacle kind must be one of {OBSTACLE_KINDS}")
    return frame
