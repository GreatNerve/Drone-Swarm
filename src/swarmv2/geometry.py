"""Collision geometry shared by the CLI, the tests and (from Phase 1) the simulator.

Scene obstacles carry presentation hints for the GUI (a building style, a tree model), but anything
that computes uses only these primitives: yawed boxes, upright cylinders and spheres. The viewer scales
each 3D model to exactly the primitive's bounds, so what the GUI shows is what training sees.
"""
import numpy as np

SHAPES = ("box", "cylinder", "sphere")


def primitives(obstacles):
    """Collision primitives for scene obstacles (buildings, pillars, trees, rocks, gates, spheres)."""
    result = []
    for o in obstacles:
        kind = o["kind"]
        if kind == "box":
            result.append({"shape": "box", "center": [o["x"], o["y"], o["base"] + o["sz"] / 2],
                           "size": [o["sx"], o["sy"], o["sz"]], "yaw": o.get("yaw", 0.0), "source": kind})
        elif kind in ("pillar", "tree"):
            result.append({"shape": "cylinder", "center": [o["x"], o["y"], o["base"] + o["height"] / 2],
                           "radius": o["radius"], "height": o["height"], "source": kind})
        elif kind == "rock":
            result.append({"shape": "sphere", "center": [o["x"], o["y"], o["base"] + 0.5 * o["radius"]],
                           "radius": o["radius"], "source": kind})
        elif kind == "sphere":
            result.append({"shape": "sphere", "center": list(o["p"]), "radius": o["radius"], "source": kind})
        elif kind == "gate":
            c, s = np.cos(o.get("yaw", 0.0)), np.sin(o.get("yaw", 0.0))
            half, t, h = o["width"] / 2, o["thickness"], o["height"]
            for offset in (-half, half):
                result.append({"shape": "box", "center": [o["x"] - s * offset, o["y"] + c * offset, o["base"] + h / 2],
                               "size": [t, t, h], "yaw": o.get("yaw", 0.0), "source": kind})
            result.append({"shape": "box", "center": [o["x"], o["y"], o["base"] + h + t / 2],
                           "size": [t, o["width"] + t, t], "yaw": o.get("yaw", 0.0), "source": kind})
        else:
            raise ValueError(f"unknown obstacle kind {kind!r}")
    return result


def signed_distance(prims, points):
    """Signed distance (m) from each point (N, 3) to the nearest primitive surface; negative inside."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    best = np.full(len(points), np.inf)
    for p in prims:
        center = np.asarray(p["center"], dtype=float)
        rel = points - center
        if p["shape"] == "box":
            c, s = np.cos(p.get("yaw", 0.0)), np.sin(p.get("yaw", 0.0))
            local = np.stack((c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]), axis=1)
            q = np.abs(local) - np.asarray(p["size"], dtype=float) / 2
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            distance = outside + np.minimum(q.max(axis=1), 0.0)
        elif p["shape"] == "cylinder":
            radial = np.linalg.norm(rel[:, :2], axis=1) - p["radius"]
            axial = np.abs(rel[:, 2]) - p["height"] / 2
            q = np.stack((radial, axial), axis=1)
            distance = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
        elif p["shape"] == "sphere":
            distance = np.linalg.norm(rel, axis=1) - p["radius"]
        else:
            raise ValueError(f"unknown shape {p['shape']!r}")
        best = np.minimum(best, distance)
    return best


def top_height(prims, x, y):
    """Highest primitive top directly above (x, y), or -inf where there is none (used to lift wind arrows)."""
    top = -np.inf
    for p in prims:
        cx, cy, cz = p["center"]
        if p["shape"] == "box":
            c, s = np.cos(p.get("yaw", 0.0)), np.sin(p.get("yaw", 0.0))
            lx, ly = c * (x - cx) + s * (y - cy), -s * (x - cx) + c * (y - cy)
            if abs(lx) <= p["size"][0] / 2 and abs(ly) <= p["size"][1] / 2:
                top = max(top, cz + p["size"][2] / 2)
        elif p["shape"] == "cylinder":
            if np.hypot(x - cx, y - cy) <= p["radius"]:
                top = max(top, cz + p["height"] / 2)
        elif np.hypot(x - cx, y - cy) <= p["radius"]:
            top = max(top, cz + p["radius"])
    return top
