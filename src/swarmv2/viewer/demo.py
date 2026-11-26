"""Phase 0 demo scene: a scripted swarm over procedural terrain. There is NO physics here.

Everything is a deterministic function of (seed, time): the path, the formation, the OU wind and the
mock Layer 3 decisions. Drone tilt is the direction thrust would need to balance the path acceleration,
gravity and a wind drag proxy, so drones lean into the wind as real ones would. Phase 1 replaces this
source with the simulator; the frame schema stays the same.
"""
import math

import numpy as np

from ..config import FLIGHT_MODES, FORMATIONS, DemoConfig
from .scene import make_frame, make_terrain, quat_from_euler

GRAVITY = 9.81
DRAG_PER_MASS = 0.035          # 1/m: (m/s^2) of drag per (m/s)^2 of relative air speed (illustrative)
MAX_ACCEL = 6.0                # m/s^2: the path's terrain-following kinks would otherwise demand more than g
MAX_TILT = math.radians(35.0)  # attitude limit, as on a real flight controller
CANYON_X = (-6.0, 20.0)
PILLAR_X = (25.0, 38.0)
PATH_AMPLITUDE_M = 32.0
PATH_RATE = 0.08               # rad/s of the back-and-forth path phase
SAFE_DECISION = {"thought": "Holding safe defaults.", "flight_mode": "STEADY", "agility_alpha": 0.2,
                 "formation_preset": "DIAMOND", "formation_scale_m": 1.6, "cruise_speed_cap_mps": 1.5,
                 "target_agl_m": 3.0, "waypoint_offset_world": [0.0, 0.0, 0.0], "replan_trigger_sec": 1.5}


def _smoothstep(edge0, edge1, x):
    t = np.clip((np.asarray(x, dtype=float) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def value_noise(rng, cells, octaves=4, base=3):
    """Multi-octave smooth value noise on a cells x cells grid, in [0, 1]."""
    total = np.zeros((cells, cells))
    amplitude, weight = 1.0, 0.0
    for octave in range(octaves):
        n = base * 2 ** octave + 2
        grid = rng.random((n, n))
        coords = np.linspace(0, n - 1.000001, cells)
        index = coords.astype(int)
        frac = coords - index
        frac = frac * frac * (3 - 2 * frac)
        rows = grid[index] * (1 - frac)[:, None] + grid[index + 1] * frac[:, None]
        total += amplitude * (rows[:, index] * (1 - frac) + rows[:, index + 1] * frac)
        weight += amplitude
        amplitude *= 0.5
    return total / weight


def formation_offsets(preset, count, scale):
    """Body-frame slot offsets (x forward, y left) for a formation preset."""
    i = np.arange(count, dtype=float)
    if preset == "COLUMN":
        offsets = np.stack((-i * scale, np.zeros(count)), axis=1)
    elif preset == "LINE":
        offsets = np.stack((np.zeros(count), (i - (count - 1) / 2) * scale), axis=1)
    elif preset == "V_SHAPE":
        rank = np.ceil(i / 2)
        side = np.where(i % 2 == 1, 1.0, -1.0)
        offsets = np.stack((-rank * scale, side * rank * scale), axis=1)
    else:  # DIAMOND: drones spaced evenly on a diamond (rotated square) perimeter
        s = 4 * i / count
        edge, f = np.floor(s), s - np.floor(s)
        corners = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]], dtype=float)
        offsets = corners[edge.astype(int)] * (1 - f)[:, None] + corners[edge.astype(int) + 1] * f[:, None]
        offsets *= scale * max(1.0, count / 4)
    offsets -= offsets.mean(axis=0)
    return np.concatenate((offsets, np.zeros((count, 1))), axis=1)


def verify_decision(raw, radius_m):
    """Deterministic verifier of the Layer 3 contract: enums checked, numbers clamped to safe bounds.

    Returns (decision, clamped field names, fault message or None). Invalid output falls back to
    safe STEADY defaults.
    """
    if not isinstance(raw, dict):
        return dict(SAFE_DECISION), [], "invalid output (not a JSON object); holding STEADY defaults"
    if raw.get("flight_mode") not in FLIGHT_MODES or raw.get("formation_preset") not in FORMATIONS:
        return dict(SAFE_DECISION), [], "unknown flight_mode or formation_preset; holding STEADY defaults"
    bounds = {"agility_alpha": (0.0, 1.0), "formation_scale_m": (max(2.5 * radius_m, 0.5), 4.0),
              "cruise_speed_cap_mps": (0.3, 6.0), "target_agl_m": (1.5, 8.0), "replan_trigger_sec": (0.5, 5.0)}
    decision, clamped = {"thought": str(raw.get("thought", ""))[:240],
                         "flight_mode": raw["flight_mode"], "formation_preset": raw["formation_preset"]}, []
    for key, (low, high) in bounds.items():
        try:
            value = float(raw.get(key, SAFE_DECISION[key]))
        except (TypeError, ValueError):
            value = SAFE_DECISION[key]
        if not math.isfinite(value):
            value = SAFE_DECISION[key]
        bounded = min(max(value, low), high)
        if bounded != value:
            clamped.append(key)
        decision[key] = bounded
    offset = raw.get("waypoint_offset_world", SAFE_DECISION["waypoint_offset_world"])
    try:
        offset = [float(v) for v in offset][:3]
    except (TypeError, ValueError):
        offset = list(SAFE_DECISION["waypoint_offset_world"])
    bounded = [min(max(v, -50.0), 50.0) for v in offset] + [0.0] * (3 - len(offset))
    if bounded != offset:
        clamped.append("waypoint_offset_world")
    decision["waypoint_offset_world"] = bounded
    return decision, clamped, None


class DemoScene:
    def __init__(self, config=None, seed=1000):
        self.config = config or DemoConfig()
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        t = self.config.terrain
        self.size, self.cells = t.size_m, t.cells
        axis = np.linspace(-t.size_m / 2, t.size_m / 2, t.cells)
        self.axis = axis
        xx, yy = np.meshgrid(axis, axis)
        heights = t.max_height_m * value_noise(rng, t.cells) ** 1.6
        in_canyon = _smoothstep(CANYON_X[0] - 4, CANYON_X[0], xx) * (1 - _smoothstep(CANYON_X[1], CANYON_X[1] + 4, xx))
        half = t.canyon_width_m / 2
        walls = t.canyon_depth_m * _smoothstep(half, half + 2.5, np.abs(yy)) * (1 - _smoothstep(half + 7, half + 12, np.abs(yy)))
        floor = 1 - 0.75 * (1 - _smoothstep(half - 1, half + 1, np.abs(yy)))
        self.heights = heights * (1 - in_canyon + in_canyon * floor) + in_canyon * walls
        self.pillars = self._place_pillars(rng, t)
        self.mean_wind = np.asarray(self.config.wind.mean_mps, dtype=float)
        self.gust_dt, self.gusts = self._ou_gusts(rng, 600.0)
        self.wind_points = np.array([(x, y) for x in np.linspace(-32, 32, 7) for y in np.linspace(-32, 32, 7)])
        self.decisions = []

    # -- static world -----------------------------------------------------------------------
    def _place_pillars(self, rng, t):
        pillars = []
        while len(pillars) < t.pillars:
            x, y = rng.uniform(*PILLAR_X), rng.uniform(-14, 14)
            if abs(y) < 5.0 or any(math.hypot(x - p["x"], y - p["y"]) < 2.5 for p in pillars):
                continue
            radius = float(rng.uniform(*t.pillar_radius_m))
            base = float(self.ground(x, y))
            pillars.append({"kind": "pillar", "x": round(x, 3), "y": round(y, 3), "base": round(base, 3),
                            "height": round(float(rng.uniform(6, 11)), 3), "radius": round(radius, 3)})
        return pillars

    def ground(self, x, y):
        """Bilinear terrain height at world (x, y); broadcasts."""
        step = self.axis[1] - self.axis[0]
        fx = np.clip((np.asarray(x, dtype=float) + self.size / 2) / step, 0, self.cells - 1.000001)
        fy = np.clip((np.asarray(y, dtype=float) + self.size / 2) / step, 0, self.cells - 1.000001)
        ix, iy = fx.astype(int), fy.astype(int)
        ax, ay = fx - ix, fy - iy
        h = self.heights
        return ((h[iy, ix] * (1 - ax) + h[iy, ix + 1] * ax) * (1 - ay)
                + (h[iy + 1, ix] * (1 - ax) + h[iy + 1, ix + 1] * ax) * ay)

    def terrain(self):
        regions = [{"name": "canyon", "x": list(CANYON_X)}, {"name": "pillar forest", "x": list(PILLAR_X)}]
        return make_terrain(self.heights, self.size, self.pillars, regions)

    # -- wind -------------------------------------------------------------------------------
    def _ou_gusts(self, rng, duration):
        w = self.config.wind
        dt = 0.05
        steps = int(duration / dt)
        decay = math.exp(-dt / w.gust_tau_s)
        noise = w.gust_sigma_mps * math.sqrt(1 - decay ** 2)
        gusts = np.zeros((steps, 3))
        for k in range(1, steps):
            gusts[k] = decay * gusts[k - 1] + noise * rng.standard_normal(3) * (1, 1, 0.3)
        return dt, gusts

    def wind(self, positions, time_s):
        """Mean wind + OU gust + terrain speed-up and slope lift at world positions (..., 3)."""
        positions = np.asarray(positions, dtype=float)
        gust = self.gusts[int(time_s / self.gust_dt) % len(self.gusts)]
        base = self.mean_wind + gust
        x, y = positions[..., 0], positions[..., 1]
        height = self.ground(x, y)
        e = 0.5
        slope = np.stack(((self.ground(x + e, y) - self.ground(x - e, y)) / (2 * e),
                          (self.ground(x, y + e) - self.ground(x, y - e)) / (2 * e)), axis=-1)
        speedup = (1 + 0.05 * height)[..., None]
        horizontal = base[:2] * speedup
        lift = 0.5 * np.sum(slope * horizontal, axis=-1)
        return np.concatenate((horizontal, (base[2] + lift)[..., None]), axis=-1)

    # -- path and decisions -----------------------------------------------------------------
    def _path_xy(self, time_s):
        phase = PATH_RATE * np.asarray(time_s, dtype=float)
        return np.stack((PATH_AMPLITUDE_M * np.sin(phase), 1.5 * np.sin(2.3 * phase)), axis=-1)

    def _region(self, x):
        if CANYON_X[0] <= x <= CANYON_X[1]:
            return "canyon"
        if PILLAR_X[0] - 3 <= x <= PILLAR_X[1] + 3:
            return "pillars"
        return "open"

    def _decision(self, index):
        while len(self.decisions) <= index:
            k = len(self.decisions)
            t = k * self.config.reasoning_period_s
            x, y = self._path_xy(t + 3.0)
            region = self._region(float(x))
            wind = self.wind(np.array([x, y, self.ground(x, y) + 3.0]), t)
            speed = float(np.linalg.norm(wind[:2]))
            if region == "canyon":
                raw = {"thought": f"Gorge about {self.config.terrain.canyon_width_m:.0f} m wide ahead and "
                                  f"{speed:.1f} m/s wind: single file, STEADY to keep tilt low.",
                       "flight_mode": "STEADY", "agility_alpha": 0.15, "formation_preset": "COLUMN",
                       "formation_scale_m": 1.3, "cruise_speed_cap_mps": 1.5, "target_agl_m": 2.5}
            elif region == "pillars":
                raw = {"thought": "Pillar forest ahead: narrow V to keep lanes between columns, moderate agility.",
                       "flight_mode": "STEADY", "agility_alpha": 0.35, "formation_preset": "V_SHAPE",
                       "formation_scale_m": 1.4, "cruise_speed_cap_mps": 2.0, "target_agl_m": 3.0}
            else:
                agile = speed < 4.5
                raw = {"thought": f"Open terrain, wind {speed:.1f} m/s: " + (
                           "wide DIAMOND and AGILE cruise." if agile else "wide DIAMOND but STEADY in strong wind."),
                       "flight_mode": "AGILE" if agile else "STEADY", "agility_alpha": 0.7 if agile else 0.3,
                       "formation_preset": "DIAMOND", "formation_scale_m": 1.6,
                       "cruise_speed_cap_mps": 3.0, "target_agl_m": 3.2}
            raw.update({"waypoint_offset_world": [12.0, 0.0, 0.0], "replan_trigger_sec": 1.5})
            if k % 11 == 10:
                raw = "<<mock backend timeout>>"
            elif k % 7 == 6:
                raw = {**raw, "cruise_speed_cap_mps": 12.0, "target_agl_m": 0.4, "formation_scale_m": 0.2,
                       "thought": raw["thought"] + " (mock out-of-range values)"}
            decision, clamped, fault = verify_decision(raw, self.config.vehicle.radius_m)
            self.decisions.append({"index": k, "time": round(t, 3), "region": region, "decision": decision,
                                   "clamped": clamped, "fault": fault})
        return self.decisions[index]

    def trace(self, time_s, limit=None):
        """Decisions made up to time_s, newest last."""
        count = int(max(0.0, time_s) // self.config.reasoning_period_s) + 1
        self._decision(count - 1)
        rows = self.decisions[:count]
        return rows[-limit:] if limit else rows

    def _blend(self, time_s, key):
        period = self.config.reasoning_period_s
        k = int(max(0.0, time_s) // period)
        current = self._decision(k)["decision"][key]
        previous = self._decision(max(0, k - 1))["decision"][key]
        s = float(_smoothstep(0, period, time_s - k * period))
        return previous, current, s

    def _formation(self, time_s):
        n = self.config.drones
        old, new, s = self._blend(time_s, "formation_preset")
        scale_old, scale_new, _ = self._blend(time_s, "formation_scale_m")
        return (formation_offsets(old, n, scale_old) * (1 - s) + formation_offsets(new, n, scale_new) * s)

    def _target(self, time_s):
        xy = self._path_xy(time_s)
        x, y = float(xy[0]), float(xy[1])
        agl_old, agl_new, s = self._blend(time_s, "target_agl_m")
        ahead = [self.ground(x + dx, y) for dx in (-2.0, 0.0, 2.0, 4.0)]
        return np.array([x, y, max(ahead) + agl_old * (1 - s) + agl_new * s])

    def _poses(self, time_s):
        dt = 0.25
        p_prev, p_now, p_next = (self._target(time_s + d) for d in (-dt, 0.0, dt))
        velocity = (p_next - p_prev) / (2 * dt)
        acceleration = (p_next - 2 * p_now + p_prev) / dt ** 2
        norm = np.linalg.norm(acceleration)
        if norm > MAX_ACCEL:
            acceleration *= MAX_ACCEL / norm
        yaw = math.atan2(velocity[1], velocity[0]) if np.linalg.norm(velocity[:2]) > 0.05 else 0.0
        rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1.0]])
        slots = p_now + self._formation(time_s) @ rotation.T
        slots[:, 2] = np.maximum(slots[:, 2], self.ground(slots[:, 0], slots[:, 1]) + 1.2)
        n = len(slots)
        phase = np.arange(n) * 1.7
        wobble = 0.06 * np.stack((np.sin(1.3 * time_s + phase), np.cos(1.1 * time_s + phase),
                                  0.5 * np.sin(0.9 * time_s + phase)), axis=1)
        positions = slots + wobble
        air = self.wind(positions, time_s) - velocity
        drag = DRAG_PER_MASS * air * np.linalg.norm(air, axis=1, keepdims=True)
        thrust = acceleration + np.array([0.0, 0.0, GRAVITY]) - drag
        thrust /= np.linalg.norm(thrust, axis=1, keepdims=True)
        horizontal = np.linalg.norm(thrust[:, :2], axis=1)
        too_steep = np.arctan2(horizontal, thrust[:, 2]) > MAX_TILT
        if np.any(too_steep):
            scale = math.tan(MAX_TILT) / np.maximum(horizontal[too_steep], 1e-9)
            thrust[too_steep, :2] *= scale[:, None]
            thrust[too_steep, 2] = 1.0
            thrust /= np.linalg.norm(thrust, axis=1, keepdims=True)
        heading = np.array([[math.cos(-yaw), -math.sin(-yaw), 0.0], [math.sin(-yaw), math.cos(-yaw), 0.0], [0, 0, 1.0]])
        local = thrust @ heading.T
        pitch = np.arctan2(local[:, 0], local[:, 2]) + np.radians(1.5) * np.sin(2.1 * time_s + phase)
        roll = np.arctan2(-local[:, 1], np.hypot(local[:, 0], local[:, 2])) + np.radians(1.5) * np.cos(1.7 * time_s + phase)
        quaternions = quat_from_euler(roll, pitch, np.full(n, yaw))
        return positions, quaternions, slots, p_now, velocity

    def _crossing(self, time_s):
        """A sphere that crosses ahead of the swarm for 8 s out of every 20 s."""
        cycle, local = divmod(time_s, 20.0)
        if not 8.0 <= local < 16.0:
            return []
        start = cycle * 20.0 + 8.0
        center = self._target(start + 4.0)
        direction = 1.0 if math.cos(PATH_RATE * (start + 4.0)) >= 0 else -1.0
        y = -12.0 + 3.0 * (local - 8.0)
        x = float(center[0]) + 6.0 * direction
        return [{"kind": "sphere", "p": [round(x, 3), round(y, 3), round(float(center[2]), 3)], "radius": 0.7}]

    def frame(self, time_s):
        positions, quaternions, slots, target, velocity = self._poses(time_s)
        points = np.column_stack((self.wind_points, self.ground(self.wind_points[:, 0], self.wind_points[:, 1]) + 4.5))
        wind = self.wind(points, time_s)
        latest = self.trace(time_s)[-1]
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (quaternions[:, 1] ** 2 + quaternions[:, 2] ** 2), -1, 1)))
        centroid_wind = self.wind(positions.mean(axis=0), time_s)
        metrics = {"mean_tilt_deg": round(float(tilt.mean()), 2), "max_tilt_deg": round(float(tilt.max()), 2),
                   "wind_speed_mps": round(float(np.linalg.norm(centroid_wind)), 2),
                   "speed_mps": round(float(np.linalg.norm(velocity)), 2),
                   "agl_m": round(float(target[2] - self.ground(target[0], target[1])), 2),
                   "region": self._region(float(target[0])), "physics": "none (Phase 0 scripted demo)"}
        return make_frame(time_s, positions, quaternions, slots, target, velocity, zip(points, wind),
                          self._crossing(time_s), latest, metrics)
