"""Episode metrics, 95% intervals and acceptance checks (ported from swarmflow.evaluate, new schema).

An episode record is a dict with: scenario, seed, drones, duration_s, failure (None or a reason),
collision, min_clearance_m, slot_rmse_m, events [{name, recovery_s}], and optionally
tilt_jitter_deg, wind_tracking_error_mps, reasoning_transitions, shield_share.
"""
import math

import numpy as np

Z_95 = 1.959963984540054


def mean_ci95(values):
    """Mean with a normal-approximation 95% interval (None for fewer than two values)."""
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return {"mean": None, "std": None, "low": None, "high": None, "n": 0}
    mean = float(values.mean())
    if values.size < 2:
        return {"mean": mean, "std": 0.0, "low": None, "high": None, "n": 1}
    std = float(values.std(ddof=1))
    half = Z_95 * std / math.sqrt(values.size)
    return {"mean": mean, "std": std, "low": mean - half, "high": mean + half, "n": int(values.size)}


def proportion_ci95(successes, n):
    """Wilson 95% interval for a success rate; well behaved at 0% and 100%, unlike the normal interval."""
    if n <= 0:
        return {"rate": None, "low": None, "high": None, "n": 0}
    rate = successes / n
    z2 = Z_95 ** 2
    centre = (rate + z2 / (2 * n)) / (1 + z2 / n)
    half = Z_95 * math.sqrt(rate * (1 - rate) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return {"rate": rate, "low": max(0.0, centre - half), "high": min(1.0, centre + half), "n": int(n)}


def tilt_deg(quaternions):
    """Tilt of the body z axis from world up, in degrees, for unit quaternions (..., 4) ordered w, x, y, z."""
    q = np.asarray(quaternions, dtype=float)
    cos_tilt = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)
    return np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))


def tilt_jitter_deg(quaternions, steady=None):
    """Steady-mode tilt jitter: standard deviation over time of each drone's tilt, averaged over drones.

    quaternions: (steps, drones, 4). steady: optional (steps,) boolean mask of STEADY-mode steps.
    """
    tilt = tilt_deg(quaternions)
    if steady is not None:
        tilt = tilt[np.asarray(steady, dtype=bool)]
    if tilt.shape[0] < 2:
        return None
    return float(tilt.std(axis=0, ddof=1).mean())


def wind_tracking_error(estimated, true):
    """RMS norm of the wind-estimate error, in m/s; arrays of shape (..., 3)."""
    error = np.asarray(estimated, dtype=float) - np.asarray(true, dtype=float)
    return float(np.sqrt(np.mean(np.sum(error ** 2, axis=-1))))


def reasoning_transitions(decisions, keys=("flight_mode", "formation_preset")):
    """How often consecutive Layer 3 decisions changed each discrete field, plus the total."""
    counts = {key: 0 for key in keys}
    for before, after in zip(decisions, decisions[1:]):
        for key in keys:
            counts[key] += int(before.get(key) != after.get(key))
    counts["total"] = sum(counts[key] for key in keys)
    return counts


def aggregate(episodes, recovery_limit_s=3.5):
    """Per-scenario summary with 95% intervals."""
    result = {}
    for scenario in dict.fromkeys(e["scenario"] for e in episodes):
        rows = [e for e in episodes if e["scenario"] == scenario]
        n = len(rows)
        failures = sum(bool(e["failure"]) for e in rows)
        events = [(event, row) for row in rows for event in row.get("events", [])]
        quick = [event for event, row in events if not row["failure"] and event.get("recovery_s") is not None
                 and event["recovery_s"] <= recovery_limit_s]
        clearances = [e["min_clearance_m"] for e in rows if e.get("min_clearance_m") is not None]
        summary = {"episodes": n, "failures": failures,
                   "success": proportion_ci95(n - failures, n),
                   "slot_rmse_m": mean_ci95(e.get("slot_rmse_m") for e in rows),
                   "min_clearance_m": min(clearances) if clearances else None,
                   "events": len(events), "recovered_within_limit": len(quick)}
        for key in ("tilt_jitter_deg", "wind_tracking_error_mps", "reasoning_transitions", "shield_share"):
            if any(e.get(key) is not None for e in rows):
                summary[key] = mean_ci95(e.get(key) for e in rows)
        result[scenario] = summary
    return result


def selection_score(summary):
    """One number to rank checkpoints: success dominates, then slot error and recovery (as in v1)."""
    rows = list(summary.values())
    success = float(np.mean([row["success"]["rate"] for row in rows]))
    rmse = float(np.mean([min(row["slot_rmse_m"]["mean"], 2.0) for row in rows]))
    events = sum(row["events"] for row in rows)
    recovery = sum(row["recovered_within_limit"] for row in rows) / events if events else 1.0
    return success - 0.25 * rmse - 0.25 * (1.0 - recovery)


def learning_checks(episodes, success_target=0.95, recovery_limit_s=3.5, recovery_target=0.95):
    """Pass/fail checks of a set of episodes against the configured gate (spec Phase 3 by default)."""
    n = len(episodes)
    completed = sum(not e["failure"] for e in episodes)
    events = [event for e in episodes for event in e.get("events", [])]
    quick = [event for e in episodes if not e["failure"] for event in e.get("events", [])
             if event.get("recovery_s") is not None and event["recovery_s"] <= recovery_limit_s]
    checks = [{"name": f"Collision-free completion >= {success_target:.0%}",
               "passed": n > 0 and completed >= success_target * n, "detail": f"{completed}/{n} episodes"}]
    if events:
        checks.append({"name": f">= {recovery_target:.0%} of slots recovered within {recovery_limit_s:g} s",
                       "passed": len(quick) >= recovery_target * len(events), "detail": f"{len(quick)}/{len(events)}"})
    return checks
