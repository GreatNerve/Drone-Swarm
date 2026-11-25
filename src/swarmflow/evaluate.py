import json
from pathlib import Path
import numpy as np

from .baseline import PredictiveController, SlotController
from .config import Config
from .env import RECOVERY_WINDOW_S, SwarmEnv
from .files import read_json, write_json_atomic, write_text_atomic
from .scenarios import SCENARIOS

# Acceptance targets from docs/research-plan.md.
FORMATION_SCENARIOS = ("hold", "follow", "turn")
FORMATION_RMSE_M = 0.35
RECOVERY_LIMIT_S = RECOVERY_WINDOW_S
RECOVERY_RATE = 0.95
# Checkpoint selection asks for a safety margin, not just survival: a model that passes the selection
# seeds with 2 cm to spare grazes on unseen seeds (run 3), so clearance below 0.15 m costs score.
CLEARANCE_TARGET_M = 0.15
CLEARANCE_WEIGHT = 0.5


def run_episode(config, scenario, seed, controller, trace=False):
    env = SwarmEnv(config, scenario)
    observations, _ = env.reset(seed=seed)
    if hasattr(controller, "reset"):
        controller.reset()
    frames = [env.snapshot()] if trace else None
    while env.agents:
        observations, _, _, _, _ = env.step(controller(observations))
        if trace and (env.steps % 2 == 0 or not env.agents):
            frames.append(env.snapshot())
    summary = env.summary()
    if hasattr(controller, "episode_stats"):
        summary.update(controller.episode_stats())
    env.close()
    return summary, frames


def _success_rate(row):
    if "success_rate" in row:
        return row["success_rate"]
    n = row.get("episodes") or 0
    return None if n == 0 else 1.0 - row["failures"] / n


def aggregate(episodes):
    result = {}
    for scenario in dict.fromkeys(e["scenario"] for e in episodes):
        rows = [e for e in episodes if e["scenario"] == scenario]
        events = [(event, row) for row in rows for event in row["events"]]
        recovered = [event["recovery_s"] for event, row in events
                     if event["recovery_s"] is not None and not row["failure"]]
        n = len(rows)
        slot = np.array([e["slot_rmse_m"] for e in rows], dtype=float)
        mean = float(np.mean(slot))
        std = float(np.std(slot, ddof=1)) if n >= 2 else 0.0
        failures = sum(bool(e["failure"]) for e in rows)
        clearances = [e["min_surface_clearance_m"] for e in rows if e["min_surface_clearance_m"] is not None]
        interval = None
        if n >= 2:
            half = 1.96 * std / float(np.sqrt(n))
            interval = {"low": float(mean - half), "high": float(mean + half)}
        result[scenario] = {"episodes": n, "failures": failures,
            "collisions": sum(e["collision"] for e in rows),
            "success_rate": 1.0 - failures / n,
            "mean_slot_rmse_m": mean,
            "slot_rmse_std_m": std,
            "slot_rmse_ci95_m": interval,
            "min_surface_clearance_m": min(clearances) if clearances else None,
            "events": len(events), "recovered_events": len(recovered),
            "mean_recovery_s_conditional": float(np.mean(recovered)) if recovered else None,
            "max_recovery_s_conditional": max(recovered) if recovered else None}
        if any("filter_decisions" in e for e in rows):
            decisions = sum(e.get("filter_decisions", 0) for e in rows)
            result[scenario]["filter_interventions"] = sum(e.get("filter_interventions", 0) for e in rows)
            result[scenario]["filter_decisions"] = decisions
            result[scenario]["filter_share"] = result[scenario]["filter_interventions"] / decisions if decisions else 0.0
    return result


def selection_score(summary):
    """Single number used to rank checkpoints; 1.0 means no failures, exact slots, every event recovered.

    Survival dominates: one failed episode in five costs 0.2, the same as a 0.8 m slot RMSE.
    """
    rows = list(summary.values())
    success = float(np.mean([row["success_rate"] for row in rows]))
    rmse = float(np.mean([min(row["mean_slot_rmse_m"], 2.0) for row in rows]))
    events = sum(row["events"] for row in rows)
    recovery = sum(row["recovered_events"] for row in rows) / events if events else 1.0
    return success - 0.25 * rmse - 0.25 * (1.0 - recovery)


def min_obstacle_clearance(summary):
    """Lowest surface clearance in the scenarios with moving objects (None if there are none)."""
    values = [row["min_surface_clearance_m"] for row in summary.values()
              if row["events"] and row["min_surface_clearance_m"] is not None]
    return min(values) if values else None


def checkpoint_score(summary):
    """selection_score minus 0.5 x the mean shortfall of each obstacle scenario's minimum clearance below 0.15 m.

    Used only to choose best.pt during training; reported test scores stay selection_score.
    """
    shortfalls = [max(0.0, CLEARANCE_TARGET_M - row["min_surface_clearance_m"]) for row in summary.values()
                  if row["events"] and row["min_surface_clearance_m"] is not None]
    penalty = CLEARANCE_WEIGHT * float(np.mean(shortfalls)) if shortfalls else 0.0
    return selection_score(summary) - penalty


def learning_checks(report):
    """Pass/fail checks of a report against the research-plan acceptance targets."""
    episodes = report["episodes"]
    failures = [e for e in episodes if e["failure"]]
    events = [event for e in episodes for event in e["events"]]
    quick = [event for e in episodes if not e["failure"] for event in e["events"]
             if event["recovery_s"] is not None and event["recovery_s"] <= RECOVERY_LIMIT_S]
    checks = [{"name": "No collisions or other failures", "passed": not failures,
               "detail": f"{len(episodes) - len(failures)}/{len(episodes)} episodes completed"}]
    if events:
        checks.append({"name": f">= {RECOVERY_RATE:.0%} of disturbances recovered within {RECOVERY_LIMIT_S:g} s",
                       "passed": len(quick) >= RECOVERY_RATE * len(events),
                       "detail": f"{len(quick)}/{len(events)} events"})
    formation = {name: row["mean_slot_rmse_m"] for name, row in report["summary"].items()
                 if name in FORMATION_SCENARIOS}
    if formation:
        checks.append({"name": f"Slot RMSE below {FORMATION_RMSE_M:g} m when undisturbed",
                       "passed": max(formation.values()) < FORMATION_RMSE_M,
                       "detail": ", ".join(f"{name} {value:.3f} m" for name, value in formation.items())})
    return checks


def filter_share(report):
    """Share of drone decisions the safety filter handed to the teacher in a report (None without a filter)."""
    episodes = report["episodes"]
    decisions = sum(e.get("filter_decisions", 0) for e in episodes)
    if not any("filter_decisions" in e for e in episodes):
        return None
    return sum(e.get("filter_interventions", 0) for e in episodes) / decisions if decisions else 0.0


def compare_reports(paths, output="runs/compare.json"):
    paths = [Path(p) for p in paths]
    if len(paths) != 2:
        raise ValueError("compare requires exactly two report paths")
    loaded = []
    for path in paths:
        try:
            loaded.append(read_json(path))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}: {error}") from error
    left, right = (item.get("summary") or {} for item in loaded)
    names = list(dict.fromkeys([*left, *right]))
    summary = {}
    for name in names:
        a, b = left.get(name, {}), right.get(name, {})
        success = [_success_rate(a), _success_rate(b)]
        rmse = [a.get("mean_slot_rmse_m"), b.get("mean_slot_rmse_m")]
        entry = {"success_rate": success, "mean_slot_rmse_m": rmse}
        if None not in success:
            entry["delta_success_rate"] = success[1] - success[0]
        if None not in rmse:
            entry["delta_mean_slot_rmse_m"] = rmse[1] - rmse[0]
        summary[name] = entry
    report = {"reports": [str(p) for p in paths], "summary": summary}
    write_json_atomic(output, report)
    return report


def make_controller(config, policy="baseline", checkpoint=None, safety_filter=False, filter_options=None):
    """Controller for a report: baseline, slot-only or a learned checkpoint, optionally behind the safety filter."""
    if policy == "learned":
        from .policy import LearnedController
        if checkpoint is None:
            raise ValueError("--checkpoint is required for learned evaluation")
        controller = LearnedController(checkpoint)
    else:
        controller = PredictiveController(config) if policy == "baseline" else SlotController(config)
    if safety_filter:
        from .shield import SafetyFilter
        controller = SafetyFilter(controller, config, **(filter_options or {}))
    return controller


def evaluate(config=None, scenarios=SCENARIOS, seeds=range(1000, 1003), policy="baseline", checkpoint=None,
             output="runs/evaluation.json", trace=False, safety_filter=False, filter_options=None, quiet=False):
    config = config or Config()
    controller = make_controller(config, policy, checkpoint, safety_filter, filter_options)
    episodes = []
    for scenario in scenarios:
        for seed in seeds:
            summary, frames = run_episode(config, scenario, seed, controller, trace)
            episodes.append(summary)
            if frames is not None:
                target = Path(output).with_name(f"{Path(output).stem}-{scenario}-{seed}-trace.json")
                write_text_atomic(target, json.dumps(frames))
            if not quiet:
                extra = (f" filter={summary['filter_interventions']}/{summary['filter_decisions']}"
                         if "filter_decisions" in summary else "")
                print(f"{scenario:16s} seed={seed} failure={summary['failure']} "
                      f"slot_rmse={summary['slot_rmse_m']:.3f} m{extra}", flush=True)
    report = {"model": "3D stabilized flight; bounded acceleration and finite velocity response",
              "policy": policy, "checkpoint": str(checkpoint) if checkpoint else None,
              "safety_filter": controller.describe() if safety_filter else None,
              "config": config.to_dict(), "summary": aggregate(episodes), "episodes": episodes}
    if output:
        write_json_atomic(output, report)
    return report
