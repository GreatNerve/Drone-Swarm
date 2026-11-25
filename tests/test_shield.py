import numpy as np

from swarmflow import Config, SwarmEnv
from swarmflow.baseline import SlotController
from swarmflow.evaluate import evaluate, filter_share, run_episode
from swarmflow.shield import SafetyFilter

CONFIG = Config(drones=4)


def test_filter_turns_certain_collisions_into_safe_flights():
    for scenario in ("head_on", "crossing"):
        blind, _ = run_episode(CONFIG, scenario, 1, SlotController(CONFIG))
        assert blind["failure"] == "collision"
        shielded, _ = run_episode(CONFIG, scenario, 1, SafetyFilter(SlotController(CONFIG), CONFIG))
        assert shielded["failure"] is None and shielded["min_surface_clearance_m"] > 0
        assert 0 < shielded["filter_interventions"] < shielded["filter_decisions"]


def test_filter_leaves_an_undisturbed_flight_untouched():
    env = SwarmEnv(CONFIG, "hold")
    observations, _ = env.reset(seed=2)
    inner = SlotController(CONFIG)
    shield = SafetyFilter(inner, CONFIG)
    while env.agents:
        expected, actual = inner(observations), shield(observations)
        assert all(np.array_equal(expected[a], actual[a]) for a in observations)
        observations, *_ = env.step(actual)
    assert shield.interventions == 0 and shield.decisions == env.steps * CONFIG.drones


def test_reports_count_filter_interventions_per_episode():
    report = evaluate(CONFIG, ["hold", "head_on"], range(1, 3), "slot", output=None, safety_filter=True,
                      quiet=True)
    assert report["safety_filter"]["margin_m"] == 0.1
    assert all("filter_decisions" in episode for episode in report["episodes"])
    assert report["summary"]["hold"]["filter_interventions"] == 0
    assert report["summary"]["head_on"]["filter_share"] > 0
    assert 0 < filter_share(report) < 0.5
