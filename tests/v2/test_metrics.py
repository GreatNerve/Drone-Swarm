import json

import numpy as np
import pytest

from swarmv2.eval.compare import compare_reports
from swarmv2.eval.metrics import (aggregate, learning_checks, mean_ci95, proportion_ci95, reasoning_transitions,
                                  selection_score, tilt_deg, tilt_jitter_deg, wind_tracking_error)
from swarmv2.eval.report import build_report, markdown, write_report
from swarmv2.viewer.scene import quat_from_euler


def test_wilson_interval_stays_inside_zero_and_one():
    assert proportion_ci95(95, 100)["low"] == pytest.approx(0.8882, abs=1e-3)
    assert proportion_ci95(95, 100)["high"] == pytest.approx(0.9785, abs=1e-3)
    perfect = proportion_ci95(20, 20)
    assert perfect["rate"] == 1.0 and perfect["high"] == 1.0 and 0.8 < perfect["low"] < 0.9
    assert proportion_ci95(0, 0)["rate"] is None


def test_mean_interval():
    stat = mean_ci95([1.0, 2.0, 3.0, None])
    assert stat["mean"] == 2.0 and stat["n"] == 3
    assert stat["low"] == pytest.approx(2 - 1.96 / np.sqrt(3), abs=1e-3)
    assert mean_ci95([4.0])["low"] is None


def test_tilt_and_steady_mode_jitter():
    np.testing.assert_allclose(tilt_deg(quat_from_euler(np.radians(12.0), 0.0, 1.0)), 12.0, atol=1e-6)
    steps = np.arange(400)
    roll = np.radians(5.0 + 2.0 * np.sin(steps * 0.1))
    q = quat_from_euler(roll, 0.0, 0.0)[:, None, :].repeat(3, axis=1)
    assert tilt_jitter_deg(q) == pytest.approx(np.std(np.degrees(roll), ddof=1), rel=1e-6)
    steady = steps < 200
    constant = quat_from_euler(np.where(steady, np.radians(3.0), np.radians(20.0)), 0.0, 0.0)[:, None, :]
    assert tilt_jitter_deg(constant, steady) == pytest.approx(0.0, abs=1e-9)


def test_wind_error_and_reasoning_transitions():
    assert wind_tracking_error(np.zeros((5, 3)), np.tile([3.0, 4.0, 0.0], (5, 1))) == pytest.approx(5.0)
    decisions = [{"flight_mode": "STEADY", "formation_preset": "DIAMOND"},
                 {"flight_mode": "AGILE", "formation_preset": "DIAMOND"},
                 {"flight_mode": "AGILE", "formation_preset": "COLUMN"},
                 {"flight_mode": "STEADY", "formation_preset": "LINE"}]
    assert reasoning_transitions(decisions) == {"flight_mode": 2, "formation_preset": 2, "total": 4}


def _episodes():
    rows = []
    for seed in range(100000, 100010):
        failed = seed == 100003
        rows.append({"scenario": "canyon", "seed": seed, "drones": 8, "duration_s": 30.0,
                     "failure": "collision" if failed else None, "collision": failed,
                     "min_clearance_m": -0.01 if failed else 0.2 + 0.01 * (seed % 7),
                     "slot_rmse_m": 0.2 + 0.01 * (seed % 5), "events": [{"name": "o", "recovery_s": 2.0 + seed % 3}],
                     "tilt_jitter_deg": 1.5, "reasoning_transitions": 3})
    return rows


def test_aggregate_checks_report_and_compare(tmp_path):
    episodes = _episodes()
    summary = aggregate(episodes)["canyon"]
    assert summary["episodes"] == 10 and summary["failures"] == 1
    assert summary["success"]["rate"] == 0.9 and summary["min_clearance_m"] == -0.01
    assert summary["recovered_within_limit"] == 6 and summary["tilt_jitter_deg"]["mean"] == 1.5
    checks = learning_checks(episodes)
    assert [c["passed"] for c in checks] == [False, False]
    report = build_report(episodes, "synthetic", metadata={"seeds": "100000..100009"})
    assert report["score"] == pytest.approx(selection_score(report["summary"]))
    text = markdown(report)
    assert "| canyon | 10 | 90.0% [" in text and "FAIL" in text
    write_report(report, tmp_path / "a.json", tmp_path / "a.md")
    better = build_report([{**e, "failure": None, "collision": False, "min_clearance_m": 0.3} for e in episodes], "b")
    write_report(better, tmp_path / "b.json")
    diff = compare_reports((tmp_path / "a.json", tmp_path / "b.json"), tmp_path / "diff.json")
    assert diff["summary"]["canyon"]["delta_success"] == pytest.approx(0.1)
    assert json.loads((tmp_path / "diff.json").read_text())["summary"]["canyon"]["success"] == [0.9, 1.0]
    assert (tmp_path / "a.md").read_text().startswith("# synthetic")
