import itertools

import numpy as np
import pytest

from swarmv2.config import FORMATIONS, DemoConfig, WindConfig
from swarmv2.viewer.demo import DemoScene, formation_offsets, verify_decision
from swarmv2.viewer.scene import quat_from_euler, quat_rotate, validate_frame


@pytest.fixture(scope="module")
def demo():
    return DemoScene(seed=1000)


def test_frames_are_valid_with_eight_unit_quaternion_drones(demo):
    for t in np.linspace(0, 240, 961):
        frame = validate_frame(demo.frame(float(t)))
        assert len(frame["drones"]) == 8
        q = np.array([d["q"] for d in frame["drones"]])
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)
        tilt = np.array([frame["metrics"]["mean_tilt_deg"], frame["metrics"]["max_tilt_deg"]])
        assert np.all(tilt >= 0) and tilt[1] <= 35 + 1.6, f"tilt {tilt[1]} at t={t}"
        ground = demo.ground(np.array([d["p"][0] for d in frame["drones"]]), np.array([d["p"][1] for d in frame["drones"]]))
        assert np.all(np.array([d["p"][2] for d in frame["drones"]]) > ground + 0.8)


def test_terrain_is_finite_and_matches_its_size(demo):
    terrain = demo.terrain()
    heights = np.array(terrain["heights"])
    assert heights.shape == (terrain["cells"], terrain["cells"]) and np.isfinite(heights).all()
    assert len([o for o in terrain["obstacles"] if o["kind"] == "pillar"]) == demo.config.terrain.pillars
    canyon_floor = demo.ground(np.linspace(0, 15, 6), np.zeros(6))
    canyon_wall = demo.ground(np.linspace(0, 15, 6), np.full(6, 7.0))
    assert np.all(canyon_wall > canyon_floor + 2)


def test_scene_is_deterministic_per_seed(demo):
    again = DemoScene(seed=1000)
    assert again.frame(42.0) == demo.frame(42.0)
    assert not np.array_equal(DemoScene(seed=1001).heights, demo.heights)


def test_drones_lean_into_stronger_wind():
    calm = DemoScene(DemoConfig(wind=WindConfig(mean_mps=(0.0, 0.0, 0.0), gust_sigma_mps=0.0)), seed=1000)
    windy = DemoScene(DemoConfig(wind=WindConfig(mean_mps=(9.0, 0.0, 0.0), gust_sigma_mps=0.0)), seed=1000)
    times = np.linspace(5, 60, 12)
    calm_tilt = np.mean([calm.frame(float(t))["metrics"]["mean_tilt_deg"] for t in times])
    windy_tilt = np.mean([windy.frame(float(t))["metrics"]["mean_tilt_deg"] for t in times])
    assert windy_tilt > calm_tilt + 3
    frame = windy.frame(30.0)
    up = quat_rotate(np.array([d["q"] for d in frame["drones"]]), np.array([0.0, 0.0, 1.0]))
    assert np.mean(up[:, 0]) < 0, "thrust must tilt upwind (against +x wind)"


def test_quaternions_from_euler_rotate_as_expected():
    q = quat_from_euler(np.radians(10.0), 0.0, 0.0)
    up = quat_rotate(q, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(np.degrees(np.arccos(up[2])), 10.0, atol=1e-6)
    assert up[1] < 0
    forward = quat_rotate(quat_from_euler(0.0, 0.0, np.pi / 2), np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(forward, [0.0, 1.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("preset", FORMATIONS)
def test_formations_are_centred_and_keep_their_spacing(preset):
    offsets = formation_offsets(preset, 8, 1.4)
    np.testing.assert_allclose(offsets.mean(axis=0), 0.0, atol=1e-9)
    gaps = [np.linalg.norm(a - b) for a, b in itertools.combinations(offsets, 2)]
    assert min(gaps) >= 1.4 * 0.95


def test_verifier_clamps_numbers_and_rejects_invalid_output():
    decision, clamped, fault = verify_decision({"flight_mode": "AGILE", "formation_preset": "LINE",
                                                "agility_alpha": 3, "formation_scale_m": 0.1,
                                                "cruise_speed_cap_mps": 40, "target_agl_m": 0.2,
                                                "waypoint_offset_world": [500, 0, 0]}, radius_m=0.18)
    assert fault is None
    assert decision["agility_alpha"] == 1.0 and decision["formation_scale_m"] >= 2.5 * 0.18
    assert decision["cruise_speed_cap_mps"] == 6.0 and decision["target_agl_m"] == 1.5
    assert decision["waypoint_offset_world"] == [50.0, 0.0, 0.0]
    assert set(clamped) == {"agility_alpha", "formation_scale_m", "cruise_speed_cap_mps", "target_agl_m",
                            "waypoint_offset_world"}
    for bad in ("not json", {"flight_mode": "WARP", "formation_preset": "LINE"}):
        decision, _, fault = verify_decision(bad, radius_m=0.18)
        assert fault and decision["flight_mode"] == "STEADY"


def test_trace_shows_clamps_and_faults_from_the_mock_planner(demo):
    trace = demo.trace(30.0)
    assert [entry["index"] for entry in trace] == list(range(len(trace)))
    assert any(entry["clamped"] for entry in trace) and any(entry["fault"] for entry in trace)
    assert {entry["decision"]["formation_preset"] for entry in demo.trace(120.0)} >= {"DIAMOND", "COLUMN"}
