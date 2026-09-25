import numpy as np

from swarm.config import load
from swarm.env.swarm_env import SwarmEnv, body_overlap, distance_cost


def test_collision_flag_when_close():
    cfg = load()
    env = SwarmEnv(cfg, test_id=1, seed=0)
    env.drones[0].pos[:] = [0.0, 0.0, 200.0]
    env.drones[1].pos[:] = [5.0, 0.0, 200.0]
    pair, _floor = env._collisions(env.positions())
    assert pair


def test_same_xy_different_altitude_is_not_collision():
    cfg = load()
    env = SwarmEnv(cfg, test_id=1, seed=0)
    env.drones[0].pos[:] = [0.0, 0.0, 200.0]
    env.drones[1].pos[:] = [5.0, 0.0, 250.0]
    pair, _floor = env._collisions(env.positions())
    assert not pair


def test_observation_length_and_k_cap():
    cfg = load()
    env = SwarmEnv(cfg, test_id=1, seed=1)
    obs = env.reset()
    k = int(cfg["neighbor_k"])
    assert obs.shape == (int(cfg["num_drones"]), 18 + 6 * k)
    nei = obs[0, 18:].reshape(k, 6)
    assert nei.shape[0] == k


def test_collision_impulse_separates():
    cfg = load()
    env = SwarmEnv(cfg, test_id=1, seed=0)
    env.drones[0].pos[:] = [0.0, 0.0, 200.0]
    env.drones[1].pos[:] = [5.0, 0.0, 200.0]
    env.drones[0].vel[:] = [50.0, 0.0, 0.0]
    env.drones[1].vel[:] = [-50.0, 0.0, 0.0]
    pos = env.positions()
    env._separate(pos, body_overlap(pos, env.collision_cm, env.body_z))
    n = env.drones[0].pos - env.drones[1].pos
    n = n / (np.linalg.norm(n) + 1e-12)
    closing = float(np.dot(env.drones[0].vel - env.drones[1].vel, n))
    assert closing >= -1e-4


def test_train_episode_ends_on_collision():
    cfg = load()
    env = SwarmEnv(cfg, test_id=1, seed=0, eval_mode=False)
    env.reset()
    env.drones[0].pos[:] = [0.0, 0.0, 200.0]
    env.drones[1].pos[:] = [5.0, 0.0, 200.0]
    act = np.zeros((env.n, 4))
    step = env.step(act)
    assert env.collided
    assert step.done


def test_eval_episode_keeps_running_after_collision():
    cfg = load()
    env = SwarmEnv(cfg, test_id=1, seed=0, eval_mode=True)
    env.reset()
    env.drones[0].pos[:] = [0.0, 0.0, 200.0]
    env.drones[1].pos[:] = [5.0, 0.0, 200.0]
    act = np.zeros((env.n, 4))
    step = env.step(act)
    assert env.collided
    assert not step.done


def test_test1_reward_prefers_standoff_over_stack():
    cfg = load()
    env = SwarmEnv(cfg, test_id=1, seed=0)
    env.reset()
    tgt = env.task_state.shared_target
    stand = float(cfg["test1_dist_cm"]) * 0.75
    stacked = np.repeat(tgt[None, :], env.n, axis=0)
    ring = stacked.copy()
    for i in range(env.n):
        ang = 2.0 * np.pi * i / env.n
        ring[i, 0] = tgt[0] + stand * np.cos(ang)
        ring[i, 1] = tgt[1] + stand * np.sin(ang)
    act = np.zeros((env.n, 4))
    r_stack = env._rewards(stacked, act, False, False)
    r_ring = env._rewards(ring, act, False, False)
    assert float(r_ring.mean()) > float(r_stack.mean())


def test_reward_far_from_slot_is_much_worse_than_near():
    cfg = load()
    env = SwarmEnv(cfg, test_id=2, seed=0)
    env.reset()
    aim = env.task_state.aim_points
    near = aim + np.array([10.0, 0.0, 0.0])
    mid = aim + np.array([50.0, 0.0, 0.0])
    far = aim + np.array([200.0, 0.0, 0.0])
    act = np.zeros((env.n, 4))
    r_near = float(env._rewards(near, act, False, False).mean())
    r_mid = float(env._rewards(mid, act, False, False).mean())
    r_far = float(env._rewards(far, act, False, False).mean())
    assert r_near > r_mid > r_far


def test_reward_ignores_error_inside_one_body():
    """~10 cm is one body. 1–5 cm is on-station; 20 cm and 50 cm are not."""
    cfg = load()
    env = SwarmEnv(cfg, test_id=2, seed=0)
    env.reset()
    aim = env.task_state.aim_points
    act = np.zeros((env.n, 4))
    for drone in env.drones:
        drone.vel[:] = 0.0
    r0 = float(env._rewards(aim, act, False, False).mean())
    r5 = float(env._rewards(aim + np.array([5.0, 0.0, 0.0]), act, False, False).mean())
    r20 = float(env._rewards(aim + np.array([20.0, 0.0, 0.0]), act, False, False).mean())
    r50 = float(env._rewards(aim + np.array([50.0, 0.0, 0.0]), act, False, False).mean())
    assert abs(r0 - r5) < 0.05
    assert r20 < r5
    assert r50 < r20


def test_distance_cost_uncapped_and_on_station():
    """10 m costs more than 2 m; 5 cm is on-station."""
    c2 = float(distance_cost(np.array([200.0]))[0])
    c10 = float(distance_cost(np.array([1000.0]))[0])
    assert c10 > c2
    assert float(distance_cost(np.array([5.0]))[0]) == 0.0
    assert float(distance_cost(np.array([10.0]))[0]) == 0.0


def test_reward_does_not_prefer_stall_when_far():
    cfg = load()
    env = SwarmEnv(cfg, test_id=2, seed=0)
    env.reset()
    aim = env.task_state.aim_points
    far = aim + np.array([200.0, 0.0, 0.0])
    for i, drone in enumerate(env.drones):
        drone.pos[:] = far[i]
        drone.vel[:] = 0.0
    rest = np.zeros((env.n, 4))
    slam = np.ones((env.n, 4)) * 0.9
    r_rest = env._rewards(far, rest, False, False)
    r_slam = env._rewards(far, slam, False, False)
    assert float(r_slam.mean()) >= float(r_rest.mean()) - 1e-5


def test_reward_prefers_closing_when_far():
    cfg = load()
    env = SwarmEnv(cfg, test_id=2, seed=0)
    env.reset()
    aim = env.task_state.aim_points
    far = aim + np.array([200.0, 0.0, 0.0])
    act = np.zeros((env.n, 4))
    for i, drone in enumerate(env.drones):
        drone.pos[:] = far[i]
        drone.vel[:] = [100.0, 0.0, 0.0]
    r_away = env._rewards(far, act, False, False)
    for drone in env.drones:
        drone.vel[:] = [-100.0, 0.0, 0.0]
    r_close = env._rewards(far, act, False, False)
    assert float(r_close.mean()) > float(r_away.mean())


def test_reward_prefers_altitude_split_over_same_height():
    cfg = load()
    env = SwarmEnv(cfg, test_id=2, seed=0)
    env.reset()
    # On-station XY so distance cost does not hide the proximity term.
    stacked = env.task_state.aim_points.copy()
    stacked[0] = np.array([0.0, 0.0, 200.0])
    stacked[1] = np.array([12.0, 0.0, 200.0])
    env.task_state.aim_points[0] = stacked[0].copy()
    env.task_state.aim_points[1] = stacked[1].copy()
    split = stacked.copy()
    split[1, 2] = 231.0
    for drone in env.drones:
        drone.vel[:] = 0.0
    act = np.zeros((env.n, 4))
    env.prev_dist_cm = None
    r_stack = env._rewards(stacked, act, False, False)
    env.prev_dist_cm = None
    r_split = env._rewards(split, act, False, False)
    assert float(r_split[:2].mean()) > float(r_stack[:2].mean())


def test_set_test_id_switches_task():
    cfg = load()
    env = SwarmEnv(cfg, test_id=1, seed=0)
    env.reset()
    t1_aim = env.task_state.aim_points.copy()
    env.set_test_id(2)
    assert env.test_id == 2
    assert not np.allclose(env.task_state.aim_points, t1_aim)
    try:
        env.set_test_id(9)
        assert False, "expected ValueError"
    except ValueError:
        pass
