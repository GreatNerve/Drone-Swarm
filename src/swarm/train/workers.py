from __future__ import annotations

import os
from multiprocessing import get_context
from typing import Any

import numpy as np

from swarm.env.swarm_env import SwarmEnv

_ctx = get_context("spawn")
_KINDS = ("square", "rectangle", "circle")


def _shuffle_geometry(env: SwarmEnv) -> None:
    if env.test_id != 2:
        return
    env.set_geometry_kind(_KINDS[int(env.rng.integers(0, 3))])


def _worker(conn, cfg: dict, test_id: int, seed: int) -> None:
    env = SwarmEnv(cfg, test_id=test_id, seed=seed)
    _shuffle_geometry(env)
    conn.send(env.reset())
    while True:
        cmd, payload = conn.recv()
        if cmd == "reset":
            _shuffle_geometry(env)
            conn.send(env.reset())
        elif cmd == "step":
            result = env.step(payload)
            conn.send((result.obs, result.rewards, result.done, result.info))
        elif cmd == "close":
            conn.close()
            return


class EnvWorkers:
    def __init__(self, cfg: dict, num_workers: int, base_seed: int):
        self.n = max(1, num_workers)
        self.cfg = cfg
        self._view_rng = np.random.default_rng(base_seed + 7)
        hz = float(cfg.get("control_hz", 100))
        self._switch_every = max(1, int(hz * 8.0))
        self._since_switch = 0
        self.local_env = SwarmEnv(cfg, test_id=int(self._view_rng.integers(1, 4)), seed=base_seed)
        self.procs: list[Any] = []
        self.conns: list[Any] = []
        for i in range(1, self.n):
            parent, child = _ctx.Pipe()
            proc = _ctx.Process(target=_worker, args=(child, cfg, (i % 3) + 1, base_seed + i), daemon=True)
            proc.start()
            child.close()
            self.procs.append(proc)
            self.conns.append(parent)
        extra = [conn.recv() for conn in self.conns]
        self._obs = [self._reroll_local()] + extra

    def _reroll_local(self) -> np.ndarray:
        tid = int(self._view_rng.integers(1, 4))
        obs = self.local_env.set_test_id(tid)
        if tid == 2:
            _shuffle_geometry(self.local_env)
            obs = self.local_env.reset()
        return obs

    def obs_batch(self) -> np.ndarray:
        return np.stack(self._obs, axis=0)

    def step(self, actions: np.ndarray):
        for i, conn in enumerate(self.conns):
            conn.send(("step", actions[i + 1]))
        r0 = self.local_env.step(actions[0])
        rest = [conn.recv() for conn in self.conns]
        obs = [r0.obs] + [x[0] for x in rest]
        rew = [r0.rewards] + [x[1] for x in rest]
        done = [r0.done] + [x[2] for x in rest]
        infos = [r0.info] + [x[3] for x in rest]
        self._since_switch += 1
        if done[0] or self._since_switch >= self._switch_every:
            obs[0] = self._reroll_local()
            done[0] = True
            infos[0] = {**infos[0], "test_id": self.local_env.test_id}
            self._since_switch = 0
        for i, finished in enumerate(done):
            if not finished or i == 0:
                continue
            self.conns[i - 1].send(("reset", None))
            obs[i] = self.conns[i - 1].recv()
        self._obs = obs
        return np.stack(obs), np.stack(rew), np.array(done, dtype=bool), infos, self.local_env.snapshot()

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()


def resolve_workers(cfg: dict) -> int:
    raw = cfg.get("num_workers", "auto")
    if raw == "auto" or raw is None:
        return max(1, os.cpu_count() or 1)
    return max(1, int(raw))
