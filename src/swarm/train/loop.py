from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from swarm.baseline.pid import GeometricPid
from swarm.config import repo_root
from swarm.env.swarm_env import SwarmEnv
from swarm.gates.pass_gate import combined_score, evaluate_episode, ring_error, standoff_cm
from swarm.policy.actor_critic import ActorCritic
from swarm.train.workers import EnvWorkers, resolve_workers

log = logging.getLogger("swarm.train")


def fmt_dur(sec: float) -> str:
    if sec != sec or sec < 0 or sec == float("inf"):
        return "—"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def cli_bar(step: int, total: int, extra: str = "") -> None:
    if total <= 0:
        msg = f"\r[{step} steps] {extra}"
    else:
        total = max(int(total), 1)
        step_c = min(max(int(step), 0), total)
        pct = 100.0 * step_c / total
        width = 28
        filled = int(width * step_c / total)
        bar = "#" * filled + "-" * (width - filled)
        msg = f"\r[{bar}] {step_c}/{total} {pct:5.1f}% {extra}"
    sys.stderr.write(msg[:160].ljust(160))
    sys.stderr.flush()
    if total > 0 and step >= total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        log.error("cuda requested but not available; using cpu")
        return torch.device("cpu")
    return torch.device(name)


def flatten_obs(batch: np.ndarray) -> np.ndarray:
    w, n, d = batch.shape
    return batch.reshape(w * n, d)


def _expand_actions(flat: np.ndarray, workers: int, n_drones: int) -> np.ndarray:
    return flat.reshape(workers, n_drones, 4)


def _tb(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(str(log_dir))
    except Exception as exc:
        log.info("tensorboard off: %s", exc)
        return None


@torch.no_grad()
def eval_episodes(cfg: dict, policy: ActorCritic | None, device: torch.device, seed: int) -> list[dict]:
    packs = []
    for test_id in (1, 2, 3):
        env = SwarmEnv(cfg, test_id=test_id, seed=seed, eval_mode=True)
        baseline = GeometricPid.from_cfg(cfg, test_id=test_id)
        obs = env.reset()
        done = False
        while not done:
            if policy is None:
                act = baseline.step(obs)
            else:
                t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                act, _, _ = policy.act(t, deterministic=True)
                act = act.cpu().numpy()
            step = env.step(act)
            obs, done = step.obs, step.done
        row = evaluate_episode(env.history, cfg, test_id, env.collided, env.out_of_bounds)
        row["test_id"] = test_id
        packs.append(
            {
                "gate": row,
                "history": env.history,
                "snap": env.snapshot(),
                "test_id": test_id,
                "collided": env.collided,
                "out_of_bounds": env.out_of_bounds,
                "floor_hit": env.floor_hit,
            }
        )
    return packs


@torch.no_grad()
def eval_policy(cfg: dict, policy: ActorCritic | None, device: torch.device, seed: int) -> list[dict]:
    return [p["gate"] for p in eval_episodes(cfg, policy, device, seed)]


def save_ckpt(path: Path, policy: ActorCritic, cfg: dict, step: int, opt=None, best_score: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"state_dict": policy.state_dict(), "cfg": cfg, "step": step}
    if opt is not None:
        blob["opt"] = opt.state_dict()
    if best_score is not None:
        blob["best_score"] = best_score
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


def load_policy(path: Path, neighbor_k: int, hidden: int, device: torch.device) -> ActorCritic:
    if not path.is_file():
        raise RuntimeError(f"policy file missing: {path}")
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except OSError as exc:
        raise RuntimeError(f"cannot load {path}: {exc}") from exc
    policy = ActorCritic(neighbor_k, hidden).to(device)
    missing = policy.load_state_dict(blob["state_dict"], strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        log.info("checkpoint keys missing=%s unexpected=%s", missing.missing_keys, missing.unexpected_keys)
    policy.eval()
    log.info("loaded %s step=%s score=%s", path.name, blob.get("step"), blob.get("best_score"))
    return policy


def _metrics_path(seed: int) -> Path:
    path = repo_root() / "runs" / f"seed{seed}" / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_metrics(path: Path, row: dict) -> None:
    keys = [
        "step",
        "reward",
        "collisions",
        "score",
        "t1_err",
        "t2_err",
        "t3_err",
        "t1_pass",
        "t2_pass",
        "t3_pass",
    ]
    new = not path.is_file()
    try:
        with path.open("a", encoding="utf-8") as f:
            if new:
                f.write(",".join(keys) + "\n")
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    except OSError as exc:
        log.error("cannot write metrics: %s", exc)


def _log_train_eval(cfg: dict, seed: int, step: int, packs: list[dict]) -> None:
    from swarm.report import write_run_log

    try:
        write_run_log(
            cfg=cfg,
            cmd="train",
            mode="train",
            policy_name="rl",
            seed=seed,
            extra={"step": step},
            episodes=[
                {
                    "test_id": p["test_id"],
                    "history": p["history"],
                    "collided": p.get("collided"),
                    "out_of_bounds": p.get("out_of_bounds"),
                    "floor_hit": p.get("floor_hit"),
                }
                for p in packs
            ],
        )
    except Exception as exc:
        log.error("run log failed: %s", exc)


def _probe_test(
    cfg: dict,
    policy: ActorCritic,
    device: torch.device,
    test_id: int,
    kind: str | None = None,
) -> tuple[float, bool]:
    """Full eval-length probe. Seed 0 matches make view-train."""
    policy.eval()
    env = SwarmEnv(cfg, test_id=test_id, seed=0, eval_mode=True)
    if test_id == 2 and kind:
        env.set_geometry_kind(kind)
    obs = env.reset()
    n = int(float(cfg["episode_s"]) * float(cfg["control_hz"]))
    try:
        for _ in range(n):
            t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                act, _, _ = policy.act(t, deterministic=True)
            obs = env.step(act.cpu().numpy()).obs
        pos = env.positions()
        if test_id in (1, 3):
            dist = float(ring_error(pos, env.task_state.shared_target, standoff_cm(cfg)).mean())
        else:
            dist = float(np.linalg.norm(pos - env.task_state.aim_points, axis=1).mean())
        return dist, bool(env.collided or env.out_of_bounds)
    finally:
        policy.train()


def _probe_t2(cfg: dict, policy: ActorCritic, device: torch.device) -> tuple[float, bool]:
    """Worst of square and circle — view-train on C was a wall while square looked ok."""
    d0, h0 = _probe_test(cfg, policy, device, 2, kind="square")
    d1, h1 = _probe_test(cfg, policy, device, 2, kind="circle")
    return max(d0, d1), bool(h0 or h1)


def _pack_holds(t1: float, hit1: bool, t2: float, hit2: bool, keep_t1: float, keep_t2: float) -> bool:
    if hit1 or hit2:
        return False
    if t1 > keep_t1 + 2.0:
        return False
    if t2 > keep_t2 + 2.0:
        return False
    return True


def warmstart_pid(cfg: dict, policy: ActorCritic, opt, device: torch.device, ticks: int = 12000) -> float:
    """Clone GeometricPid flight cmds on-policy so spawn errors get corrected, not only hover."""
    policy.train()
    clone_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    envs = []
    pids = []
    last = float("nan")
    try:
        kind_i = 0
        for i, tid in enumerate((1, 1, 2, 2, 2, 2, 3, 3)):
            env = SwarmEnv(cfg, test_id=tid, seed=10_000 + i * 17)
            if tid == 2:
                env.set_geometry_kind(("square", "rectangle", "circle", "square")[kind_i])
                kind_i += 1
                env.reset()
            envs.append(env)
            pids.append(GeometricPid.from_cfg(cfg, test_id=tid))
        n = int(cfg["num_drones"])
        for t in range(max(1, ticks)):
            obs_rows = []
            act_rows = []
            for env, pid in zip(envs, pids):
                obs = env._obs()
                obs_rows.append(obs)
                act_rows.append(pid.step(obs))
            obs_t = torch.as_tensor(np.concatenate(obs_rows), dtype=torch.float32, device=device)
            act_t = torch.as_tensor(np.concatenate(act_rows), dtype=torch.float32, device=device)
            mean, _, _ = policy.forward(obs_t)
            pred = torch.tanh(mean)
            loss = (pred - act_t).pow(2).mean()
            clone_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            clone_opt.step()
            rolled = pred.detach().cpu().numpy()
            pid_frac = max(0.2, 1.0 - t / max(ticks * 0.5, 1))
            for i, env in enumerate(envs):
                chunk = rolled[i * n : (i + 1) * n]
                use_pid = (i + t) % 5 == 0 or (t / max(ticks, 1)) < 0.2
                if t > ticks * 0.5:
                    use_pid = (i + t) % 8 == 0
                step_act = act_rows[i] if use_pid else chunk
                if env.step(step_act).done:
                    env.reset()
            last = float(loss.detach().cpu())
            if t % 500 == 0 or t + 1 == ticks:
                cli_bar(t + 1, ticks, f"pid-clone mse={last:.4f} pid={pid_frac:.2f}")
            if t > 0 and t % 4000 == 3999:
                t1, h1 = _probe_test(cfg, policy, device, 1)
                t2, h2 = _probe_t2(cfg, policy, device)
                log.info("clone probe t=%s t1=%.3f hit=%s t2=%.3f hit=%s", t + 1, t1, h1, t2, h2)
                if t1 < 55.0 and not h1 and t2 < 80.0 and not h2:
                    sys.stderr.write("\n")
                    return last
        sys.stderr.write("\n")
        return last
    except Exception as exc:
        log.error("pid warmstart failed: %s", exc)
        raise


def warmstart_critic(cfg: dict, policy: ActorCritic, device: torch.device, ticks: int = 4000) -> float:
    """Fit V on the frozen clone so the first PPO advantages are not noise."""
    policy.train()
    opt = torch.optim.Adam(policy.critic.parameters(), lr=1e-3)
    gamma = float(cfg["gamma"])
    env = SwarmEnv(cfg, test_id=1, seed=21, eval_mode=True)
    obs = env.reset()
    hs: list[torch.Tensor] = []
    rews: list[np.ndarray] = []
    last = float("nan")
    try:
        for t in range(max(1, ticks)):
            ot = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                act, _, _ = policy.act(ot, deterministic=True)
                hs.append(policy.features(ot).cpu())
            step = env.step(act.cpu().numpy())
            rews.append(np.asarray(step.rewards, dtype=np.float32))
            obs = step.obs
            ended = bool(step.done) or t + 1 == ticks
            if not ended:
                continue
            r = np.stack(rews)
            ret = np.zeros_like(r)
            running = np.zeros(r.shape[1], dtype=np.float32)
            for i in reversed(range(len(r))):
                running = r[i] + gamma * running
                ret[i] = running
            h_t = torch.cat(hs, dim=0).to(device)
            y = torch.as_tensor(ret.reshape(-1), dtype=torch.float32, device=device)
            pred = policy.critic(h_t).squeeze(-1)
            loss = (pred - y).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.detach().cpu())
            hs, rews = [], []
            if step.done:
                obs = env.reset()
            if t % 500 == 0 or t + 1 == ticks:
                cli_bar(t + 1, ticks, f"critic-fit mse={last:.4f}")
        sys.stderr.write("\n")
        return last
    except Exception as exc:
        log.error("critic warmstart failed: %s", exc)
        raise


def _clone_state(policy: ActorCritic) -> dict:
    return {k: v.detach().clone() for k, v in policy.state_dict().items()}


def train(cfg: dict, seed: int, view: str, resume: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    seed_all(seed)
    device = resolve_device(str(cfg.get("device", "auto")))
    workers_n = resolve_workers(cfg)
    n_drones = int(cfg["num_drones"])
    k = int(cfg["neighbor_k"])
    hidden = int(cfg.get("hidden", 128))
    policy = ActorCritic(k, hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=float(cfg["lr"]))
    models_dir = repo_root() / str(cfg.get("models_dir", "models"))
    latest = models_dir / "latest.pt"
    best_path = models_dir / "best.pt"
    best_score = -1e9
    steps_done = 0
    if resume:
        if not latest.is_file():
            raise RuntimeError(f"nothing to resume: {latest} missing")
        blob = torch.load(latest, map_location=device, weights_only=False)
        missing = policy.load_state_dict(blob["state_dict"], strict=False)
        if missing.missing_keys or missing.unexpected_keys:
            log.info("resume keys missing=%s unexpected=%s", missing.missing_keys, missing.unexpected_keys)
        if "opt" in blob:
            opt.load_state_dict(blob["opt"])
        steps_done = int(blob.get("step", 0))
        best_score = float(blob.get("best_score", best_score))
        log.info("resumed step=%s best=%.3f from %s", steps_done, best_score, latest)
    else:
        mse = warmstart_pid(cfg, policy, opt, device)
        log.info("pid warmstart mse=%.4f — first view should hover before PPO", mse)
        vmse = warmstart_critic(cfg, policy, device)
        log.info("critic warmstart mse=%.4f", vmse)
        save_ckpt(latest, policy, cfg, steps_done, opt=opt, best_score=best_score)
    log.info("device=%s workers=%s view=%s resume=%s", device, workers_n, view, resume)
    workers = EnvWorkers(cfg, workers_n, seed)
    viewer = None
    if view in {"python", "gazebo", "1"}:
        from swarm.viewer import make_viewer

        kind = "python" if view == "1" else view
        viewer = make_viewer(kind, cfg)
        if viewer is None:
            log.error("viewer %s unavailable; training headless", kind)
    run_dir = repo_root() / str(cfg.get("log_dir", "runs")) / f"seed{seed}"
    writer = _tb(run_dir)
    metrics_csv = _metrics_path(seed)
    last_eval = {"score": float("nan"), "t1_err": float("nan"), "t2_err": float("nan"), "t3_err": float("nan")}
    packs = eval_episodes(cfg, policy, device, seed)
    results = [p["gate"] for p in packs]
    score = combined_score(results)
    last_eval = {
        "score": score,
        "t1_err": float(results[0].get("error") or 0.0),
        "t2_err": float(results[1].get("error") or 0.0),
        "t3_err": float(results[2].get("error") or 0.0),
    }
    if not resume:
        best_score = score
        save_ckpt(best_path, policy, cfg, steps_done, opt=opt, best_score=best_score)
        save_ckpt(latest, policy, cfg, steps_done, opt=opt, best_score=best_score)
    log.info(
        "eval score=%.3f t1=%.3f t2=%.3f t3=%.3f tests=%s",
        score,
        last_eval["t1_err"],
        last_eval["t2_err"],
        last_eval["t3_err"],
        [(r["passed"], r.get("reason"), r.get("error")) for r in results],
    )
    _log_train_eval(cfg, seed, steps_done, packs)
    gamma = float(cfg["gamma"])
    lam = float(cfg["gae_lambda"])
    horizon = int(cfg["horizon"])
    total = int(cfg["total_steps"])
    eval_every = int(cfg["eval_every"])
    run_loop = True
    if total > 0 and total < horizon * max(workers_n, 1):
        log.info("total_steps=%s is less than one PPO batch; keeping the cloned net", total)
        run_loop = False
    keep_state = _clone_state(policy)
    keep_t1 = float(last_eval["t1_err"])
    keep_t2 = float(last_eval["t2_err"])
    critic_only = 8 if not resume else 0
    obs_w = workers.obs_batch()
    t0 = time.monotonic()
    start_steps = steps_done
    last_save = "—"
    log.info("Ctrl+C stops; latest.pt is only written when Test 1 and Test 2 do not get worse")
    log.info("train window cycles Test 1/2/3 every 8 s (noisy). After stop: make view-train TEST=2")
    if total <= 0:
        log.info("no step cap (hours/days ok). set total_steps in train.yaml to cap")
    try:
        while run_loop and (total <= 0 or steps_done < total):
            mb_obs, mb_act, mb_logp, mb_rew, mb_val, mb_done = [], [], [], [], [], []
            collisions = 0
            for _ in range(horizon):
                flat = flatten_obs(obs_w)
                t_obs = torch.as_tensor(flat, dtype=torch.float32, device=device)
                with torch.no_grad():
                    act, logp, val = policy.act(t_obs, deterministic=critic_only > 0)
                env_act = _expand_actions(act.cpu().numpy(), obs_w.shape[0], n_drones)
                next_obs, rew, done, infos, snap = workers.step(env_act)
                collisions += sum(1 for inf in infos if inf.get("collided"))
                elapsed = time.monotonic() - t0
                rate = (steps_done - start_steps) / max(elapsed, 1e-6)
                remain = (total - steps_done) / rate if total > 0 and rate > 0 else float("nan")
                extra = f"rew={float(np.mean(rew)):.2f} {fmt_dur(elapsed)} eta={fmt_dur(remain) if total > 0 else 'Ctrl+C'}"
                if viewer is not None:
                    try:
                        viewer.draw(
                            snap,
                            {
                                "step": steps_done,
                                "total": total,
                                "reward": float(np.mean(rew)),
                                "score": last_eval["score"],
                                "t1_err": last_eval["t1_err"],
                                "t2_err": last_eval["t2_err"],
                                "t3_err": last_eval["t3_err"],
                                "elapsed": fmt_dur(elapsed),
                                "eta": fmt_dur(remain) if total > 0 else "until Ctrl+C",
                                "rate": f"{rate:.0f}/s",
                                "saved": last_save,
                                "collisions": collisions,
                            },
                        )
                    except Exception as exc:
                        log.error("viewer draw failed: %s", exc)
                        viewer = None
                mb_obs.append(flat)
                mb_act.append(act.cpu().numpy())
                mb_logp.append(logp.cpu().numpy())
                mb_rew.append(rew.reshape(-1))
                mb_val.append(val.cpu().numpy())
                mb_done.append(np.repeat(done, n_drones))
                obs_w = next_obs
                steps_done += workers_n
                if steps_done % (workers_n * 8) == 0:
                    cli_bar(steps_done, total, extra)
            with torch.no_grad():
                last_val = (
                    policy.forward(torch.as_tensor(flatten_obs(obs_w), dtype=torch.float32, device=device))[2]
                    .cpu()
                    .numpy()
                )
            _ppo_update(
                policy,
                opt,
                cfg,
                device,
                mb_obs,
                mb_act,
                mb_logp,
                mb_rew,
                mb_val,
                mb_done,
                last_val,
                gamma,
                lam,
                actor=critic_only <= 0,
            )
            if critic_only > 0:
                critic_only -= 1
            else:
                t1, h1 = _probe_test(cfg, policy, device, 1)
                t2, h2 = _probe_t2(cfg, policy, device)
                if _pack_holds(t1, h1, t2, h2, keep_t1, keep_t2):
                    keep_t1 = min(keep_t1, float(t1))
                    keep_t2 = min(keep_t2, float(t2))
                    keep_state = _clone_state(policy)
                    last_eval["t1_err"] = keep_t1
                    last_eval["t2_err"] = keep_t2
                    save_ckpt(latest, policy, cfg, steps_done, opt=opt, best_score=best_score)
                    last_save = f"latest.pt @ {fmt_dur(time.monotonic() - t0)}"
                    log.info("kept PPO pack t1=%.3f t2=%.3f cm", keep_t1, keep_t2)
                else:
                    policy.load_state_dict(keep_state)
                    log.info(
                        "rejected PPO pack t1=%.3f t2=%.3f (keep t1=%.3f t2=%.3f cm)",
                        t1,
                        t2,
                        keep_t1,
                        keep_t2,
                    )
            mean_rew = float(np.mean(mb_rew))
            row = {
                "step": steps_done,
                "reward": mean_rew,
                "collisions": collisions,
                "score": last_eval["score"],
                "t1_err": last_eval["t1_err"],
                "t2_err": last_eval["t2_err"],
                "t3_err": last_eval["t3_err"],
                "t1_pass": "",
                "t2_pass": "",
                "t3_pass": "",
            }
            if writer is not None:
                writer.add_scalar("train/reward", mean_rew, steps_done)
                writer.add_scalar("train/collisions", collisions, steps_done)
                writer.flush()
            if steps_done % eval_every < workers_n * horizon:
                packs = eval_episodes(cfg, policy, device, seed)
                results = [p["gate"] for p in packs]
                score = combined_score(results)
                last_eval = {
                    "score": score,
                    "t1_err": float(results[0].get("error") or 0.0),
                    "t2_err": float(results[1].get("error") or 0.0),
                    "t3_err": float(results[2].get("error") or 0.0),
                }
                row.update(
                    {
                        "score": score,
                        "t1_err": last_eval["t1_err"],
                        "t2_err": last_eval["t2_err"],
                        "t3_err": last_eval["t3_err"],
                        "t1_pass": int(results[0]["passed"]),
                        "t2_pass": int(results[1]["passed"]),
                        "t3_pass": int(results[2]["passed"]),
                    }
                )
                sys.stderr.write("\n")
                log.info(
                    "step=%s reward=%.3f score=%.3f t1=%.3f t2=%.3f t3=%.3f tests=%s",
                    steps_done,
                    mean_rew,
                    score,
                    last_eval["t1_err"],
                    last_eval["t2_err"],
                    last_eval["t3_err"],
                    [(r["passed"], r.get("reason"), r.get("error")) for r in results],
                )
                _log_train_eval(cfg, seed, steps_done, packs)
                if writer is not None:
                    writer.add_scalar("eval/combined", score, steps_done)
                    for r in results:
                        writer.add_scalar(f"eval/test{r.get('test_id', 0)}_err", float(r.get("error") or 0.0), steps_done)
                    writer.flush()
                t1_hit = results[0].get("reason") == "collision_or_bounds"
                t2_hit = results[1].get("reason") == "collision_or_bounds"
                if not _pack_holds(
                    last_eval["t1_err"],
                    t1_hit,
                    last_eval["t2_err"],
                    t2_hit,
                    keep_t1,
                    keep_t2,
                ):
                    policy.load_state_dict(keep_state)
                    last_eval["t1_err"] = keep_t1
                    last_eval["t2_err"] = keep_t2
                    log.info("rejected eval; restored keep t1=%.3f t2=%.3f cm", keep_t1, keep_t2)
                else:
                    keep_t1 = min(keep_t1, last_eval["t1_err"])
                    keep_t2 = min(keep_t2, last_eval["t2_err"])
                    keep_state = _clone_state(policy)
                    if score > best_score:
                        best_score = score
                        save_ckpt(best_path, policy, cfg, steps_done, opt=opt, best_score=best_score)
                        last_save = f"best.pt @ {fmt_dur(time.monotonic() - t0)}"
                        log.info("wrote %s", best_path)
                    save_ckpt(latest, policy, cfg, steps_done, opt=opt, best_score=best_score)
            _append_metrics(metrics_csv, row)
            elapsed = time.monotonic() - t0
            rate = (steps_done - start_steps) / max(elapsed, 1e-6)
            remain = (total - steps_done) / rate if total > 0 and rate > 0 else float("nan")
            cli_bar(
                steps_done,
                total,
                f"rew={mean_rew:.2f} score={last_eval['score']:.3f} t1={last_eval['t1_err']:.2f} "
                f"t2={last_eval['t2_err']:.2f} {fmt_dur(elapsed)} eta={fmt_dur(remain) if total > 0 else 'Ctrl+C'}",
            )
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        try:
            policy.load_state_dict(keep_state)
        except Exception:
            pass
        log.info("stopped at step=%s — saved keep t1=%.3f t2=%.3f cm", steps_done, keep_t1, keep_t2)
    finally:
        try:
            save_ckpt(latest, policy, cfg, steps_done, opt=opt, best_score=best_score)
        except Exception as exc:
            log.error("final save failed: %s", exc)
        try:
            from swarm.report import dump_pack

            packs = eval_episodes(cfg, policy, device, seed)
            dump_pack(
                cfg=cfg,
                mode="train",
                policy_name="rl",
                ckpt=latest,
                seed=seed,
                gates=[p["gate"] for p in packs],
                histories={p["test_id"]: p["history"] for p in packs},
                snaps={p["test_id"]: p["snap"] for p in packs},
                extra={"step": steps_done, "best_score": best_score},
            )
        except Exception as exc:
            log.error("results dump failed: %s", exc)
        sys.stderr.write("\n")
        workers.close()
        if viewer is not None:
            viewer.close()
        if writer is not None:
            writer.close()


def _ppo_update(
    policy, opt, cfg, device, mb_obs, mb_act, mb_logp, mb_rew, mb_val, mb_done, last_val, gamma, lam, actor: bool = True
) -> None:
    rew = np.stack(mb_rew)
    val = np.stack(mb_val)
    done = np.stack(mb_done).astype(np.float32)
    adv = np.zeros_like(rew, dtype=np.float32)
    last = np.zeros(rew.shape[1], dtype=np.float32)
    for t in reversed(range(rew.shape[0])):
        nxt = last_val if t == rew.shape[0] - 1 else val[t + 1]
        mask = 1.0 - done[t]
        delta = rew[t] + gamma * nxt * mask - val[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    ret = adv + val
    t_len, batch = rew.shape
    obs = torch.as_tensor(np.stack(mb_obs).reshape(t_len * batch, -1), dtype=torch.float32, device=device)
    act = torch.as_tensor(np.stack(mb_act).reshape(t_len * batch, -1), dtype=torch.float32, device=device)
    old_logp = torch.as_tensor(np.stack(mb_logp).reshape(t_len * batch), dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(adv.reshape(-1), dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(ret.reshape(-1), dtype=torch.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    clip = float(cfg["clip"])
    n = obs.shape[0]
    mb = max(1, min(int(cfg.get("minibatch", n)), n))
    for _ in range(int(cfg["ppo_epochs"])):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, mb):
            idx = perm[start : start + mb]
            logp, entropy, value = policy.evaluate(obs[idx], act[idx])
            vloss = (value - ret_t[idx]).pow(2).mean()
            if actor:
                ratio = (logp - old_logp[idx]).exp()
                surr = torch.min(ratio * adv_t[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[idx])
                loss = (
                    -surr.mean()
                    + float(cfg["value_coef"]) * vloss
                    - float(cfg["entropy_coef"]) * entropy.mean()
                )
            else:
                loss = float(cfg["value_coef"]) * vloss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
