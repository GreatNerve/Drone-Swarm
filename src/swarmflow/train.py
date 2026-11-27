import copy
import json
import math
import os
import shutil
import signal
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .baseline import PredictiveController
from .config import Config
from .env import SwarmEnv
from .evaluate import aggregate, checkpoint_score, min_obstacle_clearance, run_episode, selection_score
from .files import (append_jsonl, dumps, load_torch, portable_path, read_json, read_jsonl, save_torch_atomic,
                    sanitize, write_json_atomic, write_text_atomic)
from .policy import (ACTOR_FORMAT, CHECKPOINT_FORMAT, CHECKPOINT_FORMATS, OBSERVATION_KEYS, ActorCritic,
                     ModelController, stack_observations, stack_teams, tensors)

# Episode seeds: training rollouts and cloning draw from 0..999, checkpoint selection uses 5000+,
# development validation uses 1000+ and the final held-out test uses 100000+.
TRAINING_SEEDS = (0, 1000)
SELECTION_SEED_START = 5000
HARD_OBJECT_GAP_M = 2.0
HARD_NEIGHBOR_GAP_M = 0.6
TEACHER_DATA_DIR = "teacher-data"
MAX_BAD_UPDATES = 3


def advantages(rewards, values, next_values, terminated, ended, gamma=0.99, lam=0.95):
    """Bootstrap time limits, but never propagate advantage across an episode reset."""
    result = np.zeros_like(rewards)
    carry = np.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
        carry = delta + gamma * lam * (1 - ended[t]) * carry
        result[t] = carry
    return result, result + values


class RewardScaler:
    """Divides rewards by a running standard deviation of the discounted return."""

    def __init__(self, gamma, count=1e-4, mean=0.0, var=1.0):
        self.gamma = gamma
        self.count, self.mean, self.var = float(count), float(mean), float(var)
        self.returns = None

    def __call__(self, rewards, ended):
        if self.returns is None or self.returns.shape != rewards.shape:
            self.returns = np.zeros_like(rewards)
        self.returns = self.gamma * self.returns + rewards
        self._update(self.returns)
        self.returns = self.returns * (1.0 - ended)
        return rewards / math.sqrt(self.var + 1e-8)

    def _update(self, values):
        mean, var, n = float(values.mean()), float(values.var()), values.size
        delta, total = mean - self.mean, self.count + n
        self.var = (self.var * self.count + var * n + delta ** 2 * self.count * n / total) / total
        self.mean += delta * n / total
        self.count = total

    def state(self):
        return {"count": self.count, "mean": self.mean, "var": self.var}


class TrainingStopped(KeyboardInterrupt):
    """Raised after a requested stop once latest.pt has been saved; callers treat it like Ctrl+C."""


class StopRequest:
    """The first Ctrl+C (Ctrl+Break on Windows) asks for a stop at the next safe point; a second one stops now."""

    def __init__(self):
        self.requested = False
        self._previous = {}

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGINT", "SIGBREAK"):
                number = getattr(signal, name, None)
                if number is None:
                    continue
                try:
                    self._previous[number] = signal.signal(number, self._handle)
                except (ValueError, OSError):
                    pass
        return self

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\nStopping after the current update or cloning round, then saving. "
              "Press Ctrl+C again to stop at once (the last saved update is kept).", flush=True)

    def __exit__(self, *exc):
        for number, handler in self._previous.items():
            try:
                signal.signal(number, handler)
            except (ValueError, OSError):
                pass
        return False


class Worker:
    """One environment that starts a new random training episode whenever the current one ends."""

    def __init__(self, config, scenarios, rng, failure_reset):
        self.config, self.scenarios, self.rng, self.failure_reset = config, scenarios, rng, failure_reset
        self.interval = max(1, int(2.0 / config.dt))
        self.ring = deque(maxlen=4)
        self.restores = 0
        self.start_episode()

    def start_episode(self):
        self.env = SwarmEnv(self.config, str(self.rng.choice(self.scenarios)))
        self.observations, _ = self.env.reset(seed=int(self.rng.integers(*TRAINING_SEEDS)))
        self.restores = 0
        self.ring.clear()

    def step(self, actions):
        env = self.env
        final, reward, term, trunc, _ = env.step(dict(zip(self.observations, actions)))
        agents = list(reward)
        result = {"final": final, "done": not env.agents, "collision": bool(env.collision),
                  "restored": False, "summary": None,
                  "reward": np.array([reward[a] for a in agents], dtype=np.float32),
                  "terminated": np.array([term[a] for a in agents], dtype=np.float32),
                  "ended": np.array([term[a] or trunc[a] for a in agents], dtype=np.float32)}
        if env.agents:
            if self.failure_reset and env.steps % self.interval == 0:
                self.ring.append(env.capture_state())
            self.observations = final
        elif self.failure_reset and env.collision and self.ring and self.restores < 2:
            env.restore_state(self.ring[0])
            self.observations = env.last_observations
            self.restores += 1
            self.ring.clear()
            result["restored"] = True
        else:
            result["summary"] = env.summary()
            # A replayed crash still counts: the episode failed before it was given another try.
            result["failed"] = env.failure is not None or self.restores > 0
            self.start_episode()
        return result


def _hard_share(observation, radius):
    """Per labelled state: True when an object is within 2 m or a neighbour within 0.6 m (surface gap)."""
    near = torch.zeros(len(observation["self"]), dtype=torch.bool, device=observation["self"].device)
    for key, limit in (("obstacles", HARD_OBJECT_GAP_M), ("neighbors", HARD_NEIGHBOR_GAP_M)):
        rows = observation[key]
        gap = rows[..., :3].norm(dim=-1) - rows[..., 6] - radius
        near |= ((rows[..., 7] > 0.5) & (gap < limit)).any(-1)
    return near.cpu().numpy()


def _demo_tensors(data, labels, device, radius, hard_weight):
    """Teacher dataset (observations, actions, sampling CDF or None, share of hard states)."""
    observation = tensors({k: np.concatenate(v) for k, v in data.items()}, device)
    target = torch.as_tensor(np.concatenate(labels), dtype=torch.float32, device=device)
    near = _hard_share(observation, radius)
    cdf = None
    if hard_weight != 1.0:
        weights = 1.0 + (hard_weight - 1.0) * near
        cdf = np.cumsum(weights) / weights.sum()
    return observation, target, cdf, float(near.mean())


def _demo_ids(demo, count, rng):
    """Sample demonstration indices; near-obstacle states are drawn hard_weight times more often."""
    cdf = demo[2]
    if cdf is None:
        return rng.integers(0, len(demo[1]), count)
    return np.minimum(np.searchsorted(cdf, rng.random(count)), len(cdf) - 1)


def _fit_clone(model, optimizer, demo, epochs, rng, device, batch_size=1024):
    batch, target = demo[0], demo[1]
    parameters = model.actor_parameters()
    losses = []
    for _ in range(epochs):
        losses = []
        order = rng.permutation(len(target)) if demo[2] is None else _demo_ids(demo, len(target), rng)
        for start in range(0, len(target), batch_size):
            ids = torch.as_tensor(order[start:start + batch_size], device=device)
            loss = F.mse_loss(model.act({k: v[ids] for k, v in batch.items()}), target[ids])
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _collect(model, teacher, config, scenarios, steps, rng, device, data, labels, noise=None):
    """Fly `steps` team steps and record every state with the teacher's action as its label.

    With `noise` the teacher flies with that much action noise (DART); otherwise the student
    flies (DAgger). Returns (episodes finished, failed episodes).
    """
    student = ModelController(model, device)
    env = None
    episodes = failures = 0
    for _ in range(steps):
        if env is None or not env.agents:
            env = SwarmEnv(config, str(rng.choice(scenarios)))
            observations, _ = env.reset(seed=int(rng.integers(*TRAINING_SEEDS)))
        targets = teacher(observations)
        packed = stack_observations(observations)
        for key in data:
            data[key].append(packed[key])
        labels.append(np.stack(list(targets.values())))
        if noise is not None:
            actions = {a: np.clip(v + rng.normal(0, noise, 3), -1, 1) for a, v in targets.items()}
        else:
            actions = student(observations)
        observations, _, _, _, _ = env.step(actions)
        if not env.agents:
            episodes += 1
            failures += int(env.failure is not None)
    return episodes, failures


def clone(model, config, scenarios, steps, epochs, rng, device, rounds=0, noise=0.3, report=None,
          teacher_margins=(0.35, 0.08), hard_weight=1.0):
    """Behavior cloning from the predictive controller, followed by DAgger rounds.

    Round 0 flies the teacher with injected action noise while recording its clean actions, so the
    data already contains states away from the slots. Every later round flies the current student,
    labels each state it visits with the teacher's action and refits on all data collected so far.
    Returns the log rows and the aggregated teacher dataset. (train() runs the same rounds itself so
    it can checkpoint after each one.)
    """
    teacher = PredictiveController(config, *teacher_margins)
    optimizer = torch.optim.Adam(model.actor_parameters(), lr=5e-4)
    data = {key: [] for key in OBSERVATION_KEYS}
    labels = []
    rows = []
    demo = None
    for round_index in range(rounds + 1):
        episodes, failures = _collect(model, teacher, config, scenarios, steps, rng, device, data, labels,
                                      noise if round_index == 0 else None)
        demo = _demo_tensors(data, labels, device, config.radius, hard_weight)
        mse = _fit_clone(model, optimizer, demo, epochs, rng, device)
        rows.append(_clone_row(round_index, steps, epochs, demo, episodes, failures, mse))
        if report:
            report(rows[-1])
    return rows, demo


def _clone_row(round_index, steps, epochs, demo, episodes, failures, mse):
    return {"type": "behavior_cloning", "round": round_index,
            "driver": "teacher with action noise" if round_index == 0 else "student (DAgger)",
            "environment_steps": steps, "agent_samples": len(demo[1]), "epochs": epochs,
            "driver_episodes_finished": episodes, "driver_failures": failures,
            "hard_state_share": demo[3], "final_epoch_mse": mse}


def demonstrations(model, config, scenarios, steps, rng, device, teacher_margins=(0.35, 0.08), hard_weight=1.0):
    """Teacher-labelled states visited by the current student, without refitting (fallback for old runs)."""
    data = {key: [] for key in OBSERVATION_KEYS}
    labels = []
    _collect(model, PredictiveController(config, *teacher_margins), config, scenarios, steps, rng, device,
             data, labels)
    return _demo_tensors(data, labels, device, config.radius, hard_weight)


def _save_teacher_round(folder, round_index, data, labels, start):
    chunk = {"observations": {k: torch.from_numpy(np.concatenate(v[start:])) for k, v in data.items()},
             "labels": torch.from_numpy(np.concatenate(labels[start:]))}
    save_torch_atomic(chunk, Path(folder) / f"round{round_index}.pt")


def _load_teacher_rounds(folder, rounds, data, labels):
    """Append saved cloning rounds 0..rounds-1 to data/labels; False if any round file is missing."""
    chunks = []
    for index in range(rounds):
        path = Path(folder) / f"round{index}.pt"
        if not path.exists():
            return False
        chunks.append(load_torch(path))
    for chunk in chunks:
        for key in data:
            data[key].append(chunk["observations"][key].numpy())
        labels.append(chunk["labels"].numpy())
    return True


def _trim_logs(folder, steps, progress):
    """Drop log rows written after latest.pt was saved (a hard kill between the two), so a continued run
    never logs the same cloning round, update or validation twice."""
    cloning = progress.get("phase") == "cloning"
    rounds = int(progress.get("clone_rounds_done", 0))
    unsaved_validation = bool(progress.get("pending_eval")) or progress.get("last_validated") != steps

    def keep_training(row):
        if row.get("type") == "behavior_cloning":
            return not cloning or row.get("round", 0) < rounds
        return row.get("environment_steps", 0) <= steps

    def keep_evaluation(row):
        at = row.get("environment_steps", 0)
        if at > steps or (cloning and row.get("reason") == "start"):
            return False
        return not (at == steps and row.get("reason") in ("periodic", "final") and unsaved_validation)

    for name, keep in (("training.jsonl", keep_training), ("evaluations.jsonl", keep_evaluation)):
        path = Path(folder) / name
        rows = read_jsonl(path)
        kept = [row for row in rows if keep(row)]
        if len(kept) != len(rows):
            write_text_atomic(path, "".join(dumps(row) + "\n" for row in kept))


def _clock(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _warn(message):
    print(f"Warning: {message}", flush=True)


def train(config, scenarios, total_steps=10000, horizon=128, epochs=4, seed=0,
          output="runs/train", clone_steps=0, clone_epochs=15, resume=None, device="cpu",
          critic="local", failure_reset=False, envs=8, separate_critic=True, lr=3e-4, minibatch=1024,
          gamma=0.99, lam=0.95, clip=0.2, entropy=0.0, target_kl=0.03, dagger_rounds=0, clone_noise=0.3,
          init_log_std=None, critic_warmup=None, normalize_rewards=True, eval_every=25000, eval_seeds=5,
          eval_scenarios=None, snapshot_every=0, threads=None, continue_run=False, stage=None,
          width=64, min_log_std=-3.0, bc_coef=1.0, bc_coef_end=0.5, bc_batch=1024, pipeline_version=None,
          teacher_margins=(0.7, 0.3), hard_weight=1.0, gap_features=False):
    """Train the shared drone policy and return the path of latest.pt.

    The output folder receives latest.pt (after every update and every cloning round, for resuming),
    best.pt (highest checkpoint score so far), best.json, evaluations.jsonl, training.jsonl,
    config.json and finished.json; teacher-data/ holds the cloning data until the run finishes.
    With clone_steps the run first clones the predictive controller (DART + DAgger) on its own
    scenarios, also when resuming from an earlier stage, and keeps that teacher data as an
    imitation loss inside PPO whose weight decays from bc_coef to bc_coef_end over the run.

    continue_run picks up exactly where latest.pt stopped: weights, both optimizers, reward scale,
    counters, random-number state, validation schedule and teacher data; only the episodes that were
    in flight restart.
    """
    if (total_steps < 1 or horizon < 2 or epochs < 1 or clone_steps < 0 or clone_epochs < 1 or envs < 1
            or minibatch < 1 or dagger_rounds < 0 or eval_seeds < 1):
        raise ValueError("Invalid training budget")
    if critic not in ("local", "central"):
        raise ValueError("critic must be 'local' or 'central'")
    target = Path(output)
    if continue_run:
        resume = target / "latest.pt"
        if not resume.exists():
            raise ValueError(f"Nothing to continue: {resume} does not exist")
    same_folder = bool(resume) and Path(resume).resolve().parent == target.resolve()
    if not same_folder and any((target / name).exists() for name in ("latest.pt", "best.pt")):
        raise ValueError(f"{target} already holds a trained model; use --continue to finish it, "
                         "or choose a new --output")
    hyper = {"scenarios": list(scenarios), "horizon": horizon, "epochs": epochs, "envs": envs, "lr": lr,
             "minibatch": minibatch, "gamma": gamma, "lam": lam, "clip": clip, "entropy": entropy,
             "target_kl": target_kl, "normalize_rewards": normalize_rewards, "failure_reset": failure_reset,
             "eval_every": eval_every, "eval_seeds": eval_seeds,
             "eval_scenarios": list(eval_scenarios or scenarios), "snapshot_every": snapshot_every,
             "seed": seed, "stage": stage, "bc_coef": bc_coef, "bc_coef_end": bc_coef_end, "bc_batch": bc_batch,
             "pipeline_version": pipeline_version, "teacher_margins": [float(m) for m in teacher_margins],
             "hard_weight": float(hard_weight)}
    central_critic = critic == "central"
    prior_steps = 0
    checkpoint = None
    previous = {}
    progress = {}
    if resume:
        checkpoint = load_torch(resume, map_location=device)
        if checkpoint.get("format") == ACTOR_FORMAT:
            raise ValueError(f"{resume} is an actor-only (deployment or quantized) model; it cannot be "
                             "trained further. Resume from the stage's latest.pt or best.pt instead")
        if checkpoint.get("format") not in CHECKPOINT_FORMATS:
            raise ValueError("Unsupported checkpoint format")
        if checkpoint["format"] != CHECKPOINT_FORMAT:
            raise ValueError(f"{resume} is a {checkpoint['format']} checkpoint trained with the old reward and "
                             "network; start a fresh run")
        if Config(**checkpoint["config"]).to_dict() != config.to_dict():
            raise ValueError("Resume checkpoint has incompatible environment configuration")
        previous = checkpoint.get("training") or {}
        central_critic = bool(checkpoint.get("central_critic", False))
        separate_critic = bool(checkpoint.get("separate_critic", False))
        prior_steps = int(checkpoint["environment_steps"])
        if continue_run:
            progress = checkpoint.get("progress") or {}
    run_start, target_steps = prior_steps, prior_steps + total_steps
    initialized_from = portable_path(resume) if resume else None
    if continue_run:
        hyper.update({key: previous[key] for key in hyper if key in previous})
        run_start = int(previous.get("stage_start_environment_steps", prior_steps))
        target_steps = int(previous.get("target_environment_steps", prior_steps + total_steps))
        initialized_from = previous.get("initialized_from")
        seed = hyper["seed"]
        clone_steps = int(previous.get("clone_steps") or 0)
        clone_epochs = int(previous.get("clone_epochs") or clone_epochs)
        dagger_rounds = int(previous.get("dagger_rounds") or 0)
        clone_noise = float(previous.get("clone_noise", clone_noise))
        if init_log_std is None and previous.get("init_log_std") is not None:
            init_log_std = float(previous["init_log_std"])
        if critic_warmup is None:
            critic_warmup = int(previous.get("critic_warmup_updates") or 0) if progress else 0
    total_steps = target_steps - prior_steps
    threads_used = threads or min(4, os.cpu_count() or 1)
    torch.set_num_threads(threads_used)
    torch.manual_seed(seed + prior_steps)
    rng = np.random.default_rng([seed, prior_steps])
    if progress.get("numpy_rng"):
        rng.bit_generator.state = json.loads(progress["numpy_rng"])
    if progress.get("torch_rng") is not None:
        torch.set_rng_state(progress["torch_rng"].cpu())
    target.mkdir(parents=True, exist_ok=True)
    if init_log_std is None:
        init_log_std = -1.6 if clone_steps else -1.0
    if critic_warmup is None:
        critic_warmup = 10 if clone_steps else 5 if resume and not same_folder else 0
    network = {"width": width, "min_log_std": min_log_std, "gap_features": gap_features}
    if checkpoint:
        network = checkpoint.get("network") or {"width": 64, "min_log_std": -5.0}
    model = ActorCritic(central_critic=central_critic, separate_critic=separate_critic,
                        init_log_std=init_log_std, **network).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper["lr"], eps=1e-5)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    scaler = None
    if hyper["normalize_rewards"]:
        scaler = RewardScaler(hyper["gamma"], **((checkpoint or {}).get("reward_scaler") or {}))
        if progress.get("scaler_returns") is not None:
            scaler.returns = progress["scaler_returns"].cpu().numpy()
    actor_params, critic_params = model.actor_parameters(), model.critic_parameters()
    n_agents, n_envs = config.drones, hyper["envs"]
    eval_scenarios = hyper["eval_scenarios"]
    selection_seeds = range(SELECTION_SEED_START, SELECTION_SEED_START + hyper["eval_seeds"])
    algorithm = ("parameter-sharing PPO; "
                 + ("centralized critic (MAPPO-style value)" if central_critic else "local critic (IPPO)")
                 + ("; separate critic network" if separate_critic else "; critic shares the actor encoder")
                 + "; tanh Gaussian velocity actions"
                 + ("; decaying imitation loss on teacher data" if hyper["bc_coef"] > 0 else ""))
    settings = {**hyper, "config": config.to_dict(), "network": model.network_spec(), "total_steps_this_run": total_steps,
                "stage_start_environment_steps": run_start, "target_environment_steps": target_steps,
                "clone_steps": clone_steps, "clone_epochs": clone_epochs, "dagger_rounds": dagger_rounds,
                "clone_noise": clone_noise, "init_log_std": float(init_log_std),
                "critic_warmup_updates": critic_warmup, "initialized_from": initialized_from,
                "resume": portable_path(resume) if resume else None, "continued": bool(continue_run),
                "algorithm": algorithm, "central_critic": central_critic, "separate_critic": separate_critic,
                "training_scenario_seeds": f"{TRAINING_SEEDS[0]}..{TRAINING_SEEDS[1] - 1}",
                "selection_seeds": f"{selection_seeds.start}..{selection_seeds.stop - 1}",
                "selection_rule": "checkpoint_score: selection score minus 0.5 x the mean shortfall of the "
                                  "minimum clearance below 0.15 m in the obstacle scenarios",
                "resume_semantics": "weights, optimizers, reward scale, counters, random state, validation "
                                    "schedule and teacher data; episodes in flight restart"}
    if continue_run:
        settings["critic_warmup_updates"] = previous.get("critic_warmup_updates", critic_warmup)
    write_json_atomic(target / "config.json", settings)
    status = "swarmflow-v4; attention set encoder"
    status += "; separate critic" if separate_critic else ""
    status += "; centralized critic" if central_critic else ""
    status += "; evaluate before claiming success"
    log, evaluations = target / "training.jsonl", target / "evaluations.jsonl"
    if progress:
        _trim_logs(target, prior_steps, progress)
    best_file = target / "best.json"
    teacher_dir = target / TEACHER_DATA_DIR
    best_score = -math.inf
    if same_folder and best_file.exists():
        try:
            best_score = read_json(best_file)["score"]
        except (OSError, ValueError, KeyError):
            best_score = progress.get("best_score") if progress.get("best_score") is not None else -math.inf
    eval_every = hyper["eval_every"]

    def first_eval_after(steps):
        return run_start + eval_every * ((steps - run_start) // eval_every + 1) if eval_every > 0 else None

    if progress:
        state = {"phase": progress.get("phase", "ppo"), "clone_rounds_done": int(progress.get("clone_rounds_done", 0)),
                 "updates": int(progress.get("updates", 0)), "episodes": int(progress.get("episodes", 0)),
                 "collisions": int(progress.get("collisions", 0)), "restores": int(progress.get("restores", 0)),
                 "recent": deque(((bool(f), float(s)) for f, s in progress.get("recent", [])), maxlen=100),
                 "seconds": float(progress.get("training_seconds", 0.0)),
                 "next_eval": progress.get("next_eval"), "pending_eval": bool(progress.get("pending_eval")),
                 "start_validated": bool(progress.get("start_validated", True)),
                 "last_validated": progress.get("last_validated"), "lr_scale": float(progress.get("lr_scale", 1.0)),
                 "bad_updates": 0}
        if eval_every > 0 and state["next_eval"] is None:
            state["next_eval"] = first_eval_after(prior_steps)
    else:
        state = {"phase": "cloning" if clone_steps and not continue_run else "ppo", "clone_rounds_done": 0,
                 "updates": 0, "episodes": 0, "collisions": 0, "restores": 0, "recent": deque(maxlen=100),
                 "seconds": 0.0, "next_eval": first_eval_after(prior_steps), "pending_eval": False,
                 "start_validated": False, "last_validated": None, "lr_scale": 1.0, "bad_updates": 0}
    done_steps = 0
    last_trained_at = (checkpoint or {}).get("last_trained_at")
    clone_optimizer = None
    session_started = time.monotonic()
    pending = {"best": None}

    def current_steps():
        return prior_steps + done_steps

    def progress_state():
        return {"phase": state["phase"], "clone_rounds_done": state["clone_rounds_done"],
                "updates": state["updates"], "episodes": state["episodes"], "collisions": state["collisions"],
                "restores": state["restores"], "recent": [[bool(f), float(s)] for f, s in state["recent"]],
                "training_seconds": state["seconds"], "next_eval": state["next_eval"],
                "pending_eval": state["pending_eval"], "start_validated": state["start_validated"],
                "last_validated": state["last_validated"],
                "best_score": best_score if math.isfinite(best_score) else None,
                "lr_scale": state["lr_scale"], "numpy_rng": json.dumps(rng.bit_generator.state),
                "torch_rng": torch.get_rng_state(),
                "scaler_returns": (torch.from_numpy(np.array(scaler.returns, dtype=np.float32))
                                   if scaler is not None and scaler.returns is not None else None),
                "clone_optimizer": clone_optimizer.state_dict() if state["phase"] == "cloning" and clone_optimizer
                else None}

    def checkpoint_data(extra=None):
        return {"format": CHECKPOINT_FORMAT, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config.to_dict(), "environment_steps": current_steps(), "training": settings,
                "last_trained_at": last_trained_at,
                "central_critic": central_critic, "separate_critic": separate_critic,
                "network": model.network_spec(),
                "reward_scaler": scaler.state() if scaler else None, "status": status,
                "progress": progress_state(), **(extra or {})}

    def save_latest():
        try:
            save_torch_atomic(checkpoint_data(), target / "latest.pt")
            return True
        except OSError as error:
            _warn(f"could not save latest.pt ({error}); training continues and saves again after the next update")
            return False

    def write_best(data, row):
        try:
            save_torch_atomic(data, target / "best.pt")
            write_json_atomic(best_file, row)
            pending["best"] = None
            return True
        except OSError as error:
            if pending["best"] is None or pending["best"][0] is not data:
                pending["best"] = (copy.deepcopy(data), row)
            _warn(f"could not save best.pt ({error}); it is kept in memory and saved after the next update")
            return False

    def write(path, row):
        append_jsonl(path, row)

    controller = ModelController(model, device)

    def validate(reason):
        nonlocal best_score
        began = time.monotonic()
        episodes = [run_episode(config, scenario, s, controller)[0] for scenario in eval_scenarios
                    for s in selection_seeds]
        summary = aggregate(episodes)
        score = checkpoint_score(summary)
        rows = list(summary.values())
        row = {"type": "evaluation", "reason": reason, "environment_steps": current_steps(),
               "score": score, "best": score > best_score,
               "plain_score": selection_score(summary),
               "min_obstacle_clearance_m": min_obstacle_clearance(summary),
               "success_rate": float(np.mean([r["success_rate"] for r in rows])),
               "mean_slot_rmse_m": float(np.mean([r["mean_slot_rmse_m"] for r in rows])),
               "events": sum(r["events"] for r in rows),
               "recovered_events": sum(r["recovered_events"] for r in rows),
               "scenarios": eval_scenarios, "seeds": [selection_seeds.start, selection_seeds.stop - 1],
               "seconds": round(time.monotonic() - began, 1), "summary": summary}
        write(evaluations, row)
        note = ""
        if row["best"]:
            best_score = score
            data = checkpoint_data({"selection": {key: row[key] for key in
                                    ("environment_steps", "score", "success_rate", "mean_slot_rmse_m",
                                     "scenarios", "seeds")}})
            saved = write_best(data, sanitize({**row, "checkpoint": "best.pt"}))
            note = "  -> new best, saved best.pt" if saved else "  -> new best (best.pt save delayed)"
        recovered = f"  recovered {row['recovered_events']}/{row['events']}" if row["events"] else ""
        clearance = row["min_obstacle_clearance_m"]
        clearance = f"  min clearance {clearance:.3f} m" if clearance is not None else ""
        print(f"[eval:{reason}] steps {row['environment_steps']:,}  score {score:.3f} (best {best_score:.3f})  "
              f"success {row['success_rate']:.0%}  slot RMSE {row['mean_slot_rmse_m']:.3f} m{recovered}{clearance}"
              f"{note}", flush=True)
        state["last_validated"] = current_steps()
        return time.monotonic() - began

    def stop_now(what):
        print(f"\nStopped after the last {what}. {target / 'latest.pt'} holds it and best.pt the best validated "
              "model; run the same command to continue from here.", flush=True)
        raise TrainingStopped

    workers = []
    demo = None
    with StopRequest() as stop:
        try:
            data = {key: [] for key in OBSERVATION_KEYS}
            labels = []
            if state["phase"] == "cloning":
                clone_optimizer = torch.optim.Adam(model.actor_parameters(), lr=5e-4)
                rounds_done = state["clone_rounds_done"]
                if rounds_done:
                    if _load_teacher_rounds(teacher_dir, rounds_done, data, labels):
                        if progress.get("clone_optimizer"):
                            clone_optimizer.load_state_dict(progress["clone_optimizer"])
                        print(f"Continuing behavior cloning after round {rounds_done - 1}", flush=True)
                    else:
                        _warn(f"saved cloning data in {teacher_dir} is incomplete; cloning restarts at round 0")
                        data = {key: [] for key in OBSERVATION_KEYS}
                        labels = []
                        rounds_done = state["clone_rounds_done"] = 0
                teacher = PredictiveController(config, *hyper["teacher_margins"])
                if rounds_done <= dagger_rounds:
                    print(f"Behavior cloning on {', '.join(hyper['scenarios'])}: {dagger_rounds + 1} round(s) x "
                          f"{clone_steps:,} steps", flush=True)
                for round_index in range(rounds_done, dagger_rounds + 1):
                    start = len(labels)
                    episodes_done, failures = _collect(model, teacher, config, hyper["scenarios"], clone_steps, rng,
                                                       device, data, labels, clone_noise if round_index == 0 else None)
                    demo = _demo_tensors(data, labels, device, config.radius, hyper["hard_weight"])
                    mse = _fit_clone(model, clone_optimizer, demo, clone_epochs, rng, device)
                    if math.isfinite(mse):
                        last_trained_at = datetime.now(UTC).isoformat(timespec="milliseconds")
                    row = _clone_row(round_index, clone_steps, clone_epochs, demo, episodes_done, failures, mse)
                    write(log, row)
                    print(f"[clone] round {row['round']} ({row['driver']}): {row['agent_samples']:,} samples "
                          f"({row['hard_state_share']:.0%} near objects), mse {row['final_epoch_mse']:.5f}, "
                          f"driver failures {row['driver_failures']}/{row['driver_episodes_finished']}", flush=True)
                    try:
                        _save_teacher_round(teacher_dir, round_index, data, labels, start)
                        state["clone_rounds_done"] = round_index + 1
                        save_torch_atomic(checkpoint_data(), target / "latest.pt")
                    except OSError as error:
                        state["clone_rounds_done"] = round_index + 1
                        _warn(f"could not save cloning round {round_index} ({error}); an interruption now "
                              "would repeat it")
                    if stop.requested and round_index < dagger_rounds:
                        stop_now("cloning round")
                if demo is None and labels:
                    demo = _demo_tensors(data, labels, device, config.radius, hyper["hard_weight"])
                if resume:
                    with torch.no_grad():
                        model.log_std.clamp_(min=init_log_std)
                state["phase"] = "ppo"
                clone_optimizer = None
            elif clone_steps and hyper["bc_coef"] > 0 and total_steps > 0:
                rounds_saved = int(progress.get("clone_rounds_done") or dagger_rounds + 1)
                if progress and _load_teacher_rounds(teacher_dir, rounds_saved, data, labels):
                    demo = _demo_tensors(data, labels, device, config.radius, hyper["hard_weight"])
                    print(f"Reloaded the teacher dataset for the imitation loss ({len(demo[1]):,} samples)",
                          flush=True)
                else:
                    print("Rebuilding the teacher dataset for the imitation loss", flush=True)
                    demo = demonstrations(model, config, hyper["scenarios"], clone_steps, rng, device,
                                          hyper["teacher_margins"], hyper["hard_weight"])
            del data, labels
            if total_steps > 0:
                if not state["start_validated"] or (continue_run and not progress):
                    validate("start")
                    state["start_validated"] = True
                    save_latest()
                if state["pending_eval"]:
                    validate("periodic")
                    state["pending_eval"] = False
                    while state["next_eval"] is not None and state["next_eval"] <= current_steps():
                        state["next_eval"] += eval_every
                    save_latest()
                if stop.requested:
                    stop_now("validation")
            workers = [Worker(config, hyper["scenarios"], rng, hyper["failure_reset"]) for _ in range(n_envs)] \
                if total_steps > 0 else []
            snapshot_every = hyper["snapshot_every"]
            next_snapshot = (prior_steps // snapshot_every + 1) * snapshot_every if snapshot_every > 0 else None
            started = time.monotonic()
            paused = 0.0
            good = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
            if total_steps > 0:
                print(f"PPO: {target_steps - run_start:,} environment steps over {n_envs} parallel environments "
                      f"({algorithm})" + (f"; continuing at {prior_steps - run_start:,}" if prior_steps > run_start
                                         else ""), flush=True)
            while current_steps() < target_steps:
                steps_now = min(hyper["horizon"], max(1, math.ceil((target_steps - current_steps()) / n_envs)))
                obs_store = {key: [] for key in OBSERVATION_KEYS}
                raw_store, logp_store, value_store, reward_store, scaled_store, term_store, end_store = (
                    [] for _ in range(7))
                bootstrap = []
                for t in range(steps_now):
                    packed = stack_teams([w.observations for w in workers])
                    with torch.no_grad():
                        action, raw, logp, value = model.sample(tensors(packed, device), n_agents)
                    actions = action.cpu().numpy().reshape(n_envs, n_agents, 3)
                    results = [worker.step(actions[e]) for e, worker in enumerate(workers)]
                    rewards = np.concatenate([r["reward"] for r in results])
                    ended = np.concatenate([r["ended"] for r in results])
                    for key in OBSERVATION_KEYS:
                        obs_store[key].append(packed[key])
                    raw_store.append(raw.cpu().numpy())
                    logp_store.append(logp.cpu().numpy())
                    value_store.append(value.cpu().numpy())
                    reward_store.append(rewards)
                    scaled_store.append(scaler(rewards, ended) if scaler else rewards)
                    term_store.append(np.concatenate([r["terminated"] for r in results]))
                    end_store.append(ended)
                    finals = [(e, r["final"]) for e, r in enumerate(results) if r["done"]]
                    if finals:
                        with torch.no_grad():
                            final_values = model.value(tensors(stack_teams([f for _, f in finals]), device),
                                                       n_agents).cpu().numpy()
                        for k, (e, _) in enumerate(finals):
                            bootstrap.append((t, e, final_values[k * n_agents:(k + 1) * n_agents]))
                    for r in results:
                        if r["done"]:
                            state["collisions"] += int(r["collision"])
                            state["restores"] += int(r["restored"])
                        if r["summary"] is not None:
                            state["episodes"] += 1
                            state["recent"].append((r["failed"], r["summary"]["slot_rmse_m"]))
                    done_steps += n_envs
                with torch.no_grad():
                    last_value = model.value(tensors(stack_teams([w.observations for w in workers]), device),
                                             n_agents).cpu().numpy()
                values = np.asarray(value_store, dtype=np.float32)
                next_values = np.empty_like(values)
                next_values[:-1] = values[1:]
                next_values[-1] = last_value
                for t, e, final_value in bootstrap:
                    next_values[t, e * n_agents:(e + 1) * n_agents] = final_value
                adv, returns = advantages(np.asarray(scaled_store, dtype=np.float32), values, next_values,
                                          np.asarray(term_store, dtype=np.float32),
                                          np.asarray(end_store, dtype=np.float32), hyper["gamma"], hyper["lam"])
                batch = tensors({k: np.concatenate(v) for k, v in obs_store.items()}, device)
                raw = torch.as_tensor(np.concatenate(raw_store), device=device)
                old_logp = torch.as_tensor(np.concatenate(logp_store), device=device)
                advantage = torch.as_tensor(adv.reshape(-1), device=device)
                advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp(min=1e-8)
                returns_t = torch.as_tensor(returns.reshape(-1), device=device)
                warmup = state["updates"] < critic_warmup
                fraction = min(1.0, (current_steps() - run_start) / max(1, target_steps - run_start))
                lr_now = hyper["lr"] * (1.0 - 0.9 * fraction) * state["lr_scale"]
                bc_now = hyper["bc_coef"] + (hyper["bc_coef_end"] - hyper["bc_coef"]) * fraction
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
                loss_rows = []
                unit = len(advantage) // n_agents if central_critic else len(advantage)
                mb_items = max(1, hyper["minibatch"] // n_agents) if central_critic else hyper["minibatch"]
                epochs_done = minibatches = skipped = 0
                for _ in range(hyper["epochs"]):
                    order = rng.permutation(unit)
                    kl_values = []
                    for start in range(0, unit, mb_items):
                        chunk = order[start:start + mb_items]
                        if central_critic:
                            index = (chunk[:, None] * n_agents + np.arange(n_agents)).reshape(-1)
                        else:
                            index = chunk
                        index = torch.as_tensor(index, device=device)
                        logp, entropy_values, value = model.evaluate({k: v[index] for k, v in batch.items()},
                                                                     raw[index], n_agents if central_critic else None)
                        log_ratio = logp - old_logp[index]
                        ratio = log_ratio.exp()
                        adv_mb = advantage[index]
                        policy_loss = -torch.minimum(ratio * adv_mb, ratio.clamp(1 - hyper["clip"], 1 + hyper["clip"])
                                                     * adv_mb).mean()
                        value_loss = 0.5 * (value - returns_t[index]).square().mean()
                        bc_loss = torch.zeros((), device=device)
                        if warmup:
                            loss = 0.5 * value_loss
                        else:
                            loss = policy_loss + 0.5 * value_loss - hyper["entropy"] * entropy_values.mean()
                            if demo is not None and bc_now > 0:
                                ids = torch.as_tensor(_demo_ids(demo, hyper["bc_batch"], rng), device=device)
                                bc_loss = F.mse_loss(model.act({k: v[ids] for k, v in demo[0].items()}), demo[1][ids])
                                loss = loss + bc_now * bc_loss
                        minibatches += 1
                        if not torch.isfinite(loss):
                            skipped += 1
                            optimizer.zero_grad(set_to_none=True)
                            continue
                        optimizer.zero_grad()
                        loss.backward()
                        if warmup:
                            for parameter in actor_params:
                                parameter.grad = None
                        if separate_critic:
                            if not warmup:
                                torch.nn.utils.clip_grad_norm_(actor_params, 0.5)
                            torch.nn.utils.clip_grad_norm_(critic_params, 0.5)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()
                        kl = float(((ratio - 1) - log_ratio).mean().detach().cpu())
                        kl_values.append(kl)
                        loss_rows.append((float(policy_loss.detach().cpu()), float(value_loss.detach().cpu()), kl,
                                          float(entropy_values.mean().detach().cpu()), float(bc_loss.detach().cpu())))
                    epochs_done += 1
                    if not warmup and kl_values and np.mean(kl_values) > hyper["target_kl"]:
                        break
                healthy = (skipped * 2 <= minibatches
                           and all(bool(torch.isfinite(p).all()) for p in model.parameters()))
                if healthy:
                    last_trained_at = datetime.now(UTC).isoformat(timespec="milliseconds")
                    good = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
                    state["bad_updates"] = 0
                else:
                    model.load_state_dict(good[0])
                    optimizer.load_state_dict(good[1])
                    state["lr_scale"] *= 0.5
                    state["bad_updates"] += 1
                    _warn(f"update {state['updates'] + 1} produced non-finite values in {skipped}/{minibatches} "
                          f"minibatches; restored the last good weights and halved the learning rate")
                    if state["bad_updates"] >= MAX_BAD_UPDATES:
                        raise RuntimeError(f"PPO produced non-finite values in {MAX_BAD_UPDATES} updates in a row; "
                                           f"training stopped. {target / 'latest.pt'} holds the last good update: "
                                           "run the same command to continue")
                state["updates"] += 1
                flat_returns, flat_values = returns.reshape(-1), values.reshape(-1)
                spread = float(np.var(flat_returns))
                explained = 1.0 - float(np.var(flat_returns - flat_values)) / spread if spread > 1e-12 else None
                losses = np.mean(loss_rows, axis=0) if loss_rows else np.full(5, np.nan)
                session_seconds = max(time.monotonic() - started - paused, 1e-9)
                rate = done_steps / session_seconds
                recent = state["recent"]
                row = {"type": "ppo", "phase": "critic_warmup" if warmup else "ppo", "update": state["updates"],
                       "environment_steps": current_steps(),
                       "agent_transitions": current_steps() * n_agents,
                       "episodes_this_run": state["episodes"], "collisions_this_run": state["collisions"],
                       "failure_resets_this_run": state["restores"],
                       "recent_failure_rate": float(np.mean([f for f, _ in recent])) if recent else None,
                       "recent_slot_rmse_m": float(np.mean([s for _, s in recent])) if recent else None,
                       "mean_step_reward": float(np.mean(reward_store)), "policy_loss": float(losses[0]),
                       "value_loss": float(losses[1]), "approx_kl": float(losses[2]), "entropy": float(losses[3]),
                       "bc_loss": float(losses[4]), "bc_coef": bc_now if demo is not None else 0.0,
                       "explained_variance": explained, "epochs_done": epochs_done, "lr": lr_now,
                       "skipped_minibatches": skipped,
                       "action_std": float(model.log_std.detach().clamp(model.min_log_std, 1).exp().mean()),
                       "environment_steps_per_second": rate}
                write(log, row)
                failure = "-" if row["recent_failure_rate"] is None else f"{row['recent_failure_rate']:.0%}"
                rmse = "-" if row["recent_slot_rmse_m"] is None else f"{row['recent_slot_rmse_m']:.2f} m"
                imitation = f"  imitation {row['bc_loss']:.4f}" if demo is not None and not warmup else ""
                stage_total = max(1, target_steps - run_start)
                stage_done = current_steps() - run_start
                print(f"[{row['phase']}] {stage_done:,}/{stage_total:,} ({stage_done / stage_total:.1%})  "
                      f"reward/step {row['mean_step_reward']:+.4f}  failures(last 100) {failure}  "
                      f"slot RMSE(last 100) {rmse}  kl {row['approx_kl']:.4f}  value loss {row['value_loss']:.3f}"
                      f"{imitation}  std {row['action_std']:.3f}  {rate:,.0f} steps/s  "
                      f"ETA {_clock((target_steps - current_steps()) / rate)}", flush=True)
                state["seconds"] += time.monotonic() - session_started
                session_started = time.monotonic()
                due = state["next_eval"] is not None and current_steps() >= state["next_eval"]
                state["pending_eval"] = due
                save_latest()
                if pending["best"] is not None:
                    write_best(*pending["best"])
                if next_snapshot is not None and current_steps() >= next_snapshot:
                    try:
                        save_torch_atomic(checkpoint_data(),
                                          target / "checkpoints" / f"step_{current_steps():09d}.pt")
                    except OSError as error:
                        _warn(f"could not save the step {current_steps():,} snapshot ({error})")
                    next_snapshot += snapshot_every
                if due:
                    paused += validate("periodic")
                    state["pending_eval"] = False
                    while state["next_eval"] <= current_steps():
                        state["next_eval"] += eval_every
                    save_latest()
                if stop.requested and current_steps() < target_steps:
                    stop_now("update")
            if state["last_validated"] != current_steps():
                validate("final")
                save_latest()
            if pending["best"] is not None and not write_best(*pending["best"]):
                raise OSError(f"best.pt in {target} stayed locked; close the program holding it and run the same "
                              "command again")
            state["seconds"] += time.monotonic() - session_started
            best_steps = read_json(best_file)["environment_steps"] if best_file.exists() else None
            write_json_atomic(target / "finished.json", {
                "environment_steps": current_steps(), "target_environment_steps": target_steps,
                "best_score": best_score, "best_environment_steps": best_steps,
                "updates_this_run": state["updates"],
                "training_seconds_this_run": round(time.monotonic() - started, 1),
                "training_seconds_total": round(state["seconds"], 1)})
            shutil.rmtree(teacher_dir, ignore_errors=True)
        except KeyboardInterrupt as error:
            if not isinstance(error, TrainingStopped):
                print(f"\nInterrupted. {target / 'latest.pt'} holds the last completed update or cloning round and "
                      "best.pt the best validated one; run the same command to continue.", flush=True)
            raise
        finally:
            for worker in workers:
                worker.env.close()
    return target / "latest.pt"
