from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time

import numpy as np

from swarm.baseline.pid import GeometricPid
from swarm.config import load, repo_root
from swarm.env.swarm_env import SwarmEnv
from swarm.gates.pass_gate import evaluate_episode
from swarm.report import dump_pack, write_run_log
from swarm.train.loop import cli_bar, eval_episodes, load_policy, resolve_device, train
from swarm.viewer import make_viewer

log = logging.getLogger("swarm")


def main(argv: list[str] | None = None) -> int:
    mp.freeze_support()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="swarm")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--view", default="0")
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--steps", type=int, default=None, help="override train.yaml total_steps")
    p_view = sub.add_parser("view")
    p_view.add_argument("--seed", type=int, required=True)
    p_view.add_argument("--test", type=int, default=1, choices=(1, 2, 3))
    p_view.add_argument("--policy", choices=("baseline", "rl"), default="baseline")
    p_view.add_argument("--ckpt", choices=("best", "latest"), default="latest")
    p_view.add_argument("--mode", choices=("train", "prod", "baseline"), default=None)
    p_view.add_argument("--view", default="python")
    p_view.add_argument("--speed", type=float, default=None, help="view time scale (default 4)")
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--seed", type=int, required=True)
    p_eval.add_argument("--mode", choices=("train", "prod", "baseline"), default="prod")
    p_eval.add_argument("--test", type=int, default=0, choices=(0, 1, 2, 3))
    p_eval.add_argument("--view", default="0")
    args = parser.parse_args(argv)
    if args.cmd == "train":
        cfg = load(train=True)
        if args.steps is not None:
            cfg["total_steps"] = int(args.steps)
        train(cfg, seed=args.seed, view=str(args.view), resume=bool(args.resume))
        return 0
    cfg = load(train=False)
    if args.cmd == "eval":
        return _eval(cfg, args)
    return _view(cfg, args)


def _mode_policy_ckpt(args: argparse.Namespace) -> tuple[str, str, str]:
    mode = getattr(args, "mode", None)
    policy = getattr(args, "policy", "rl")
    ckpt = getattr(args, "ckpt", "latest")
    if mode == "prod":
        return "prod", "rl", "best"
    if mode == "train":
        return "train", "rl", "latest"
    if mode == "baseline":
        return "baseline", "baseline", ckpt
    if policy == "baseline":
        return "baseline", "baseline", ckpt
    return ("prod" if ckpt == "best" else "train"), policy, ckpt


def _load_rl(cfg: dict, ckpt_name: str, device):
    models = repo_root() / str(cfg.get("models_dir", "models"))
    path = models / f"{ckpt_name}.pt"
    if not path.is_file() and ckpt_name == "best":
        log.info("best.pt missing; falling back to latest.pt")
        path = models / "latest.pt"
    policy = load_policy(
        path,
        int(cfg["neighbor_k"]),
        int(cfg.get("hidden", 128)),
        device,
    )
    log.info("loaded %s (%s)", path, "prod" if ckpt_name == "best" else "train")
    return policy, path


def _view(cfg: dict, args: argparse.Namespace) -> int:
    mode, policy_name, ckpt_name = _mode_policy_ckpt(args)
    env = SwarmEnv(cfg, test_id=args.test, seed=args.seed, eval_mode=True)
    device = resolve_device("auto")
    controller = GeometricPid.from_cfg(cfg, test_id=args.test)
    policy = None
    ckpt_path = None
    if policy_name == "rl":
        try:
            policy, ckpt_path = _load_rl(cfg, ckpt_name, device)
        except RuntimeError as exc:
            log.error("%s", exc)
            return 1
    viewer = make_viewer(str(args.view), cfg, demo=True)
    speed = float(args.speed if args.speed is not None else cfg.get("view_speed", 4.0))
    setter = getattr(viewer, "set_speed", None)
    if setter is not None:
        setter(speed)
    obs = env.reset()
    done = False
    tick = 0
    n_steps = env.episode_steps
    t0 = time.perf_counter()
    closed_early = False
    log.info(
        "mode=%s  models: latest=%s  best=%s",
        mode,
        repo_root() / "models" / "latest.pt",
        repo_root() / "models" / "best.pt",
    )
    log.info(
        "speed=%gx  keys +/- change it, 0 = 1x. Test 3: drag the red target. Test 2: keys 1/2/3 or click S/R/C. Close the window to stop.",
        speed,
    )
    result = {"passed": False, "reason": "empty", "error": 1000.0, "test_id": args.test}
    try:
        while True:
            if viewer is not None:
                live = _apply_live(env, viewer)
                if live and live.get("view_speed") is not None:
                    speed = float(live["view_speed"])
                obs = env._obs()
            if policy is None:
                act = controller.step(obs)
            else:
                import torch

                t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    act, _, _ = policy.act(t, deterministic=True)
                act = act.detach().cpu().numpy()
            step = env.step(act)
            obs, done = step.obs, step.done
            tick += 1
            extra = f"view test={args.test} mode={mode} policy={policy_name} {speed:g}x"
            kind = getattr(env.task, "kind", None)
            if kind:
                extra += f" shape={kind}"
            cli_bar(tick, n_steps, extra)
            # matplotlib is the bottleneck; physics stays 100 Hz, draw ≤ 25 fps
            if viewer is not None:
                try:
                    viewer.draw(
                        env.snapshot(),
                        {
                            "step": tick,
                            "total": n_steps,
                            "reward": float(np.mean(step.rewards)),
                            "mode": mode,
                            "policy": policy_name,
                        },
                    )
                except Exception as exc:
                    log.error("view draw failed: %s", exc)
                    viewer = None
            if viewer is None:
                if done:
                    break
            elif not viewer.is_open():
                closed_early = True
                break
            elif done:
                try:
                    write_run_log(
                        cfg=cfg,
                        cmd="view",
                        mode=mode,
                        policy_name=policy_name,
                        seed=args.seed,
                        episodes=[
                            {
                                "test_id": args.test,
                                "history": env.history,
                                "collided": env.collided,
                                "out_of_bounds": env.out_of_bounds,
                                "floor_hit": env.floor_hit,
                                "extra": {"shape": getattr(env.task, "kind", None)},
                            }
                        ],
                    )
                except Exception as exc:
                    log.error("run log failed: %s", exc)
                try:
                    obs = env.reset()
                except Exception as exc:
                    log.error("view reset failed: %s", exc)
                    break
                controller._prev_rel = None
                tick = 0
                t0 = time.perf_counter()
            else:
                lag = tick * env.dt / max(speed, 0.25) - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(min(lag, 0.05))
    finally:
        sys.stderr.write("\n")
        if viewer is not None:
            viewer.close()
        result = evaluate_episode(env.history, cfg, args.test, env.collided, env.out_of_bounds)
        if env.history:
            try:
                write_run_log(
                    cfg=cfg,
                    cmd="view",
                    mode=mode,
                    policy_name=policy_name,
                    seed=args.seed,
                    episodes=[
                        {
                            "test_id": args.test,
                            "history": env.history,
                            "collided": env.collided,
                            "out_of_bounds": env.out_of_bounds,
                            "floor_hit": env.floor_hit,
                            "extra": {"shape": getattr(env.task, "kind", None), "closed": True},
                        }
                    ],
                )
            except Exception as exc:
                log.error("run log failed: %s", exc)
        try:
            dump_pack(
                cfg=cfg,
                mode=mode,
                policy_name=policy_name,
                ckpt=ckpt_path,
                seed=args.seed,
                gates=[result],
                histories={args.test: env.history},
                snaps={args.test: env.snapshot()},
            )
        except Exception as exc:
            log.error("results dump failed: %s", exc)
    log.info("gate=%s", result)
    log.info("submission folder: %s", repo_root() / "results" / "submit")
    if closed_early and result.get("reason") == "short_episode":
        log.info("window closed before the hold window; not a gate fail")
        return 0
    return 0 if result["passed"] else 1


def _eval(cfg: dict, args: argparse.Namespace) -> int:
    mode, policy_name, ckpt_name = _mode_policy_ckpt(args)
    device = resolve_device("auto")
    policy = None
    ckpt_path = None
    if policy_name == "rl":
        try:
            policy, ckpt_path = _load_rl(cfg, ckpt_name, device)
        except RuntimeError as exc:
            log.error("%s", exc)
            return 1
    log.info("eval mode=%s policy=%s ckpt=%s", mode, policy_name, ckpt_path)
    packs = eval_episodes(cfg, policy, device, args.seed)
    if int(args.test) in (1, 2, 3):
        packs = [p for p in packs if p["test_id"] == int(args.test)]
    gates = [p["gate"] for p in packs]
    for g in gates:
        log.info("test%s gate=%s", g.get("test_id"), g)
    try:
        out = dump_pack(
            cfg=cfg,
            mode=mode,
            policy_name=policy_name,
            ckpt=ckpt_path,
            seed=args.seed,
            gates=gates,
            histories={p["test_id"]: p["history"] for p in packs},
            snaps={p["test_id"]: p["snap"] for p in packs},
        )
        log.info("submission folder: %s", repo_root() / "results" / "submit")
        log.info("archive: %s", out)
    except Exception as exc:
        log.error("results dump failed: %s", exc)
        return 1
    try:
        write_run_log(
            cfg=cfg,
            cmd="eval",
            mode=mode,
            policy_name=policy_name,
            seed=args.seed,
            episodes=[
                {
                    "test_id": p["test_id"],
                    "history": p["history"],
                    "collided": p.get("collided"),
                    "out_of_bounds": p.get("out_of_bounds"),
                    "floor_hit": p.get("floor_hit"),
                    "extra": {"gate": p.get("gate")},
                }
                for p in packs
            ],
        )
    except Exception as exc:
        log.error("run log failed: %s", exc)
    return 0 if all(g.get("passed") for g in gates) else 1


def _apply_live(env: SwarmEnv, viewer) -> dict:
    poll = getattr(viewer, "poll", None)
    if poll is None:
        return {}
    try:
        live = poll()
    except Exception as exc:
        log.error("view poll failed: %s", exc)
        return {}
    tgt = live.get("target")
    if tgt is not None:
        env.set_joystick_target(np.asarray(tgt, dtype=float))
    kind = live.get("geometry_kind")
    if kind:
        try:
            env.set_geometry_kind(str(kind))
        except ValueError as exc:
            log.error("%s", exc)
    return live


if __name__ == "__main__":
    sys.exit(main())
