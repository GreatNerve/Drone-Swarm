from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from swarm.config import repo_root

log = logging.getLogger("swarm.report")


def results_root() -> Path:
    root = repo_root() / "results"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"cannot create results dir {root}: {exc}") from exc
    return root


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    return value


def _write(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        log.error("cannot write %s: %s", path, exc)


def _copy(src: Path, dest: Path) -> None:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    except OSError as exc:
        log.error("cannot copy %s -> %s: %s", src, dest, exc)


def _stride(n: int, cap: int = 4000) -> int:
    return max(1, n // cap) if n > cap else 1


def _write_csv(path: Path, histories: dict[int, list[dict]], hz: float) -> None:
    lines = ["t,test_id,drone,x,y,z,vx,vy,vz,ax,ay,az,tx,ty,tz"]
    for test_id, hist in histories.items():
        step = _stride(len(hist))
        for i, frame in enumerate(hist[::step]):
            t = (i * step) / hz
            pos = np.asarray(frame["pos"], dtype=float)
            vel = np.asarray(frame.get("vel", np.zeros_like(pos)), dtype=float)
            aim = np.asarray(frame.get("aim", np.zeros_like(pos)), dtype=float)
            tgt = np.asarray(frame.get("shared_target", np.zeros(3)), dtype=float)
            for d, (p, v, a) in enumerate(zip(pos, vel, aim)):
                lines.append(
                    f"{t:.4f},{test_id},{d},{p[0]:.5f},{p[1]:.5f},{p[2]:.5f},"
                    f"{v[0]:.5f},{v[1]:.5f},{v[2]:.5f},{a[0]:.5f},{a[1]:.5f},{a[2]:.5f},"
                    f"{tgt[0]:.5f},{tgt[1]:.5f},{tgt[2]:.5f}"
                )
    _write(path, "\n".join(lines) + "\n")


def _write_graphs(folder: Path, histories: dict[int, list[dict]], hz: float) -> None:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception as exc:
        log.error("matplotlib unavailable for graphs: %s", exc)
        return
    folder.mkdir(parents=True, exist_ok=True)
    n = max(len(histories), 1)
    fig = Figure(figsize=(8, 3.2 * n))
    FigureCanvasAgg(fig)
    axes = fig.subplots(n, 1, squeeze=False)
    items = list(histories.items()) or [(0, [])]
    for ax, (test_id, hist) in zip(axes[:, 0], items):
        if not hist:
            ax.set_title(f"Test {test_id} empty")
            continue
        pos = np.stack([h["pos"] for h in hist])
        aim = np.stack([h["aim"] for h in hist])
        dist = np.linalg.norm(pos - aim, axis=2)
        t = np.arange(len(hist)) / hz
        for i in range(dist.shape[1]):
            ax.plot(t, dist[:, i], label=f"d{i}")
        ax.set_title(f"Test {test_id} tracking error")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("m")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    try:
        fig.savefig(folder / "distance.png", dpi=120)
    except OSError as exc:
        log.error("cannot save graph: %s", exc)


def _write_images(folder: Path, cfg: dict, snaps: dict[int, dict]) -> None:
    if not snaps:
        return
    try:
        from swarm.viewer.python_view import PythonView
    except Exception as exc:
        log.error("cannot import viewer for images: %s", exc)
        return
    folder.mkdir(parents=True, exist_ok=True)
    view = None
    try:
        view = PythonView(cfg, demo=True, headless=True)
        for test_id, snap in snaps.items():
            view.draw(snap, {"step": 1, "total": 1, "reward": 0.0}, force=True)
            dest = folder / f"test{test_id}.png"
            try:
                view.fig.savefig(dest, dpi=120)
            except OSError as exc:
                log.error("cannot save %s: %s", dest, exc)
    except Exception as exc:
        log.error("image capture failed: %s", exc)
    finally:
        if view is not None:
            view.close()


def _result_md(meta: dict, gates: list[dict]) -> str:
    lines = [
        "# BTP1 run",
        "",
        f"- mode: **{meta.get('mode')}**",
        f"- policy: {meta.get('policy')}",
        f"- checkpoint: `{meta.get('ckpt')}`",
        f"- seed: {meta.get('seed')}",
        f"- dumped: {meta.get('dumped')}",
        "",
        "## Model files",
        "",
        "- train dump: `models/latest.pt` (current net, overwritten every PPO update)",
        "- prod dump: `models/best.pt` (best combined gate score)",
        "- this pack copies the loaded checkpoint into `models/`",
        "",
        "## Pass gates",
        "",
        "| Test | passed | error | reason |",
        "| --- | --- | --- | --- |",
    ]
    for g in gates:
        lines.append(
            f"| {g.get('test_id')} | {g.get('passed')} | {g.get('error')} | {g.get('reason')} |"
        )
    lines.extend(["", "Zip this folder (`results/submit`) for submission.", ""])
    return "\n".join(lines)


def dump_pack(
    *,
    cfg: dict,
    mode: str,
    policy_name: str,
    ckpt: Path | None,
    seed: int,
    gates: list[dict],
    histories: dict[int, list[dict]],
    snaps: dict[int, dict] | None = None,
    extra: dict | None = None,
    also_submit: bool = True,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = f"{stamp}_{mode}"
    out = results_root() / "archive" / tag
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / "images").mkdir(exist_ok=True)
        (out / "graphs").mkdir(exist_ok=True)
        (out / "data").mkdir(exist_ok=True)
        (out / "models").mkdir(exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"cannot create result pack {out}: {exc}") from exc
    hz = float(cfg.get("control_hz", 100))
    meta = {
        "mode": mode,
        "policy": policy_name,
        "ckpt": str(ckpt) if ckpt else None,
        "seed": seed,
        "dumped": stamp,
        "model_dump": {
            "latest": str(repo_root() / "models" / "latest.pt"),
            "best": str(repo_root() / "models" / "best.pt"),
        },
        "gates": _jsonable(gates),
        "extra": _jsonable(extra or {}),
    }
    _write(out / "result.json", json.dumps(meta, indent=2))
    _write(out / "result.md", _result_md(meta, gates))
    _write_csv(out / "data" / "trajectory.csv", histories, hz)
    _write_graphs(out / "graphs", histories, hz)
    if snaps:
        _write_images(out / "images", cfg, snaps)
    if ckpt is not None and ckpt.is_file():
        _copy(ckpt, out / "models" / ckpt.name)
    metrics = repo_root() / "runs" / f"seed{seed}" / "metrics.csv"
    if metrics.is_file():
        _copy(metrics, out / "data" / "metrics.csv")
    if also_submit:
        submit = results_root() / "submit"
        try:
            if submit.exists():
                shutil.rmtree(submit)
            shutil.copytree(out, submit)
        except OSError as exc:
            log.error("cannot update results/submit: %s", exc)
    log.info("wrote result pack %s", out)
    return out


KEEP_RUN_LOGS = 3


def run_logs_dir() -> Path:
    return repo_root() / "no-push" / "runs"


def _xyz(value) -> list[float]:
    return [round(float(x), 3) for x in np.asarray(value, dtype=float).reshape(-1)[:3]]


def _episode_summary(
    history: list[dict],
    cfg: dict,
    test_id: int,
    collided: bool,
    out_of_bounds: bool,
    floor_hit: bool,
    extra: dict | None = None,
) -> dict:
    from swarm.gates.pass_gate import evaluate_episode

    gate = evaluate_episode(history, cfg, test_id, collided, out_of_bounds)
    arena = float(np.asarray(cfg.get("arena_cm", [1000.0, 1000.0, 1000.0]), dtype=float)[0]) * 0.5
    drones: list[dict] = []
    issues: list[str] = []
    if not history:
        issues.append("empty_history")
        return {
            "test_id": test_id,
            "steps": 0,
            "gate": _jsonable(gate),
            "collided": collided,
            "out_of_bounds": out_of_bounds,
            "floor_hit": floor_hit,
            "drones": drones,
            "issues": issues,
            "extra": _jsonable(extra or {}),
        }
    pos = np.stack([h["pos"] for h in history])
    aim = np.stack([h["aim"] for h in history])
    if test_id in (1, 3):
        from swarm.gates.pass_gate import ring_error, standoff_cm

        tgt = np.stack(
            [np.asarray(h.get("shared_target", h["aim"][0]), dtype=float) for h in history]
        )
        err = ring_error(pos, tgt[:, None, :], standoff_cm(cfg))
    else:
        err = np.linalg.norm(pos - aim, axis=2)
    xy = np.linalg.norm(pos[:, :, :2], axis=2)
    n = pos.shape[1]
    for i in range(n):
        t_far = int(np.argmax(xy[:, i]))
        last_err = float(err[-1, i])
        max_r = float(xy[:, i].max())
        row = {
            "id": i,
            "last": _xyz(pos[-1, i]),
            "aim": _xyz(aim[-1, i]),
            "last_err_cm": round(last_err, 3),
            "max_r_cm": round(max_r, 3),
            "far_at_step": t_far,
            "far_pos": _xyz(pos[t_far, i]),
        }
        drones.append(row)
        if max_r > arena * 0.7:
            issues.append(f"d{i} far max_r={max_r:.2f}cm at step {t_far}")
        if last_err > 60.0:
            issues.append(f"d{i} off_slot {last_err:.2f}cm")
    if collided:
        issues.append("collided")
    if out_of_bounds:
        issues.append("out_of_bounds")
    if floor_hit:
        issues.append("floor_hit")
    if not gate.get("passed"):
        issues.append(f"gate:{gate.get('reason')}")
    return {
        "test_id": test_id,
        "steps": len(history),
        "gate": _jsonable(gate),
        "collided": collided,
        "out_of_bounds": out_of_bounds,
        "floor_hit": floor_hit,
        "drones": drones,
        "issues": issues,
        "extra": _jsonable(extra or {}),
    }


def write_run_log(
    *,
    cfg: dict,
    cmd: str,
    mode: str,
    policy_name: str,
    seed: int,
    episodes: list[dict],
    extra: dict | None = None,
    folder: Path | None = None,
) -> Path | None:
    """Write one run JSON and keep only the newest KEEP_RUN_LOGS files."""
    folder = folder or run_logs_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("cannot create run log dir %s: %s", folder, exc)
        return None
    summaries = []
    all_issues: list[str] = []
    for ep in episodes:
        hist = ep.get("history") or []
        blob = _episode_summary(
            hist,
            cfg,
            int(ep.get("test_id") or 0),
            bool(ep.get("collided")),
            bool(ep.get("out_of_bounds")),
            bool(ep.get("floor_hit")),
            ep.get("extra"),
        )
        summaries.append(blob)
        all_issues.extend(f"t{blob['test_id']}:{x}" for x in blob["issues"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    tests = "-".join(str(s["test_id"]) for s in summaries) or "0"
    path = folder / f"run_{stamp}_{cmd}_s{seed}_t{tests}.json"
    payload = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cmd": cmd,
        "mode": mode,
        "policy": policy_name,
        "seed": seed,
        "issues": all_issues,
        "episodes": summaries,
        "extra": _jsonable(extra or {}),
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        log.error("cannot write run log %s: %s", path, exc)
        return None
    _prune_run_logs(folder)
    log.info("run log %s  issues=%s", path, all_issues or ["none"])
    return path


def _prune_run_logs(folder: Path) -> None:
    try:
        files = sorted(folder.glob("run_*.json"), key=lambda p: p.name, reverse=True)
    except OSError as exc:
        log.error("cannot list run logs in %s: %s", folder, exc)
        return
    for old in files[KEEP_RUN_LOGS:]:
        try:
            old.unlink()
        except OSError as exc:
            log.error("cannot delete old run log %s: %s", old, exc)
