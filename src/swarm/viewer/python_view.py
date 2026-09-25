from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from swarm.config import repo_root
from swarm.env.swarm_env import body_overlap, body_z_m

log = logging.getLogger("swarm.viewer")

_KIND_KEYS = {"1": "square", "s": "square", "2": "rectangle", "r": "rectangle", "3": "circle", "c": "circle"}
_KINDS = ("square", "rectangle", "circle")
_BODY = ("#5dade2", "#f5b041", "#58d68d", "#af7ac5", "#f1948a", "#76d7c4", "#f7dc6f", "#85929e")
_FRONT_PROP = "#e74c3c"
_REAR_PROP = "#27ae60"


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.error("cannot read drone mesh %s: %s", path, exc)
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    for line in text.splitlines():
        if line.startswith("v "):
            verts.append([float(p) for p in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
            for i in range(1, len(idx) - 1):
                faces.append((idx[0], idx[i], idx[i + 1]))
    if not verts or not faces:
        log.error("drone mesh empty: %s", path)
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def _motor_body(arm_cm: float) -> np.ndarray:
    l = arm_cm / np.sqrt(2.0)
    return np.array([[l, -l, 0.0], [-l, -l, 0.0], [-l, l, 0.0], [l, l, 0.0]])


def _ensure_interactive() -> None:
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        return
    for name in ("TkAgg", "QtAgg", "Qt5Agg", "WXAgg"):
        try:
            matplotlib.use(name, force=True)
            return
        except Exception:
            continue
    log.error("no interactive matplotlib backend; the view window may not appear")


class PythonView:
    def __init__(self, cfg: dict, demo: bool = False, headless: bool = False):
        if demo and not headless:
            _ensure_interactive()
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        self.plt = plt
        self.demo = bool(demo)
        self.headless = bool(headless)
        self.fig = plt.figure(figsize=(12.5, 7.0))
        gs = GridSpec(2, 2, figure=self.fig, height_ratios=[5.5, 1.0], hspace=0.38, wspace=0.28, bottom=0.08, top=0.90)
        self.ax = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.ax_m = self.fig.add_subplot(gs[0, 1])
        self.ax_bar = self.fig.add_subplot(gs[1, :])
        self._frac = 0.0
        self._bar_label = "0/0  0%"
        arena = np.asarray(cfg["arena_cm"], dtype=float)
        self._xlim = (-arena[0] / 2, arena[0] / 2)
        self._ylim = (-arena[1] / 2, arena[1] / 2)
        self._zlim = (0.0, arena[2])
        self._standoff = float(cfg.get("test1_dist_cm", 40.0)) * 0.75
        self._arm = float(cfg.get("arm_cm", 12.5))
        self._collision_m = float(cfg.get("collision_cm", 20.0))
        self._last_draw = 0.0
        self.steps: list[float] = []
        self.rewards: list[float] = []
        self.scores: list[float] = []
        self.err1: list[float] = []
        self.err2: list[float] = []
        self.err3: list[float] = []
        mesh = repo_root() / "assets" / "drone" / "quadrotor.obj"
        self._verts, self._faces = _load_obj(mesh)
        if self._verts.size == 0:
            log.error("using line-frame drone; mesh missing at %s", mesh)
        else:
            log.info("loaded drone body %s (%s verts)", mesh.name, len(self._verts))
        # Match the square collision body, not a 3× inflated mesh (that made overlaps look like hits).
        self._vis_arm = max(self._arm, self._collision_m)
        self.fig.patch.set_facecolor("#111318")
        self.ax.set_facecolor("#1a1d24")
        self.ax_m.set_facecolor("#1a1d24")
        title = "swarm view" if self.demo else "swarm train"
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self._draw_bar()
        self._dragging = False
        self._live_target: np.ndarray | None = None
        self._geometry_kind: str | None = None
        self._kind_i = 0
        self._target_z = 200.0
        self._view_speed = float(cfg.get("view_speed", 4.0))
        self._drag_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._last_snap: dict | None = None
        self._kind_hits: list[tuple[str, tuple[float, float, float, float]]] = []
        try:
            self.ax_m.set_navigate(False)
        except Exception:
            pass
        if self.demo and not self.headless:
            try:
                self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
            except Exception:
                pass
            self.fig.canvas.mpl_connect("button_press_event", self._on_press)
            self.fig.canvas.mpl_connect("button_release_event", self._on_release)
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
            self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
            self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        if not self.headless:
            plt.ion()
            plt.show(block=False)

    def draw(self, snap: dict, stats: dict | None = None, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_draw < 0.04:
            return
        self._last_draw = now
        if stats:
            self._push_stats(stats)
            self._set_bar(stats)
        self._draw_world(snap, stats or {})
        self._last_snap = snap
        if self._live_target is None:
            self._target_z = float(np.asarray(snap["shared_target"])[2])
        else:
            self._target_z = float(self._live_target[2])
        if self.demo:
            self._draw_topdown(snap)
        else:
            self._draw_metrics(snap, stats or {})
        self._draw_bar()
        self.fig.canvas.draw_idle()
        if self.headless:
            return
        try:
            self.fig.canvas.flush_events()
        except Exception:
            self.plt.pause(0.001)

    def _push_stats(self, stats: dict) -> None:
        step = stats.get("step")
        if step is None:
            return
        if self.steps and step == self.steps[-1]:
            if "reward" in stats:
                self.rewards[-1] = float(stats["reward"])
            return
        self.steps.append(float(step))
        self.rewards.append(float(stats.get("reward", np.nan)))
        self.scores.append(float(stats.get("score", np.nan)))
        self.err1.append(float(stats.get("t1_err", np.nan)))
        self.err2.append(float(stats.get("t2_err", np.nan)))
        self.err3.append(float(stats.get("t3_err", np.nan)))

    def _set_bar(self, stats: dict) -> None:
        total = float(stats.get("total") or 0.0)
        step = float(stats.get("step") or 0.0)
        elapsed = stats.get("elapsed", "—")
        eta = stats.get("eta", "—")
        rate = stats.get("rate", "—")
        if total <= 0:
            self._frac = 0.0
            self._bar_label = f"{step:.0f} steps   {elapsed} elapsed   eta {eta}   {rate}"
        else:
            self._frac = min(max(step / total, 0.0), 1.0)
            self._bar_label = (
                f"{step:.0f}/{total:.0f}  {100.0 * self._frac:.1f}%   {elapsed} elapsed   eta {eta}   {rate}"
            )

    def _draw_bar(self) -> None:
        ax = self.ax_bar
        ax.cla()
        ax.set_facecolor("#111318")
        ax.barh([0], [1.0], color="#2a3038", height=0.85)
        ax.barh([0], [self._frac], color="#5dade2", height=0.85)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.7, 0.7)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(
            0.5,
            0.0,
            self._bar_label,
            ha="center",
            va="center",
            color="white",
            fontsize=13,
            fontweight="bold",
            transform=ax.transData,
        )
        ax.set_title("episode" if self.demo else "training progress", color="white", fontsize=11, pad=4)
        self.fig.suptitle(self._bar_label, color="white", fontsize=13, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_world(self, snap: dict, stats: dict) -> None:
        ax = self.ax
        ax.cla()
        ax.set_facecolor("#1a1d24")
        pos = np.asarray(snap["pos"], dtype=float)
        rot = np.asarray(snap["rot"], dtype=float) if snap.get("rot") is not None else np.repeat(np.eye(3)[None], len(pos), 0)
        target = np.asarray(snap["shared_target"], dtype=float)
        self._arena(ax)
        self._ground(ax)
        self._target(ax, target)
        slots = snap.get("slots")
        s = None
        if slots is not None:
            s = np.asarray(slots)
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], c="#7dcea0", s=40, marker="^", depthshade=False)
        hit = np.zeros(len(pos), dtype=bool)
        if len(pos) > 1:
            hit = body_overlap(pos, self._collision_m, body_z_m(self._collision_m)).any(axis=1)
        for i, p in enumerate(pos):
            color = "#e74c3c" if hit[i] else _BODY[i % len(_BODY)]
            R = rot[i]
            self._shadow(ax, p)
            self._square_body(ax, p, R, color)
            self._frame_body(ax, p, R, color)
            self._props_3d(ax, p, R)
            tip = p + 0.55 * self._vis_arm * R[:, 0]
            ax.plot([p[0], tip[0]], [p[1], tip[1]], [p[2], tip[2]], color="#f4d03f", lw=2.2)
            aim = s[i] if s is not None and i < len(s) else target
            d = float(np.linalg.norm(p - aim))
            ax.text(p[0], p[1], p[2] + 18.0, f"d{i} {d:.1f}cm", color=color, fontsize=7)
        test_id = int(snap.get("test_id") or 1)
        names = {1: "Test 1 — shared target", 2: "Test 2 — geometry slots", 3: "Test 3 — moving target"}
        mode = str(stats.get("mode") or "")
        policy = str(stats.get("policy") or "")
        if policy == "rl":
            kind = f"RL {mode} net"
        elif self.demo:
            kind = "PID hover"
        else:
            kind = "noisy train collect"
        ax.set_title(
            f"{names.get(test_id, 'Test')}  ({kind})   {self._view_speed:g}x  +/- speed",
            color="white",
            pad=8,
        )
        reach, span = self._frame_camera(ax, pos, target, test_id, s)
        if bool(np.any(hit)):
            ax.text2D(
                0.02,
                0.10,
                "square body overlap at this altitude",
                transform=ax.transAxes,
                color="#e74c3c",
                fontsize=8,
            )
        if self._verts.size == 0:
            ax.text2D(0.02, 0.06, "OBJ mesh missing — drawing X-frame", transform=ax.transAxes, color="#e74c3c", fontsize=8)
        ax.text2D(
            0.02,
            0.02,
            "drag the red dot on the RIGHT map  (not this 3D view)",
            transform=ax.transAxes,
            color="#f5b041",
            fontsize=8,
        )
        if reach > span * 0.98:
            ax.text2D(0.5, 0.04, "drones left the hover volume", ha="center", transform=ax.transAxes, color="#e74c3c", fontsize=8)
        ax.set_xlabel("x", color="#aaa")
        ax.set_ylabel("y", color="#aaa")
        ax.set_zlabel("z", color="#aaa")
        ax.tick_params(colors="#888")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.view_init(elev=22, azim=-60)

    def _square_body(self, ax, p: np.ndarray, R: np.ndarray, color: str) -> None:
        h = 0.5 * self._collision_m
        corners = np.array([[h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0], [-h, h, 0.0], [h, h, 0.0]])
        w = p + corners @ R.T
        ax.plot(w[:, 0], w[:, 1], w[:, 2], color=color, lw=2.6)

    def _mesh_body(self, ax, p: np.ndarray, R: np.ndarray, color: str) -> None:
        self._square_body(ax, p, R, color)

    def _frame_camera(self, ax, pos: np.ndarray, target: np.ndarray, test_id: int = 1, slots: np.ndarray | None = None) -> tuple[float, float]:
        focus = target if self.demo else pos.mean(axis=0)
        if test_id == 2 and slots is not None and len(slots):
            focus = slots.mean(axis=0)
            span = max(220.0, float(np.max(np.linalg.norm(slots - focus, axis=1))) * 1.8 + 60.0)
            span = min(span, 350.0)
            ax.set_xlim(focus[0] - span, focus[0] + span)
            ax.set_ylim(focus[1] - span, focus[1] + span)
            z0 = max(0.0, focus[2] - span * 0.6)
            ax.set_zlim(z0, z0 + span * 1.2)
            try:
                ax.set_box_aspect((1, 1, 0.7))
            except Exception:
                pass
            reach = float(np.max(np.linalg.norm(pos - focus, axis=1))) if len(pos) else 0.0
            return reach, span
        reach = float(np.max(np.linalg.norm(pos - focus, axis=1))) if len(pos) else 0.0
        span = max(220.0, reach * 1.15 + 40.0)
        if self.demo:
            span = min(span, 800.0)
        else:
            span = max(span, 350.0)
        ax.set_xlim(focus[0] - span, focus[0] + span)
        ax.set_ylim(focus[1] - span, focus[1] + span)
        z0 = max(0.0, focus[2] - span * 0.6)
        ax.set_zlim(z0, z0 + span * 1.2)
        try:
            ax.set_box_aspect((1, 1, 0.7))
        except Exception:
            pass
        return reach, span

    def _arena(self, ax) -> None:
        x0, x1 = self._xlim
        y0, y1 = self._ylim
        z0, z1 = self._zlim
        corners = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            p, q = corners[a], corners[b]
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#3d4450", lw=0.7, alpha=0.7)

    def _ground(self, ax) -> None:
        x0, x1 = self._xlim
        y0, y1 = self._ylim
        for x in np.linspace(x0, x1, 6):
            ax.plot([x, x], [y0, y1], [0, 0], color="#2a3038", lw=0.5)
        for y in np.linspace(y0, y1, 6):
            ax.plot([x0, x1], [y, y], [0, 0], color="#2a3038", lw=0.5)

    def _target(self, ax, target: np.ndarray) -> None:
        ax.scatter([target[0]], [target[1]], [target[2]], c="#e74c3c", s=90, marker="o", depthshade=False)
        th = np.linspace(0, 2 * np.pi, 48)
        ax.plot(
            target[0] + self._standoff * np.cos(th),
            target[1] + self._standoff * np.sin(th),
            np.full_like(th, target[2]),
            color="#e74c3c",
            lw=1.2,
            ls="--",
        )

    def _shadow(self, ax, p: np.ndarray) -> None:
        th = np.linspace(0, 2 * np.pi, 20)
        r = self._vis_arm * 0.9
        ax.plot(p[0] + r * np.cos(th), p[1] + r * np.sin(th), np.zeros_like(th), color="#000", lw=1.0, alpha=0.35)

    def _frame_body(self, ax, p: np.ndarray, R: np.ndarray, color: str) -> None:
        motors = p + _motor_body(self._vis_arm) @ R.T
        ax.plot([motors[0, 0], motors[2, 0]], [motors[0, 1], motors[2, 1]], [motors[0, 2], motors[2, 2]], color=color, lw=3.0)
        ax.plot([motors[1, 0], motors[3, 0]], [motors[1, 1], motors[3, 1]], [motors[1, 2], motors[3, 2]], color=color, lw=3.0)

    def _props_3d(self, ax, p: np.ndarray, R: np.ndarray) -> None:
        motors = p + _motor_body(self._vis_arm) @ R.T
        th = np.linspace(0, 2 * np.pi, 12)
        radius = self._vis_arm * 0.42
        disk = np.stack([radius * np.cos(th), radius * np.sin(th), np.zeros_like(th)], axis=1)
        for i, m in enumerate(motors):
            c = _FRONT_PROP if i in (0, 3) else _REAR_PROP
            w = m + disk @ R.T
            ax.plot(w[:, 0], w[:, 1], w[:, 2], color=c, lw=1.6)

    def _draw_topdown(self, snap: dict) -> None:
        ax = self.ax_m
        ax.cla()
        ax.set_facecolor("#1a1d24")
        pos = np.asarray(snap["pos"], dtype=float)
        rot = np.asarray(snap["rot"], dtype=float) if snap.get("rot") is not None else np.repeat(np.eye(3)[None], len(pos), 0)
        target = np.asarray(snap["shared_target"], dtype=float)
        th = np.linspace(0, 2 * np.pi, 48)
        ax.plot(target[0] + self._standoff * np.cos(th), target[1] + self._standoff * np.sin(th), color="#e74c3c", ls="--", lw=1.1)
        ax.scatter([target[0]], [target[1]], c="#e74c3c", s=220, zorder=6, edgecolors="white", linewidths=1.4)
        ax.annotate(
            "drag",
            (target[0], target[1]),
            textcoords="offset points",
            xytext=(10, 10),
            color="#f5b041",
            fontsize=9,
            fontweight="bold",
        )
        slots = snap.get("slots")
        s = None
        if slots is not None:
            s = np.asarray(slots)
            ax.scatter(s[:, 0], s[:, 1], c="#7dcea0", s=36, marker="^", zorder=3)
        hit = np.zeros(len(pos), dtype=bool)
        if len(pos) > 1:
            hit = body_overlap(pos, self._collision_m, body_z_m(self._collision_m)).any(axis=1)
        for i, p in enumerate(pos):
            color = "#e74c3c" if hit[i] else _BODY[i % len(_BODY)]
            self._quad_2d(ax, p, rot[i], color)
            ax.text(p[0] + 6.0, p[1] + 6.0, f"d{i}", color=color, fontsize=8)
        ax.set_title("top-down  |  drag red target   S/R/C shape", color="white")
        test_id = int(snap.get("test_id") or 1)
        if self._drag_limits is not None:
            (x0, x1), (y0, y1) = self._drag_limits
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            dists = np.linalg.norm(pos - s, axis=1) if test_id == 2 and s is not None else np.linalg.norm(pos - target, axis=1)
        elif test_id == 2 and s is not None and len(s):
            focus = s.mean(axis=0)
            span = min(350.0, max(180.0, float(np.max(np.linalg.norm(s[:, :2] - focus[:2], axis=1))) * 1.8 + 40.0))
            ax.set_xlim(focus[0] - span, focus[0] + span)
            ax.set_ylim(focus[1] - span, focus[1] + span)
            dists = np.linalg.norm(pos - s, axis=1)
        else:
            reach = float(np.max(np.linalg.norm(pos[:, :2] - target[:2], axis=1))) if len(pos) else 0.0
            span = max(160.0, reach * 1.4 + 30.0)
            ax.set_xlim(target[0] - span, target[0] + span)
            ax.set_ylim(target[1] - span, target[1] + span)
            dists = np.linalg.norm(pos - target, axis=1)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="#888")
        ax.grid(True, color="#2a3038")
        ax.set_xlabel("x", color="#aaa")
        ax.set_ylabel("y", color="#aaa")
        kind = snap.get("geometry_kind") or ""
        if test_id == 2:
            hint = f"drag red = move shape   S/R/C or 1/2/3   +/- speed {self._view_speed:g}x"
            self._kind_hits = [
                ("square", (0.02, 0.88, 0.42, 0.98)),
                ("rectangle", (0.02, 0.78, 0.42, 0.88)),
                ("circle", (0.02, 0.68, 0.42, 0.78)),
            ]
            for i, k in enumerate(_KINDS):
                ax.text(
                    0.03,
                    0.93 - i * 0.09,
                    f"[{k[0].upper()}] {k}",
                    transform=ax.transAxes,
                    color="#7dcea0" if k == kind else "#bbb",
                    fontsize=10,
                    fontweight="bold" if k == kind else "normal",
                )
        else:
            self._kind_hits = []
            if test_id == 3:
                hint = f"drag red target  |  wheel = z  |  +/- speed {self._view_speed:g}x"
            else:
                hint = f"drag red target  |  +/- speed {self._view_speed:g}x"
        ax.text(
            0.02,
            0.02,
            f"{hint}\nmean dist {dists.mean():.1f}cm   min pair {self._min_pair(pos):.1f}cm",
            transform=ax.transAxes,
            color="#ddd",
            fontsize=8,
            family="monospace",
        )

    def _quad_2d(self, ax, p: np.ndarray, R: np.ndarray, color: str) -> None:
        h = 0.5 * self._collision_m
        sq = np.array([[h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0], [-h, h, 0.0], [h, h, 0.0]])
        w = p + sq @ R.T
        ax.plot(w[:, 0], w[:, 1], color=color, lw=1.8, zorder=4)
        motors = p + _motor_body(self._vis_arm) @ R.T
        ax.plot([motors[0, 0], motors[2, 0]], [motors[0, 1], motors[2, 1]], color="#d5d8dc", lw=2.0, zorder=3)
        ax.plot([motors[1, 0], motors[3, 0]], [motors[1, 1], motors[3, 1]], color="#d5d8dc", lw=2.0, zorder=3)
        r = self._vis_arm * 0.36
        for i, m in enumerate(motors):
            circ = self.plt.Circle((m[0], m[1]), r, fill=False, ec=_FRONT_PROP if i in (0, 3) else _REAR_PROP, lw=1.4, zorder=4)
            ax.add_patch(circ)
        ax.scatter([p[0]], [p[1]], c=color, s=28, zorder=5)

    def _min_pair(self, pos: np.ndarray) -> float:
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(d.min()) if d.size else 0.0

    def _draw_metrics(self, snap: dict | None = None, stats: dict | None = None) -> None:
        ax = self.ax_m
        ax.cla()
        ax.set_facecolor("#1a1d24")
        ax.set_title("metrics  |  lower dist is better", color="white")
        if self.steps:
            ax.plot(self.steps, self.rewards, color="#5dade2", label="train reward")
            ax.plot(self.steps, self.scores, color="#f4d03f", label="eval score")
            ax.plot(self.steps, self.err1, color="#e74c3c", label="test1 dist")
            ax.plot(self.steps, self.err2, color="#58d68d", label="test2 dist")
            ax.plot(self.steps, self.err3, color="#af7ac5", label="test3 dist")
        ax.legend(loc="upper right", fontsize=7, facecolor="#1a1d24", edgecolor="#333", labelcolor="white")
        ax.set_xlabel("step", color="#aaa")
        ax.tick_params(colors="#888")
        ax.grid(True, color="#2a3038")
        lines = ["live view = exploration noise", "test a checkpoint: view --policy rl --ckpt latest"]
        stats = stats or {}
        if stats.get("elapsed"):
            lines.append(f"elapsed {stats.get('elapsed')}   eta {stats.get('eta')}   {stats.get('rate', '')}")
        if stats.get("saved"):
            lines.append(f"saved {stats.get('saved')}")
        if snap is not None:
            pos = np.asarray(snap["pos"])
            vel = np.asarray(snap.get("vel", np.zeros_like(pos)))
            tgt = np.asarray(snap["shared_target"])
            dists = np.linalg.norm(pos - tgt, axis=1)
            speeds = np.linalg.norm(vel, axis=1)
            lines.append(f"mean dist {dists.mean():.1f}cm   mean speed {speeds.mean():.1f}cm/s")
            for i, (d, s, p) in enumerate(zip(dists, speeds, pos)):
                lines.append(f"d{i}  {d:.1f}cm  {s:.1f}cm/s  z={p[2]:.1f}")
        if stats.get("collisions") is not None:
            lines.append(f"horizon collisions {stats.get('collisions')}")
        ax.text(
            0.02,
            0.02,
            "\n".join(lines),
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            color="#ddd",
            fontsize=8,
            family="monospace",
        )

    def set_speed(self, speed: float) -> None:
        self._view_speed = float(np.clip(speed, 0.5, 16.0))

    def poll(self) -> dict:
        out: dict = {"view_speed": self._view_speed}
        if self._live_target is not None:
            out["target"] = self._live_target.copy()
        if self._geometry_kind is not None:
            out["geometry_kind"] = self._geometry_kind
            self._geometry_kind = None
        return out

    def is_open(self) -> bool:
        try:
            return bool(self.plt.fignum_exists(self.fig.number))
        except Exception:
            return False

    def _clip_xy(self, x: float, y: float) -> tuple[float, float]:
        x = float(np.clip(x, self._xlim[0], self._xlim[1]))
        y = float(np.clip(y, self._ylim[0], self._ylim[1]))
        return x, y

    def _in_map(self, event) -> bool:
        if event.inaxes is self.ax_m:
            return True
        if event.x is None or event.y is None:
            return False
        try:
            return bool(self.ax_m.get_window_extent().contains(event.x, event.y))
        except Exception:
            return False

    def _event_xy(self, event, *, dragging: bool = False) -> tuple[float, float] | None:
        if event.inaxes is self.ax_m and event.xdata is not None and event.ydata is not None:
            return float(event.xdata), float(event.ydata)
        if not (dragging or self._in_map(event)):
            return None
        if event.x is None or event.y is None:
            return None
        try:
            x, y = self.ax_m.transData.inverted().transform((event.x, event.y))
        except Exception:
            return None
        return float(x), float(y)

    def _set_target_xy(self, x: float, y: float) -> None:
        x, y = self._clip_xy(x, y)
        z = float(np.clip(self._target_z, self._zlim[0] + 30.0, self._zlim[1] - 30.0))
        self._live_target = np.array([x, y, z], dtype=float)

    def _on_press(self, event) -> None:
        xy = self._event_xy(event)
        if xy is None:
            return
        snap = self._last_snap
        test_id = int((snap or {}).get("test_id") or 1)
        if test_id == 2:
            kind = self._hit_kind(event)
            if kind is not None:
                self._geometry_kind = kind
                return
        self._dragging = True
        self._drag_limits = (self.ax_m.get_xlim(), self.ax_m.get_ylim())
        self._set_target_xy(xy[0], xy[1])

    def _on_motion(self, event) -> None:
        if not self._dragging:
            return
        xy = self._event_xy(event, dragging=True)
        if xy is None:
            return
        self._set_target_xy(xy[0], xy[1])

    def _on_release(self, event) -> None:
        self._dragging = False
        self._drag_limits = None

    def _on_scroll(self, event) -> None:
        if not self._in_map(event):
            return
        snap = self._last_snap
        if snap is None:
            return
        dz = 12.0 if getattr(event, "button", None) == "up" else -12.0
        if self._live_target is None:
            self._live_target = np.asarray(snap["shared_target"], dtype=float).copy()
        self._target_z = float(np.clip(self._live_target[2] + dz, self._zlim[0] + 30.0, self._zlim[1] - 30.0))
        self._live_target[2] = self._target_z

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in ("+", "=", "up"):
            self._view_speed = float(min(16.0, self._view_speed * 2.0))
            return
        if key in ("-", "down"):
            self._view_speed = float(max(0.5, self._view_speed / 2.0))
            return
        if key == "0":
            self._view_speed = 1.0
            return
        if key in _KIND_KEYS:
            self._geometry_kind = _KIND_KEYS[key]
            return
        if key == "g":
            self._kind_i = (self._kind_i + 1) % len(_KINDS)
            self._geometry_kind = _KINDS[self._kind_i]

    def _hit_kind(self, event) -> str | None:
        try:
            fx, fy = self.ax_m.transAxes.inverted().transform((event.x, event.y))
        except Exception:
            return None
        for kind, (x0, y0, x1, y1) in self._kind_hits:
            if x0 <= fx <= x1 and y0 <= fy <= y1:
                return kind
        return None

    def save_images(self, folder: Path, name: str = "world.png") -> None:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(folder / name, dpi=120)
        except OSError as exc:
            log.error("cannot save image %s: %s", folder / name, exc)

    def close(self) -> None:
        self.plt.close(self.fig)
