"""Paper-style figures of the report, all rendered from recorded traces of the final (V3) model.

    PYTHONPATH=src:/tmp/mplpkgs:Report/latex/figures/scripts .venv/bin/python Report/latex/figures/scripts/make_paper_figures.py [names...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import render3d as r3

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 10, "savefig.dpi": 250, "savefig.bbox": "tight"})


def load(scenario):
    return json.loads((HERE / "traces" / f"{scenario}_100000.json").read_text())


def bounds(pos, pad=1.6):
    return (pos[..., 0].min() - pad, pos[..., 0].max() + pad), (pos[..., 1].min() - pad, pos[..., 1].max() + pad)


def centred(frames, when, hx=4.2, hy=3.6):
    t, pos, _, _ = r3.arrays(frames)
    k = int(np.argmin(np.abs(t - when)))
    c = pos[k].mean(axis=0)
    return (c[0] - hx, c[0] + hx), (c[1] - hy, c[1] + hy)


def render_panels(frames, panels, path, figsize, cols=None, elev=34, azim=-62, zoom=1.25, zlim=(0, 3.8)):
    cols = cols or len(panels)
    rows = int(np.ceil(len(panels) / cols))
    fig = plt.figure(figsize=figsize)
    for n, (t0, t1, when, title, ghosts) in enumerate(panels):
        ax = fig.add_subplot(rows, cols, n + 1, projection="3d")
        xlim, ylim = centred(frames, when)
        r3.scene(ax, frames, t0, t1, when, xlim, ylim, zlim=zlim, elev=elev, azim=azim, ghost_objects=ghosts,
                 zoom=zoom)
        ax.set_title(title, y=0.97, fontsize=10.5)
    fig.subplots_adjust(wspace=0.0, hspace=0.02, left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / path)
    plt.close(fig)


def fig_crossing_render():
    data = load("crossing")
    ev = data["summary"]["events"][0]
    rec = ev["measured_from_s"] + ev["recovery_s"]
    panels = [(2.5, 6.6, 6.6, "(a) Following in formation; object approaching", ()),
              (3.5, 8.3, 8.3, "(b) Yielding at closest approach (t = 8.3 s)", ()),
              (6.0, rec, rec, f"(c) Every drone back in its own slot (t = {rec:.1f} s)", (8.3,))]
    render_panels(data["frames"], panels, "fig_render_crossing.png", (14, 4.4))


def fig_headon_double_render():
    fig = plt.figure(figsize=(14, 4.4))
    specs = [("head_on", 3.0, 7.6, 7.6, (), "(a) Head-on object: yielding (t = 7.6 s)"),
             ("head_on", 6.0, 12.1, 12.1, (7.6,), "(b) Head-on object: recovered (t = 12.1 s)"),
             ("double_crossing", 12.5, 16.5, 16.5, (), "(c) Repeated crossings: second object (t = 16.5 s)")]
    for n, (scenario, t0, t1, when, ghosts, title) in enumerate(specs):
        frames = load(scenario)["frames"]
        ax = fig.add_subplot(1, 3, n + 1, projection="3d")
        xlim, ylim = centred(frames, when)
        r3.scene(ax, frames, t0, t1, when, xlim, ylim, zlim=(0, 3.8), elev=34, azim=-62, ghost_objects=ghosts,
                 zoom=1.25)
        ax.set_title(title, y=0.97, fontsize=10.5)
    fig.subplots_adjust(wspace=0.0, left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / "fig_render_headon_double.png")
    plt.close(fig)


THUMBS = [("hold", "(a) Hold", 12.0, 12.0, 12.0, ()), ("follow", "(b) Follow", 7.0, 12.0, 12.0, ()),
          ("turn", "(c) Turning target", 14.0, 20.0, 20.0, ()), ("gust", "(d) Wind gust", 6.0, 10.0, 9.5, ()),
          ("crossing", "(e) Crossing object", 3.5, 8.3, 8.3, ()), ("head_on", "(f) Head-on object", 3.5, 7.6, 7.6, ()),
          ("double_crossing", "(g) Repeated crossings", 12.0, 16.5, 16.5, ())]


def thumb(ax, scenario, t0, t1, when, ghosts=(), zoom=1.7):
    frames = load(scenario)["frames"]
    xlim, ylim = centred(frames, when, 3.3, 2.9)
    r3.scene(ax, frames, t0, t1, when, xlim, ylim, zlim=(0, 4.4), elev=36, azim=-62, ghost_objects=ghosts,
             zoom=zoom, show_slots=False, size=3)


def quad2d(ax, p, color, s=0.15, z=5, alpha=1.0):
    from matplotlib.patches import Circle
    d = s / np.sqrt(2)
    for sx, sy in ((1, 1), (-1, -1)), ((-1, 1), (1, -1)):
        ax.plot([p[0] + sx[0] * d, p[0] + sy[0] * d], [p[1] + sx[1] * d, p[1] + sy[1] * d], color="0.15", lw=0.9,
                zorder=z, alpha=alpha)
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        ax.add_patch(Circle((p[0] + sx * d, p[1] + sy * d), s * 0.45, fc=color, ec="0.15", lw=0.4, zorder=z,
                            alpha=alpha))


def observation_panel(ax, when=7.2, ego=3, compact=False):
    """Top view of the local observation of one drone, drawn from the recorded state."""
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
    frames = load("crossing")["frames"]
    t, pos, slot, tgt = r3.arrays(frames)
    k = int(np.argmin(np.abs(t - when)))
    p, s = pos[k], slot[k]
    e = p[ego]
    ax.set_aspect("equal")
    ax.set_xlim(e[0] - 3.3, e[0] + 3.3)
    ax.set_ylim(e[1] - 2.9, e[1] + 3.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.add_patch(Circle(e[:2], 7.0, fc="#eef4fb", ec="#4a6fb5", lw=1.0, ls="--", zorder=0))
    ax.add_patch(Rectangle((e[0] - 3.3, e[1] - 2.9), 6.6, 6.2, fc="none", ec="0.6", lw=0.8, zorder=0))
    dist = np.linalg.norm(p - e, axis=1)
    order = [j for j in np.argsort(dist) if j != ego]
    seen, unseen = order[:6], order[6:]
    for j in range(len(p)):
        ax.add_patch(Rectangle((s[j, 0] - 0.1, s[j, 1] - 0.1), 0.2, 0.2, fc="none", ec="0.55", lw=0.7, zorder=1))
    for j in seen:
        ax.add_patch(FancyArrowPatch(e[:2], p[j, :2], arrowstyle="-|>", mutation_scale=8, color="#5b9a3c",
                                     lw=0.9, zorder=3, shrinkA=6, shrinkB=6))
    for j in range(len(p)):
        quad2d(ax, p[j], r3.DRONE_COLORS[j], alpha=1.0 if j in seen or j == ego else 0.35)
    for c, r, v in r3.objects_at(frames[k]):
        ax.add_patch(Circle(c[:2], r, fc="#f6c9c3", ec="#c0392b", lw=1.0, zorder=2))
        ax.add_patch(FancyArrowPatch(c[:2], c[:2] + 0.9 * v[:2] / np.linalg.norm(v[:2]), arrowstyle="-|>",
                                     mutation_scale=10, color="#c0392b", lw=1.3, zorder=4))
        ax.add_patch(FancyArrowPatch(e[:2], c[:2], arrowstyle="-|>", mutation_scale=9, color="#d9824a", lw=1.2,
                                     zorder=3, shrinkA=6, shrinkB=0))
        if not compact:
            ax.text(c[0] + 0.15, c[1] - r - 0.25, "object: rel. position,\nrel. velocity, radius", fontsize=8,
                    color="#a0331f", va="top")
    ax.add_patch(FancyArrowPatch(e[:2], s[ego, :2], arrowstyle="-|>", mutation_scale=9, color="#4a6fb5", lw=1.4,
                                 zorder=6, shrinkA=4, shrinkB=0))
    ax.add_patch(Circle(e[:2], 0.36, fc="none", ec="black", lw=1.4, zorder=6))
    if not compact:
        ax.text(e[0] + 0.42, e[1] - 0.2, r"ego drone $i$", fontsize=9.5, ha="left", va="top")
        ax.text(s[ego, 0] + 0.15, s[ego, 1] - 0.3, r"slot error $s_i-p_i$", fontsize=8.5, color="#4a6fb5")
        ax.text(e[0] - 3.2, e[1] + 3.1, "sensing radius 7 m (dashed); 6 nearest neighbours observed",
                fontsize=8, color="#4a6fb5", va="top")
        if unseen:
            j = unseen[0]
            ax.text(p[j, 0], p[j, 1] - 0.35, "not observed\n(beyond K = 6)", fontsize=7.5, ha="center", va="top",
                    color="0.45")
    return k


def fig_observation():
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    observation_panel(ax)
    fig.savefig(OUT / "fig_local_observation.png")
    plt.close(fig)


def fig_overview():
    import matplotlib.image as mpimg
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig = plt.figure(figsize=(16, 6.3))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 16)
    bg.set_ylim(0, 6.3)
    bg.axis("off")

    def panel(x, y, w, h, title):
        bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc="none",
                                    ec="0.25", lw=1.2, ls=(0, (5, 3))))
        bg.text(x + w / 2, y + 0.12, title, ha="center", va="bottom", fontsize=10.5)

    panel(0.1, 0.25, 6.2, 5.55, "TRAINING SCENARIOS (7, randomised per seed)")
    panel(6.6, 0.25, 4.2, 5.55, "SHARED DECENTRALISED POLICY")
    panel(11.1, 0.25, 4.8, 5.55, "LEARNED BEHAVIOUR (FINAL MODEL)")
    w, h = 2.0, 1.45
    spots = [(0.15, 4.2), (2.2, 4.2), (4.25, 4.2), (0.15, 2.5), (2.2, 2.5), (4.25, 2.5), (2.2, 0.8)]
    for (scenario, title, t0, t1, when, ghosts), (x, y) in zip(THUMBS, spots):
        ax = fig.add_axes([x / 16, y / 6.3, w / 16, h / 6.3], projection="3d")
        thumb(ax, scenario, t0, t1, when, ghosts)
        bg.text(x + w / 2, y - 0.02, title, ha="center", va="top", fontsize=8.5)
    obs = fig.add_axes([6.75 / 16, 2.1 / 6.3, 2.4 / 16, 3.1 / 6.3])
    observation_panel(obs, compact=True)
    bg.text(7.95, 5.4, r"local observation $o_i$ of each drone", ha="center", fontsize=9.5)
    bg.text(7.95, 2.0, "self (17) · 6 neighbours · 6 objects", ha="center", va="top", fontsize=8.5, color="0.3")
    bg.add_patch(FancyBboxPatch((9.55, 3.2), 1.15, 1.25, boxstyle="round,pad=0,rounding_size=0.15",
                                fc="#fcc21b", ec="0.2", lw=1.2))
    bg.text(10.125, 3.825, "Attention-\nbased\npolicy " + r"$\pi_\theta$", ha="center", va="center", fontsize=10)
    bg.add_patch(FancyArrowPatch((9.2, 3.8), (9.55, 3.8), arrowstyle="-|>", mutation_scale=16, lw=2.5,
                                 color="#3a6fc4"))
    bg.text(8.7, 1.2, "same weights run on all 8 drones; action = desired\n3-D velocity "
            r"$a_i\in[-1,1]^3$, $u_i = 3a_i$ m/s", ha="center", va="center", fontsize=8.5)
    bg.text(8.7, 0.62, "trained by imitation warm start + PPO", ha="center", fontsize=8.5, style="italic")
    # action loop back to environment
    blue = "#3a6fc4"
    bg.plot([10.7, 10.95, 10.95, 3.2, 3.2], [3.8, 3.8, 6.05, 6.05, 5.85], color=blue, lw=3.2, solid_capstyle="butt")
    bg.add_patch(FancyArrowPatch((3.2, 5.9), (3.2, 5.55), arrowstyle="-|>", mutation_scale=18, color=blue, lw=0))
    bg.text(7.1, 6.1, "actions of all drones (simultaneous, 20 Hz)", ha="center", va="bottom", fontsize=9,
            color=blue)
    bg.add_patch(FancyArrowPatch((6.3, 3.65), (6.72, 3.65), arrowstyle="-|>", mutation_scale=18, lw=3.2, color=blue))
    # right panel
    ax = fig.add_axes([11.15 / 16, 3.05 / 6.3, 4.7 / 16, 2.55 / 6.3], projection="3d")
    frames = load("crossing")["frames"]
    ev = load("crossing")["summary"]["events"][0]
    rec = ev["measured_from_s"] + ev["recovery_s"]
    xlim, ylim = centred(frames, 10.5, 4.6, 3.4)
    r3.scene(ax, frames, 6.0, rec, rec, xlim, ylim, zlim=(0, 3.8), elev=34, azim=-62, ghost_objects=(8.3,),
             zoom=1.85, show_slots=False, size=3.5)
    bg.text(13.5, 3.0, "Follow → yield → recover (held-out seed)", ha="center", va="top", fontsize=8.5)
    img = mpimg.imread(OUT / "viewer" / "crossing_c_closest.png")
    ax2 = fig.add_axes([11.35 / 16, 0.95 / 6.3, 4.3 / 16, 1.85 / 6.3])
    ax2.imshow(img)
    ax2.set_axis_off()
    bg.text(13.5, 0.92, "SwarmFlow web flight laboratory", ha="center", va="top", fontsize=8.5)
    fig.savefig(OUT / "fig_system_overview.png")
    plt.close(fig)


def fig_curriculum():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    fig = plt.figure(figsize=(16, 6.6))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 16)
    bg.set_ylim(0, 6.6)
    bg.axis("off")
    blue = "#3a6fc4"
    stages = [("STAGE 1 · FORMATION KEEPING", ["hold", "follow", "turn"], "from random initialisation",
               "1 M PPO steps · lr $3\\times10^{-4}$\n320,000 labelled demonstrations"),
              ("STAGE 2 · MOVING OBSTACLES", ["crossing", "head_on"], "+ stage-1 scenarios kept",
               "2 M PPO steps · lr $1\\times10^{-4}$ · collision-state replay\n640,000 labelled demonstrations"),
              ("STAGE 3 · FULL BENCHMARK", ["double_crossing", "gust"], "+ stage-1 and stage-2 scenarios kept",
               "2 M PPO steps · lr $1\\times10^{-4}$ · collision-state replay\n640,000 labelled demonstrations")]
    titles = {s: t for s, t, *_ in THUMBS}
    specs = {s: (a, b, c, g) for s, _, a, b, c, g in THUMBS}
    x0, pw, gap = 0.15, 4.95, 0.35
    for n, (title, scen, keep, budget) in enumerate(stages):
        x = x0 + n * (pw + gap)
        bg.add_patch(FancyBboxPatch((x, 2.35), pw, 4.05, boxstyle="round,pad=0,rounding_size=0.18", fc="none",
                                    ec="0.25", lw=1.2, ls=(0, (5, 3))))
        bg.text(x + pw / 2, 6.18, title, ha="center", va="center", fontsize=11, weight="bold")
        tw = 1.6 if len(scen) == 3 else 2.2
        start = x + (pw - tw * len(scen)) / 2
        for k, s in enumerate(scen):
            ax = fig.add_axes([(start + k * tw) / 16, 3.55 / 6.6, tw / 16, 2.35 / 6.6], projection="3d")
            thumb(ax, s, *specs[s], zoom=1.6)
            bg.text(start + k * tw + tw / 2, 3.55, titles[s][4:], ha="center", va="top", fontsize=9)
        bg.text(x + pw / 2, 3.1, keep, ha="center", va="center", fontsize=9, style="italic", color="0.3")
        bg.text(x + pw / 2, 2.62, budget, ha="center", va="center", fontsize=8.8)
        if n:
            bg.add_patch(FancyArrowPatch((x - gap - 0.02, 4.4), (x + 0.02, 4.4), arrowstyle="-|>",
                                         mutation_scale=20, lw=3.0, color=blue))
            bg.text(x - gap / 2, 4.12, "best\n.pt", ha="center", va="top", fontsize=7.5, color=blue,
                    style="italic", linespacing=0.95)
    # timeline of one stage
    bg.text(0.2, 1.95, "Inside every stage:", fontsize=10.5, weight="bold", va="center")
    segs = [("Round 0: DART\nexpert flies with noise σ = 0.3", 2.3, "#fde5cc"),
            ("DAgger 1\npolicy flies,\nexpert labels", 1.5, "#fbd3a8"),
            ("DAgger 2", 0.9, "#f9c285"), ("DAgger 3", 0.9, "#f7b062"),
            ("critic\nwarm-up", 0.95, "#e4e4e4"),
            ("PPO fine-tuning: 8 envs × 256 steps × 8 drones = 16,384 samples per update;\n"
             "clipped surrogate + value loss + decaying imitation loss (β: 1.0 → 0.5)", 8.9, "#d3e8c8")]
    x = 0.2
    for label, w, color in segs:
        bg.add_patch(Rectangle((x, 0.75), w, 0.85, fc=color, ec="white", lw=1.5))
        bg.text(x + w / 2, 1.175, label, ha="center", va="center", fontsize=8.3)
        x += w
    ppo0 = 0.2 + sum(w for _, w, _ in segs[:-1])
    ppo_w = segs[-1][1]
    for k in range(11):
        xv = ppo0 + k * ppo_w / 10
        bg.plot(xv, 0.62, marker="D", ms=5, color="#2b6b2b" if k else "#7b5aa6", mec="white")
    bg.plot(ppo0 - 0.02, 0.62, marker="D", ms=5, color="#7b5aa6", mec="white")
    bg.text(ppo0 + ppo_w / 2, 0.35, "Diamonds: validation on selection seeds every 100 k steps; the best-scoring checkpoint "
            "is kept as best.pt (the post-imitation policy is validated first)", ha="center", va="center",
            fontsize=8.5, color="0.25")
    bg.text(0.2 + 2.3 + 1.65, 0.35, "15 epochs per round, Adam lr $5\\times10^{-4}$", ha="center", va="center", fontsize=8.5,
            color="0.25")
    fig.savefig(OUT / "fig_curriculum.png")
    plt.close(fig)


def fig_network():
    import paper_diagrams as pd
    pd.network_figure(OUT / "fig_network_architecture.png")


NAMES = {"crossing": fig_crossing_render, "network": fig_network, "observation": fig_observation,
         "overview": fig_overview, "curriculum": fig_curriculum, "headon": fig_headon_double_render}

if __name__ == "__main__":
    for name in (sys.argv[1:] or NAMES):
        NAMES[name]()
        print("done", name)
