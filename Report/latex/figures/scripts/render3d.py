"""Paper-style 3-D rendering of SwarmFlow episodes (quadrotor glyphs, dotted trails, shaded spheres)."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource, to_rgb

DRONE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#d4b300", "#a65628", "#f781bf"]
OBJECT_COLOR = "#e8756a"


def light_floor(ax, x0, x1, y0, y1, z=0.0, cell=1.0):
    xs = np.arange(np.floor(x0), np.ceil(x1) + cell, cell)
    ys = np.arange(np.floor(y0), np.ceil(y1) + cell, cell)
    polys, colors = [], []
    for i, x in enumerate(xs[:-1]):
        for j, y in enumerate(ys[:-1]):
            polys.append([(x, y, z), (x + cell, y, z), (x + cell, y + cell, z), (x, y + cell, z)])
            colors.append("#e9ecef" if (i + j) % 2 else "#d7dce1")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=colors, edgecolors="none", zorder=0))


def quad(ax, p, color, arm=0.30, rotor=0.15, zorder=10):
    """Quadrotor glyph: X-frame arms and four rotor discs in the horizontal plane."""
    p = np.asarray(p, float)
    d = arm / np.sqrt(2)
    tips = [p + np.array([sx * d, sy * d, 0.0]) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    for a, b in ((tips[0], tips[2]), (tips[1], tips[3])):
        ax.plot(*zip(a, b), color="0.15", lw=1.1, zorder=zorder)
    theta = np.linspace(0, 2 * np.pi, 18)
    discs = [[(t[0] + rotor * np.cos(k), t[1] + rotor * np.sin(k), t[2] + 0.01) for k in theta] for t in tips]
    ax.add_collection3d(Poly3DCollection(discs, facecolors=color, edgecolors="0.15", linewidths=0.4,
                                         alpha=0.95, zorder=zorder))
    ax.scatter(*p, color="0.1", s=6, depthshade=False, zorder=zorder + 1)


def sphere(ax, c, r, color=OBJECT_COLOR, alpha=0.9, zorder=5):
    u, v = np.mgrid[0:2 * np.pi:48j, 0:np.pi:24j]
    nx, ny, nz = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    x, y, z = c[0] + r * nx, c[1] + r * ny, c[2] + r * nz
    light = np.array([-0.5, -0.6, 0.62])
    light /= np.linalg.norm(light)
    diffuse = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, 1)
    spec = diffuse ** 24
    base = np.array(to_rgb(color))
    rgb = np.clip(base * (0.42 + 0.58 * diffuse[..., None]) + 0.35 * spec[..., None], 0, 1)
    rgba = np.concatenate([rgb, np.full((*z.shape, 1), alpha)], axis=-1)
    ax.plot_surface(x, y, z, facecolors=rgba, linewidth=0, antialiased=True, shade=False, zorder=zorder)
    ax.plot([c[0], c[0]], [c[1], c[1]], [0, c[2] - r], color="0.6", lw=0.6, ls=":", zorder=1)


def trails(ax, pos, step=3, size=5, colors=DRONE_COLORS, zorder=3):
    for i in range(pos.shape[1]):
        pts = pos[::step, i]
        sizes = np.linspace(0.6, 1.0, len(pts)) * size
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=sizes, color=colors[i % len(colors)], alpha=0.85,
                   depthshade=False, linewidths=0, zorder=zorder)


def shadow(ax, pos, colors=DRONE_COLORS):
    for i in range(pos.shape[1]):
        ax.plot(pos[:, i, 0], pos[:, i, 1], np.zeros(len(pos)), color=colors[i % len(colors)], lw=0.5, alpha=0.25)


def target(ax, p, zorder=12):
    ax.scatter(*p, marker="D", s=40, color="#f0b400", edgecolors="0.2", linewidths=0.6, depthshade=False,
               zorder=zorder)


def slots(ax, s, zorder=4):
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], marker="s", s=22, facecolors="none", edgecolors="0.35",
               linewidths=0.7, depthshade=False, zorder=zorder)


def setup(ax, xlim, ylim, zlim=(0, 4.5), elev=28, azim=-60, floor=True, zoom=1.55):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.computed_zorder = False
    if floor:
        light_floor(ax, xlim[0], xlim[1], ylim[0], ylim[1])


def arrays(frames):
    t = np.array([f["time"] for f in frames])
    pos = np.array([f["positions"] for f in frames])
    slot = np.array([f["slots"] for f in frames])
    tgt = np.array([f["target"] for f in frames])
    return t, pos, slot, tgt


def objects_at(frame):
    return [(np.array(o["position"]), o["radius"], np.array(o["velocity"])) for o in frame["obstacles"]]


def scene(ax, frames, t0, t1, when, xlim, ylim, zlim=(0, 4.5), elev=28, azim=-60, ghost_objects=(),
          show_slots=True, step=3, size=5, zoom=1.55):
    """Trails between t0 and t1, drones and objects drawn at time `when`."""
    t, pos, slot, tgt = arrays(frames)
    setup(ax, xlim, ylim, zlim, elev, azim, zoom=zoom)
    window = (t >= t0) & (t <= t1)
    inside = pos[window].copy()
    out = ((inside[..., 0] < xlim[0]) | (inside[..., 0] > xlim[1]) |
           (inside[..., 1] < ylim[0]) | (inside[..., 1] > ylim[1]))
    inside[out] = np.nan
    shadow(ax, inside)
    trails(ax, inside, step=step, size=size)
    for g in ghost_objects:
        k = int(np.argmin(np.abs(t - g)))
        for c, r, _ in objects_at(frames[k]):
            sphere(ax, c, r, alpha=0.18, zorder=2)
    k = int(np.argmin(np.abs(t - when)))
    for c, r, v in objects_at(frames[k]):
        sphere(ax, c, r)
        tip = c + 0.8 * v / max(np.linalg.norm(v), 1e-6)
        ax.quiver(*c, *(tip - c), color="0.2", lw=1.0, arrow_length_ratio=0.3, zorder=6)
    if show_slots:
        slots(ax, slot[k])
    for i in range(pos.shape[1]):
        quad(ax, pos[k, i], DRONE_COLORS[i % 8])
    target(ax, tgt[k])
    return k
