"""Paper-style network and overview diagrams (drawn neural layers, embedding bars, attention blocks)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

BLUE = ("#c9d6ee", "#4a6fb5")
GREEN = ("#d3e8c8", "#5b9a3c")
ORANGE = ("#f6d2b8", "#d9824a")
GRAY = ("#e4e4e4", "#8a8a8a")
PURPLE = ("#e3d7ef", "#7b5aa6")


def rbox(ax, x, y, w, h, colors, lw=1.6, ls="-", r=0.12, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=colors[0],
                                ec=colors[1], lw=lw, ls=ls, zorder=z))


def mlp(ax, x, y, w, h, layers, colors, title=None, title_size=11, edge_alpha=0.35):
    """Rounded box with a drawn fully-connected network inside."""
    rbox(ax, x, y, w, h, colors)
    if title:
        ax.text(x + w / 2, y + h + 0.08, title, ha="center", va="bottom", fontsize=title_size)
    xs = np.linspace(x + 0.22 * w, x + 0.78 * w, len(layers))
    cols = []
    for lx, n in zip(xs, layers):
        ys = np.linspace(y + h / 2 - (n - 1) * min(0.13, 0.8 * h / max(n, 2)) / 2,
                         y + h / 2 + (n - 1) * min(0.13, 0.8 * h / max(n, 2)) / 2, n)
        cols.append([(lx, yy) for yy in ys])
    for a, b in zip(cols[:-1], cols[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="0.35", lw=0.35, alpha=edge_alpha, zorder=2)
    rad = min(0.045, 0.3 * h / max(layers))
    for col in cols:
        for p in col:
            ax.add_patch(Circle(p, rad, fc="white", ec="0.3", lw=0.6, zorder=3))


def bar(ax, x, y, h, color, w=0.12, label=None, label_pos="top", size=11):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec="none", alpha=0.9, zorder=3))
    if label:
        ly = y + h + 0.05 if label_pos == "top" else y - 0.05
        ax.text(x + w / 2, ly, label, ha="left", va="bottom" if label_pos == "top" else "top", fontsize=size,
                style="italic")


def stack(ax, x, y, h, color, n=6, dx=0.07, dy=0.05, w=0.12):
    for k in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + k * dx, y + k * dy), w, h, fc=color, ec="white", lw=0.6, alpha=0.9, zorder=3))


def arrow(ax, x0, y0, x1, y1, lw=1.2, color="black", style="-|>", rad=0.0, z=4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=z,
                arrowprops=dict(arrowstyle=style, lw=lw, color=color, shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))


def elbow(ax, pts, lw=1.2, color="black"):
    for (a, b), (c, d) in zip(pts[:-2], pts[1:-1]):
        ax.plot([a, c], [b, d], color=color, lw=lw, zorder=4)
    arrow(ax, *pts[-2], *pts[-1], lw=lw, color=color)


def network_figure(path):
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(-0.6, 6.2)
    ax.axis("off")
    rows = {"self": 5.05, "neigh": 3.05, "obst": 1.05}
    labels = {"self": r"$o^{\,self}_i$" + "\n(17)", "neigh": r"$o^{\,nbr}_i$" + "\n(6 × 8)", "obst": r"$o^{\,obj}_i$" + "\n(6 × 8)"}
    for key, y in rows.items():
        ax.text(0.45, y, labels[key], ha="center", va="center", fontsize=13)
        arrow(ax, 0.95, y, 1.35, y)
    mlp(ax, 1.35, 4.5, 2.1, 1.1, [5, 7, 7], BLUE, "Self Encoder  (17→128)")
    mlp(ax, 1.35, 2.5, 2.1, 1.1, [4, 6, 7], GREEN, "Neighbour Row Encoder  (7→64→128)")
    mlp(ax, 1.35, 0.5, 2.1, 1.1, [4, 6, 7], ORANGE, "Object Row Encoder  (7→64→128)")
    ax.text(2.4, 2.3, "shared across the 6 rows", ha="center", va="top", fontsize=8.5, color="0.35")
    ax.text(2.4, 0.3, "shared across the 6 rows", ha="center", va="top", fontsize=8.5, color="0.35")
    # self embedding
    bar(ax, 6.55, 4.55, 1.0, BLUE[1], label=r"$e_{self}$")
    elbow(ax, [(3.45, 5.05), (6.55, 5.05)])
    # per-row embeddings and attention pooling
    for key, colors in (("neigh", GREEN), ("obst", ORANGE)):
        y = rows[key]
        stack(ax, 3.85, y - 0.5, 0.85, colors[1])
        arrow(ax, 3.45, y, 3.85, y)
        ax.text(4.1, y - 0.62, r"$\{h_j\}_{j=1}^{6}$", ha="center", va="top", fontsize=10.5)
        rbox(ax, 4.6, y - 0.55, 1.7, 1.1, GRAY, r=0.1)
        ax.text(5.45, y + 0.2, "Multi-Head\nAttention Pool", ha="center", va="center", fontsize=9.5)
        ax.text(5.45, y - 0.32, "4 heads · learned query\npadding mask", ha="center", va="center",
                fontsize=7, color="0.3", linespacing=1.1)
        arrow(ax, 4.4, y, 4.6, y)
        bar(ax, 6.55, y - 0.5, 1.0, colors[1], label=r"$\hat e_{nbr}$" if key == "neigh" else r"$\hat e_{obj}$",
            label_pos="top" if key == "neigh" else "bottom")
        arrow(ax, 6.3, y, 6.55, y)
    # concatenation
    cx, cy = 7.35, 3.05
    ax.add_patch(Circle((cx, cy), 0.13, fc="white", ec="black", lw=1.2, zorder=5))
    ax.plot([cx - 0.09, cx + 0.09], [cy, cy], color="black", lw=1.1, zorder=6)
    ax.plot([cx, cx], [cy - 0.09, cy + 0.09], color="black", lw=1.1, zorder=6)
    elbow(ax, [(6.67, 5.05), (cx, 5.05), (cx, cy + 0.13)])
    elbow(ax, [(6.67, 1.05), (cx, 1.05), (cx, cy - 0.13)])
    arrow(ax, 6.67, cy, cx - 0.13, cy)
    # concatenated vector
    for k, (col, frac) in enumerate(((ORANGE[1], 0), (GREEN[1], 1), (BLUE[1], 2))):
        ax.add_patch(Rectangle((7.85, 2.0 + 0.7 * frac), 0.14, 0.7, fc=col, ec="none", zorder=3))
    ax.text(7.92, 1.9, "384", ha="center", va="top", fontsize=9)
    arrow(ax, cx + 0.13, cy, 7.85, cy)
    # trunk
    mlp(ax, 8.4, 2.0, 1.9, 2.1, [12, 9, 6], GRAY, "Trunk MLP  (384→256→128)", title_size=10.5)
    arrow(ax, 7.99, cy, 8.4, cy)
    # actor head
    rbox(ax, 10.75, 2.55, 1.45, 1.0, PURPLE, r=0.1)
    ax.text(11.475, 3.2, "Actor head", ha="center", va="center", fontsize=10)
    ax.text(11.475, 2.85, r"$\mu_\theta(o_i),\ \sigma$", ha="center", va="center", fontsize=10.5)
    arrow(ax, 10.3, cy, 10.75, cy)
    arrow(ax, 12.2, cy, 12.65, cy)
    ax.text(12.72, cy + 0.05, r"$a_i=\tanh(x)$", ha="left", va="bottom", fontsize=12)
    ax.text(12.72, cy - 0.05, r"$x\sim\mathcal{N}(\mu,\sigma^2)$", ha="left", va="top", fontsize=10)
    # critic note
    rbox(ax, 8.2, -0.45, 4.6, 0.95, ("#ffffff", "#8a8a8a"), ls="--", r=0.1)
    ax.text(10.5, 0.17, "Critic (used only in training): separate encoder of the", ha="center", fontsize=9)
    ax.text(10.5, -0.17, r"same architecture $\rightarrow$ value head $V_\phi(o_i)$ (128 → 1)", ha="center", fontsize=9)
    ax.text(10.3, 5.55, "One set of weights θ shared by all 8 drones\n(actor 284,166 · critic 283,905 parameters)",
            ha="center", fontsize=9.5, color="0.25")
    fig.savefig(path)
    plt.close(fig)
