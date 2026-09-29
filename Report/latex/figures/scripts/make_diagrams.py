"""Print-friendly architecture and workflow diagrams, drawn from the implemented modules only.

    PYTHONPATH=/tmp/mplpkgs .venv/bin/python Report/latex/figures/scripts/make_diagrams.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "serif", "savefig.dpi": 220, "savefig.bbox": "tight"})
FILL = {"env": "#dbe9f6", "agent": "#fde5cc", "learn": "#dcefd6", "data": "#f0f0f0", "ui": "#ece0f3"}


def box(ax, x, y, w, h, text, kind="data", size=9, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=FILL[kind], ec="0.3", lw=1.0))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, weight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, text=None, size=8, rad=0.0, dx=0.0, dy=0.1, color="0.25"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=size,
                style="italic", color=color)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def legend(ax, x, y):
    for k, (kind, label) in enumerate([("env", "simulation"), ("agent", "policy / action"),
                                       ("learn", "learning"), ("data", "data"), ("ui", "outputs")]):
        ax.add_patch(FancyBboxPatch((x + k * 1.9, y), 0.3, 0.22, boxstyle="round,pad=0.01", fc=FILL[kind], ec="0.3"))
        ax.text(x + k * 1.9 + 0.4, y + 0.11, label, va="center", fontsize=7.5)


def system_architecture():
    fig, ax = canvas(12, 6.4)
    box(ax, 2.0, 5.4, 3.3, 1.0, "Scenario generator\nseeded target path, crossing /\nhead-on objects, wind gust", "env", 8.5)
    box(ax, 6.0, 5.4, 3.4, 1.0, "Multi-agent environment\nPettingZoo ParallelEnv · 8 drones", "env", 9, True)
    box(ax, 10.1, 5.4, 3.3, 1.0, "Stabilised flight model\n20 Hz, 5 sub-steps, τ = 0.28 s\n|a| ≤ 5 m/s², |u| ≤ 3 m/s", "env", 8.5)
    box(ax, 10.1, 3.45, 3.3, 1.0, "Swept collision check,\nslot error, recovery,\nper-drone reward", "env", 8.5)
    box(ax, 6.0, 3.45, 3.4, 1.1, "Local observation oᵢ\nself (17) · neighbours (6×8)\nobjects (6×8), masked", "data", 8.5)
    box(ax, 2.0, 3.45, 3.3, 1.1, "Shared policy network π_θ\n(one set of weights,\nexecuted by every drone)", "agent", 8.5, True)
    box(ax, 2.0, 1.35, 3.3, 1.0, "Action aᵢ ∈ [−1, 1]³\n→ desired velocity uᵢ", "agent", 8.5)
    box(ax, 6.0, 1.35, 3.4, 1.0, "Learning: imitation warm start\n+ PPO fine-tuning (Ch. 3)", "learn", 8.5)
    box(ax, 10.1, 1.35, 3.3, 1.0, "Outputs: trajectories, metrics,\ncheckpoints, web viewer", "ui", 8.5)
    arrow(ax, 3.65, 5.4, 4.3, 5.4)
    arrow(ax, 7.7, 5.4, 8.45, 5.4, "state")
    arrow(ax, 10.1, 4.9, 10.1, 3.95)
    arrow(ax, 8.45, 3.45, 7.7, 3.45, "oᵢ, rᵢ", dy=0.08)
    arrow(ax, 4.3, 3.45, 3.65, 3.45)
    arrow(ax, 2.0, 2.9, 2.0, 1.85)
    arrow(ax, 3.65, 1.6, 5.0, 4.9, "actions of all drones", rad=-0.25, dx=-0.55, dy=-0.3)
    arrow(ax, 6.0, 2.9, 6.0, 1.85, "rollouts", dx=0.45, dy=-0.05)
    arrow(ax, 4.3, 1.1, 3.0, 2.9, "θ update", rad=-0.35, dx=0.55, dy=-0.55, color="#2a7a2a")
    arrow(ax, 10.1, 2.95, 10.1, 1.85)
    legend(ax, 1.0, 0.2)
    fig.savefig(OUT / "fig_system_architecture.png")
    plt.close(fig)


def network_architecture():
    fig, ax = canvas(12, 5.4)
    box(ax, 1.3, 4.5, 2.3, 0.85, "self (17)\nslot error, velocity,\ntarget velocity, …", "data", 8)
    box(ax, 1.3, 2.75, 2.3, 0.85, "neighbours (6×8)\nrel. position, rel. velocity,\nradius, mask", "data", 8)
    box(ax, 1.3, 1.0, 2.3, 0.85, "objects (6×8)\nrel. position, rel. velocity,\nradius, mask", "data", 8)
    box(ax, 4.3, 4.5, 2.5, 0.85, "normalise\nLinear 17→128, tanh", "agent", 8)
    box(ax, 4.3, 2.75, 2.5, 1.05, "Set encoder\nrow MLP 7→64→128 (tanh)\nlearned-query attention\n4 heads, padding mask", "agent", 7.5)
    box(ax, 4.3, 1.0, 2.5, 1.05, "Set encoder\n(same structure,\nseparate weights)", "agent", 7.5)
    box(ax, 7.3, 2.75, 2.3, 1.25, "concatenate (384)\nLinear 384→256, tanh\nLinear 256→128, tanh", "agent", 8)
    box(ax, 10.4, 3.85, 2.6, 1.05, "Actor head 128→3: μ(o)\nlearnable log σ\na = tanh(x), x ~ N(μ, σ²)", "agent", 8)
    box(ax, 10.4, 1.55, 2.6, 1.05, "Critic: separate encoder\n(same architecture)\nvalue head 128→1: V(o)", "learn", 8)
    for y in (4.5, 2.75, 1.0):
        arrow(ax, 2.45, y, 3.05, y)
    arrow(ax, 5.55, 4.5, 6.15, 3.15)
    arrow(ax, 5.55, 2.75, 6.15, 2.75)
    arrow(ax, 5.55, 1.0, 6.15, 2.35)
    arrow(ax, 8.45, 3.05, 9.1, 3.7)
    ax.text(10.4, 0.55, "Actor: 284,166 parameters  ·  Critic: 283,905 parameters", ha="center", fontsize=8)
    fig.savefig(OUT / "fig_network_architecture.png")
    plt.close(fig)


def training_pipeline():
    fig, ax = canvas(12.6, 6.2)
    stages = [("Stage 1: formation keeping\nhold · follow · turn (random init.)\n1 M PPO steps, lr 3×10⁻⁴", 2.2),
              ("Stage 2: moving obstacles\n+ crossing · head-on\n2 M PPO steps, lr 1×10⁻⁴", 6.3),
              ("Stage 3: full benchmark\n+ repeated crossings · gust\n2 M PPO steps, lr 1×10⁻⁴", 10.4)]
    for i, (text, x) in enumerate(stages):
        box(ax, x, 5.4, 3.4, 1.0, text, "learn", 8.5)
        if i:
            arrow(ax, stages[i - 1][1] + 1.7, 5.4, x - 1.7, 5.4, "best.pt", dy=0.06)
    ax.text(6.3, 4.45, "Inside every stage", ha="center", fontsize=9.5, weight="bold")
    steps = [("Round 0 (DART)\nexpert planner flies\nwith action noise σ = 0.3;\nclean labels", 1.25, 2.2),
             ("Rounds 1–3 (DAgger)\npolicy flies, expert\nlabels visited states;\nrefit on all data", 3.7, 2.2),
             ("Validate start on\nselection seeds\n5000+ → best.pt", 6.05, 2.0),
             ("Critic warm-up\n10 updates,\nactor frozen", 8.2, 1.9),
             ("PPO loop\n8 envs × 256 steps\n× 8 drones = 16,384\nsamples per update", 10.7, 2.4)]
    for i, (text, x, w) in enumerate(steps):
        box(ax, x, 3.3, w, 1.4, text, "agent" if i < 2 else "learn", 8)
        if i:
            px, pw = steps[i - 1][1], steps[i - 1][2]
            arrow(ax, px + pw / 2, 3.3, x - w / 2, 3.3)
    box(ax, 10.7, 1.15, 3.1, 1.3, "Update: clipped PPO loss\n+ value loss + β·imitation loss\n(β: 1.0 → 0.5), KL stop 0.03,\ngradient clip 0.5", "learn", 8)
    box(ax, 6.3, 1.15, 3.1, 1.0, "Every 100 k steps: validate;\nkeep best.pt if score improves", "data", 8)
    box(ax, 2.0, 1.15, 3.2, 1.0, "After stage 3: held-out test\n7 scenarios × 20 unseen seeds", "ui", 8)
    arrow(ax, 10.7, 2.6, 10.7, 1.8)
    arrow(ax, 9.15, 1.15, 7.85, 1.15)
    arrow(ax, 4.75, 1.15, 3.6, 1.15)
    arrow(ax, 12.25, 1.3, 11.95, 2.6, "repeat", rad=0.4, dx=0.35, dy=-0.1)
    fig.savefig(OUT / "fig_training_pipeline.png")
    plt.close(fig)


def software_architecture():
    fig, ax = canvas(12, 5.0)
    box(ax, 1.9, 4.0, 3.0, 0.95, "config.py · scenarios.py\ndefaults, 11 scenarios,\nformations, worlds", "env", 8)
    box(ax, 5.8, 4.0, 3.1, 0.95, "env.py · SwarmEnv\ndynamics, sensing, reward,\ncollisions, recovery", "env", 8)
    box(ax, 9.8, 4.0, 3.1, 0.95, "policy.py\nset encoders, actor–critic,\nLearnedController", "agent", 8)
    box(ax, 1.9, 2.2, 3.0, 0.95, "train.py\nimitation, rollouts, GAE,\nPPO, validation", "learn", 8)
    box(ax, 5.8, 2.2, 3.1, 0.95, "curriculum.py\n3 stages, resumable\npipeline, held-out test", "learn", 8)
    box(ax, 9.8, 2.2, 3.1, 0.95, "evaluate.py · suite.py\nmetrics, scores,\nacceptance checks", "data", 8)
    box(ax, 3.8, 0.6, 3.6, 0.8, "cli.py: view · eval · train · pipeline", "ui", 8)
    box(ax, 8.6, 0.6, 4.4, 0.8, "server.py + web/ (HTML, JavaScript canvas)\nlocalhost REST API", "ui", 8)
    arrow(ax, 3.4, 4.0, 4.25, 4.0)
    arrow(ax, 7.35, 4.0, 8.25, 4.0)
    arrow(ax, 1.9, 3.52, 1.9, 2.68)
    arrow(ax, 3.4, 2.2, 4.25, 2.2)
    arrow(ax, 7.35, 2.2, 8.25, 2.2)
    arrow(ax, 5.8, 1.72, 4.8, 1.0)
    arrow(ax, 9.8, 1.72, 9.2, 1.0)
    arrow(ax, 9.8, 3.52, 9.8, 2.68)
    ax.text(6.0, 0.0, "Libraries: NumPy · Gymnasium spaces · PettingZoo · PyTorch", ha="center", fontsize=8,
            style="italic")
    fig.savefig(OUT / "fig_software_architecture.png")
    plt.close(fig)


if __name__ == "__main__":
    system_architecture()
    network_architecture()
    training_pipeline()
    software_architecture()
    print("diagrams written to", OUT)
