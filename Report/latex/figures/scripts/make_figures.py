"""Data figures of the BTP report, computed from the final (V3) model and its saved logs.

Graphs are print-friendly; the composite figures place the real viewer screenshots (website theme)
on a white page. Run from the project root after capture_viewer.py:
    PYTHONPATH=src:/tmp/mplpkgs:Report/latex/figures/scripts .venv/bin/python Report/latex/figures/scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import theme as th
from swarmflow.evaluate import run_episode
from swarmflow.policy import LearnedController

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "models/previous-runs/run3-pipeline-v3"
CKPT = RUN / "showcase/best.pt"
OUT = ROOT / "Report/latex/figures"
VIEWER = OUT / "viewer"
th.apply_print()
C = th.PRINT_DRONES


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def fig_slot_error_timeseries(controller):
    cases = [("crossing", "(a) Crossing object"), ("head_on", "(b) Head-on object"),
             ("double_crossing", "(c) Repeated crossings")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), sharey=True)
    facts = {}
    for ax, (scenario, title) in zip(axes, cases):
        summary, frames = run_episode(controller.config, scenario, 100000, controller, trace=True)
        t = np.array([f["time"] for f in frames])
        err = np.array([np.linalg.norm(np.array(f["positions"]) - np.array(f["slots"]), axis=1) for f in frames])
        for e in summary["events"]:
            ax.axvspan(e["start_s"], e["clear_s"], color="orange", alpha=0.13, lw=0)
            if e["recovery_s"] is not None:
                ax.axvline(e["measured_from_s"] + e["recovery_s"], color="green", ls="--", lw=1)
        for i in range(err.shape[1]):
            ax.plot(t, err[:, i], color=C[i], lw=0.8, alpha=0.75)
        ax.plot(t, np.sqrt(np.mean(err ** 2, axis=1)), color="k", lw=1.8, label="RMS over 8 drones")
        ax.axhline(0.35, color="red", ls=":", lw=1, label="0.35 m recovery tolerance")
        ax.set_title(f"{title} (seed 100000)")
        ax.set_xlabel("time (s)")
        facts[scenario] = {k: summary[k] for k in ("failure", "slot_rmse_m", "min_surface_clearance_m", "events")}
    axes[0].set_ylabel("distance to own slot (m)")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.savefig(OUT / "fig_slot_error_timeseries.png")
    plt.close(fig)
    return facts


def fig_results():
    report = json.loads((RUN / "showcase/test_learned.json").read_text())
    order = ["hold", "follow", "turn", "gust", "crossing", "head_on", "double_crossing"]
    labels = ["hold", "follow", "turn", "gust", "crossing", "head-on", "double\ncrossing"]
    summ = report["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    succ = [100 * summ[s]["success_rate"] for s in order]
    bars = axes[0].bar(labels, succ, color="#2ca02c", width=0.62)
    for b, v in zip(bars, succ):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("episodes completed (%)")
    axes[0].set_title("(a) Success rate (20 unseen seeds each)")
    rmse = [summ[s]["mean_slot_rmse_m"] for s in order]
    ci = [summ[s]["slot_rmse_ci95_m"] for s in order]
    err = [[r - c["low"] for r, c in zip(rmse, ci)], [c["high"] - r for r, c in zip(rmse, ci)]]
    colors = ["#1f77b4"] * 3 + ["#6baed6"] * 4
    axes[1].bar(labels, rmse, yerr=err, capsize=3, color=colors, width=0.62)
    axes[1].hlines(0.35, -0.4, 2.4, color="red", ls=":", lw=1.3, label="0.35 m tolerance (undisturbed)")
    axes[1].set_ylabel("mean episode slot RMSE (m)")
    axes[1].set_title("(b) Formation error (95% CI)")
    axes[1].legend(fontsize=7.5, loc="upper left")
    data, names = [], []
    for s, lab in zip(("crossing", "head_on", "double_crossing"), ("crossing", "head-on", "double\ncrossing")):
        data.append([e["recovery_s"] for ep in report["episodes"] if ep["scenario"] == s and not ep["failure"]
                     for e in ep["events"] if e["recovery_s"] is not None])
        names.append(lab)
    axes[2].boxplot(data, tick_labels=names, widths=0.5)
    axes[2].set_ylabel("recovery time after object exit (s)")
    axes[2].set_title("(c) Recovery to own slots")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_results_scenarios.png")
    plt.close(fig)


def fig_imitation():
    hist = RUN / "showcase/history"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for s, marker in zip((1, 2, 3), ("o", "s", "^")):
        bc = [r for r in rows(hist / f"stage{s}-seed0/training.jsonl") if r["type"] == "behavior_cloning"]
        axes[0].plot([r["round"] for r in bc], [1000 * r["final_epoch_mse"] for r in bc], marker=marker,
                     color=th.PRINT_STAGES[s - 1], lw=1.5, label=f"Stage {s}")
    axes[0].set_xticks([0, 1, 2, 3])
    axes[0].set_xticklabels(["0 (DART)", "1", "2", "3"])
    axes[0].set_xlabel("demonstration round (DART, then DAgger)")
    axes[0].set_ylabel("action MSE (\u00d710\u207b\u00b3)")
    axes[0].set_title("(a) Imitation warm-start convergence")
    axes[0].legend(fontsize=8)
    ppo = [r for r in rows(hist / "stage1-seed0/training.jsonl") if r["type"] == "ppo" and r["recent_slot_rmse_m"] is not None]
    axes[1].plot([r["environment_steps"] / 1e6 for r in ppo], [r["recent_slot_rmse_m"] for r in ppo],
                 color=th.PRINT_STAGES[0], lw=1.6)
    axes[1].set_xlabel("PPO environment steps (millions)")
    axes[1].set_ylabel("slot RMSE, last 100 episodes (m)")
    axes[1].set_title("(b) Stage 1 formation error during PPO")
    fig.tight_layout()
    fig.savefig(OUT / "fig_imitation_and_stage1.png")
    plt.close(fig)


def composite(images, titles, path, cols, width):
    """Real viewer screenshots on a white page with printed sub-captions."""
    arrays = [mpimg.imread(VIEWER / name) for name in images]
    rows_n = int(np.ceil(len(arrays) / cols))
    h, w = arrays[0].shape[:2]
    fig, axes = plt.subplots(rows_n, cols, figsize=(width, width * rows_n * h / (w * cols) * 1.12))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title in zip(axes, arrays, titles):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=9.5, loc="left", pad=4)
    for ax in axes[len(arrays):]:
        ax.set_axis_off()
    fig.subplots_adjust(wspace=0.03, hspace=0.14)
    fig.savefig(OUT / path, facecolor="white")
    plt.close(fig)


def main():
    controller = LearnedController(CKPT)
    facts = fig_slot_error_timeseries(controller)
    fig_results()
    fig_imitation()
    composite(["crossing_a_formation.png", "crossing_b_approach.png", "crossing_c_closest.png", "crossing_d_recovered.png"],
              ["(a) t = 2.1 s: formation keeping", "(b) t = 7.0 s: object approaching",
               "(c) t = 8.3 s: yielding at closest approach", "(d) t = 14.0 s: every drone back in its own slot"],
              "fig_viewer_crossing_sequence.png", 2, 13)
    composite(["head_on_closest.png", "head_on_recovered.png", "double_crossing_second.png"],
              ["(a) Head-on object, t = 7.6 s", "(b) Head-on object, t = 13.0 s: recovered",
               "(c) Repeated crossings, t = 16.5 s"],
              "fig_viewer_headon_double.png", 3, 15)
    (OUT / "scripts/figure_facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps({k: {x: v[x] for x in ("failure", "slot_rmse_m", "min_surface_clearance_m")} for k, v in facts.items()}))


if __name__ == "__main__":
    main()
