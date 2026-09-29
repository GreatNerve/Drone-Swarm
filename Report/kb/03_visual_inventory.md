# 03 — Visual Inventory (Agent 3)

Rule (user): graphs/diagrams print-friendly (white); real work/demonstration = viewer screenshots in the
website theme, placed on a white page. All screenshots captured from the running viewer
(`swarmflow view --checkpoint models/previous-runs/run3-pipeline-v3/showcase/best.pt --port 8766`) with
headless Chrome (`figures/scripts/capture_viewer.py`), benchmark spacing 1.25 m, held-out seed 100000.
Viewer state at capture matched offline evaluation exactly (crossing: min clearance 0.298 m, recovery 1.51 s;
double crossing: slot RMSE 0.363 m, recoveries 1.51 s and 2.00 s). The viewer's amber "learning-note" banner was hidden.

| ID | File | Source | Section | Caption (short) |
|---|---|---|---|---|
| FIG-01 | fig_system_architecture.png | generated diagram (env.py, policy.py, train.py) | 3.1 | Overall architecture of the RL drone-swarm system |
| FIG-02 | fig_network_architecture.png | generated diagram (policy.py; params counted from checkpoint) | 3.6 | Shared actor-critic network with attention set encoders |
| FIG-03 | fig_software_architecture.png | generated diagram (src/swarmflow modules) | 4.1 | Software architecture |
| FIG-04 | fig_training_pipeline.png | generated diagram (curriculum.py, train.py; V3 budgets) | 4.2 | Curriculum training pipeline |
| FIG-05 | viewer/viewer_overview.png | viewer screenshot | 4.4 | SwarmFlow flight laboratory running the final model |
| FIG-06 | fig_viewer_crossing_sequence.png | 4 viewer frames, crossing seed 100000 | 5.2 | Follow, yield, recover sequence |
| FIG-07 | viewer/crossing_axis_views.png | viewer axis view at t = 8.3 s | 5.2 | Side/top/side projections at closest approach |
| FIG-08 | fig_viewer_headon_double.png | 3 viewer frames | 5.2 | Head-on and repeated-crossing demonstrations |
| FIG-09 | fig_slot_error_timeseries.png | offline trace of the final model, seed 100000 | 5.3 | Per-drone slot error over time |
| FIG-10 | fig_results_scenarios.png | models/previous-runs/run3-pipeline-v3/showcase/test_learned.json | 5.4 | Held-out success, formation error, recovery time |
| FIG-11 | fig_imitation_and_stage1.png | run3 history training.jsonl | 5.5 | Imitation convergence and stage-1 formation error |
| FIG-12 | viewer/model_training_charts.png | viewer "Did the model learn correctly?" panel, V3 history | 5.5 | Validation score, training failures, training slot error over the curriculum |

Not used: user screenshots `Screenshot_2026-09-28_at_2.11.*` (stage-3 best ≈0.86 → earlier version, not V3);
`Report/demo/Screenshot 2026-09-28 at 6.08/6.29 PM.png` (V3 mid-training, show terminal/browser tabs and
"predictive controller" text; superseded by FIG-12 captured with the complete V3 history);
`docs/flight-preview.png` (not the learned model). No manual screenshots required.
