# viewer

Draw-only. Never steps physics.

`VIEW=python` matplotlib: square collision body + X-frame (OBJ is not drawn live — `plot_trisurf` was the slow path). Missing mesh logs an error and keeps the X-frame.

`make view-pid TEST=1` is the hover demo: PID holds a standoff ring around the shared target. Small motion on that ring is expected; flying away is not.

`make view-train` loads `models/latest.pt` (untrained / in-progress). Test 1 will drift until that net actually hovers. `make view-prod` loads `models/best.pt`.

Demo: drag the red shared target on the **right** panel (Tests 1 and 3; scroll wheel changes z). Test 2: click `[S] square` / `[R] rectangle` / `[C] circle` or press `1`/`2`/`3`. Distances are to **slots**, not the red center. Close the window to stop — a pack is written to `results/submit/`. The demo loops the 15 s episode. Each episode also writes `no-push/runs/` (last 3 kept).

CLI `make view` runs in real time so motion matches a demo, not a time-lapse.

`VIEW=gazebo` launches `assets/worlds/swarm.sdf` if `gz` or `gazebo` is on PATH; otherwise falls back to python.
