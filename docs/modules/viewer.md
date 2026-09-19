# viewer

Draw-only. Never steps physics.

`VIEW=python` matplotlib: square collision body + X-frame (OBJ is not drawn live — `plot_trisurf` was the slow path). Missing mesh logs an error and keeps the X-frame.

`make view-pid TEST=1` is the hover demo: PID holds a standoff ring around the shared target. Small motion on that ring is expected; flying away is not.

`make view-train` loads `models/latest.pt` (untrained / in-progress). Test 1 will drift until that net actually hovers. `make view-prod` loads `models/best.pt`.

Demo: drag the red target on the **right** map (not the 3D view). Scroll wheel on that map changes **z**. Test 1: move the shared target; drones keep a standoff ring. Test 2: drag moves the whole geometry; `[S]`/`[R]`/`[C]` or keys `1`/`2`/`3` change the shape; distances are to **slots**. Test 3: click once to take over the scripted path, then drag and scroll; drones keep the standoff ring. Close the window to stop — a pack is written to `results/submit/`. Closing early is not a gate fail. The demo loops the 15 s episode. Each episode also writes `no-push/runs/` (last 3 kept).

`SPEED=4` (default) runs the viewer at 4× sim time. Keys `+` / `-` double or halve; `0` is 1×. `make view-pid TEST=2 SPEED=8`.

`VIEW=gazebo` launches `assets/worlds/swarm.sdf` if `gz` or `gazebo` is on PATH; otherwise falls back to python.
