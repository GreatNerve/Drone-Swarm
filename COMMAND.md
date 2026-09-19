# Commands

What each command **actually** does, and the order to run this repo on a local machine. Product meaning lives in `docs/`. This file is only how to install, watch, train, and score.

Needs: **Python ≥ 3.11**, [uv](https://docs.astral.sh/uv/), `make` (Git Bash on Windows is fine). GPU is optional (`device: auto` uses CUDA if present).

---

## 0. First-time install

From the repo root (`BTP1_v1`):

```bash
uv sync --extra dev
make test
```

`uv sync` creates `.venv` and installs numpy, torch, pyyaml, matplotlib, tensorboard, pytest. After that, prefer `make …` or `uv run python -m swarm …` so the venv is used.

`make test` is **pytest**, not Test 1 / 2 / 3. It must pass before you trust a demo.

---

## 1. Sequence to run locally

Do these in order. Do not skip to `sim-resume` on a net that already sits on the wall.

| Step | Command | Why |
| --- | --- | --- |
| 1 | `uv sync --extra dev` | Install deps once |
| 2 | `make test` | Code is healthy |
| 3 | `make view-pid TEST=1` | PID hover around the shared target. If this looks wrong, the sim is wrong — do not train |
| 4 | `make view-pid TEST=2` | PID on square / rectangle / circle slots (`S`/`R`/`C` or keys `1`/`2`/`3`). `SPEED=8` if the window feels slow |
| 5 | `make view-pid TEST=3` | PID tracks the moving red target (3D path). **Right panel:** drag = x/y, **scroll wheel = z** |
| 6 | `make eval MODE=baseline` | Headless score of that same PID on Tests 1–3 |
| 7 | `make sim` | **Fresh** RL train: PID clone → critic fit → PPO. Window is noisy collect. **Ctrl+C** to stop |
| 8 | `make view-train TEST=1` | Clean look at `models/latest.pt` (no train noise) |
| 9 | `make eval MODE=train` | Headless score of that net |
| 10 | `make view-prod` / `make eval MODE=prod` | Only if `models/best.pt` exists and gates improved |

**Do not** `make sim-resume` unless the last `t1=` in the terminal / `no-push/runs/` is well under ~1 m. Resume of a ~7 m wall net will not recover.

---

## 2. Make targets (truth)

### `make help`

Prints this list and current `SEED` / `TEST` / `MODE` values.

### `make test`

Runs `uv run pytest -q`. Unit tests only.

### `make view-pid`  (`TEST=1|2|3`, default 1)

Live window. Pilot = **geometric PID**, not a neural net. This is the **baseline**. Same physics and tests as RL.

Default `SPEED=4` (4× sim time). Keys `+` / `-` change it; `0` is 1×. Closing the window stops the demo and is **not** a gate fail.

Writes `results/submit/` and a run log under `no-push/runs/` (last 3 files).

### `make view`

Same window. Default `POLICY=baseline`, so it is PID unless you set `POLICY=rl`. Prefer `view-pid` / `view-train` / `view-prod` so the pilot is explicit.

### `make view-train`

Live window. Pilot = **`models/latest.pt`** (train dump). Deterministic: this is “does the current net hover?”, not the noisy train collect.

Needs `latest.pt`. If that file is a wall-hugger, this will drift. Check `no-push/runs/` (`last_err_cm` = centimetres to target or slot). Old `.pt` files trained in metres are invalid after the cm plant — start over with `make sim`.

### `make view-prod`

Live window. Pilot = **`models/best.pt`** (best gate score so far). That is the learned policy you would show if training actually beat the clone.

### `make eval`  (`MODE=prod|train|baseline`, default `prod`)

**No window.** Runs Tests 1–3 headless and prints pass/fail + error in centimetres. Writes `results/submit/` and `no-push/runs/`.

| `MODE` | Who flies |
| --- | --- |
| `baseline` | PID |
| `train` | `latest.pt` |
| `prod` | `best.pt` |

Pass gates (not reward): no collision, in arena, hold error under the Test 1/2/3 caps. Reward is training-only.

### `make sim`

Train **with** the Python 3D window, from scratch (unless you pass resume). Sequence inside: clone PID cmds into the actor → fit critic → PPO. Collect is **noisy**. Terminal `t1=` is Test 1 distance in centimetres (lower is better; ~40 cm is the pass band).

A PPO pack that **worsens** Test 1 or Test 2, or that collides, is rejected; `latest.pt` is only written when both hold. **Ctrl+C** saves the kept net.

The train window cycles **Test 1 / 2 / 3** at random every ~8 s (noisy). After stop, `make view-train TEST=2`.

This is **not** 10 minutes of guaranteed improvement. If `t1` stays hundreds of centimetres, stop; do not resume that file.

### `make sim-resume`

Same as `make sim` but **`--resume`**: loads `models/latest.pt` and continues PPO. Skips the PID clone. Only resume a net that was trained after the centimetre plant.

### `make train`

Headless train (`VIEW=0`). Same loop as `sim` without the window. Faster. `RESUME=1` to continue.

```bash
make train SEED=0
make train SEED=0 RESUME=1
```

---

## 3. Tests 1 / 2 / 3 vs `make test`

| Name | What it is |
| --- | --- |
| `make test` | pytest |
| Test 1 | Shared target + standoff, no collisions |
| Test 2 | Geometry slots (square / rectangle / circle) |
| Test 3 | Moving shared target (train: scripted; demo: drag + wheel) |

```bash
make view-pid TEST=1
make view-pid TEST=2 SPEED=8
make view-pid TEST=3
make view-train TEST=2
```

---

## 4. Equivalent `python -m swarm` (if you skip Make)

```bash
uv run python -m swarm view --policy baseline --test 1 --seed 0 --view python
uv run python -m swarm view --mode train --test 1 --seed 0 --view python
uv run python -m swarm view --mode prod --test 1 --seed 0 --view python
uv run python -m swarm eval --mode baseline --seed 0 --view 0
uv run python -m swarm train --view python --seed 0
uv run python -m swarm train --view python --seed 0 --resume
uv run python -m swarm train --view 0 --seed 0
```

---

## 5. Files these commands write

| Path | What |
| --- | --- |
| `models/latest.pt` | Current train net (kept only if Test 1 did not get worse) |
| `models/best.pt` | Best combined gate score |
| `no-push/runs/run_*.json` | Last **3** view / eval / train-eval logs (`last_err_cm` to target or slot) |
| `results/submit/` | Pack to zip: gates, images, graphs, trajectory |
| `runs/seed0/metrics.csv` | Train scalars (appended; `t1_err` / `t2_err` / `t3_err` in centimetres) |

Never stage `no-push/` or `.env`.

---

## 6. Viewer — what to do on each test

Use the **right** map. The left 3D view is look-only (you can orbit it; it does not move the target).

| Test | What to do | What should happen |
| --- | --- | --- |
| 1 | Drag the red dot anywhere on the right map. Scroll to change height. | Drones slide with it and stay on a ring around it, not on the point. |
| 2 | Drag the red dot to slide the whole shape. Click `[S]`/`[R]`/`[C]` or press `1`/`2`/`3`. Scroll to lift the slots. | Drones snap to the green triangles (slots). Do not judge distance to the red center. |
| 3 | Click the right map once to take over the scripted path, then drag. Scroll for height. Try a long left-right pull, then a diagonal, then wheel up/down. | Drones keep the ring and follow. Clicking is what stops the automatic 3D path. |

- `+` / `-` change view speed (default 4×). `0` is real time. Slow to `1×` if the red dot feels hard to grab.
- Close the window to stop. That is not a failed test.

The train window (`make sim`) is exploration noise. Judge the net with `make view-train`, not that window.
