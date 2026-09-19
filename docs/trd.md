# TRD — swarm package (BTP1)

Technical requirements for the first implementation. Product rules: [prd.md](./prd.md). Words: [CONTEXT.md](../CONTEXT.md).

No product code until this file is accepted.

## Stack

- Python 3.11+, [uv](https://docs.astral.sh/uv/), Makefile.
- Installable package `swarm` via `src/` layout (`src/swarm/`).
- PyTorch for the policy. CUDA if `torch.cuda.is_available()`, else CPU.
- Training: one process group using **all CPU cores** as parallel `SwarmEnv` workers; the net on GPU when present.
- No Gymnasium env class in v1. `SwarmEnv.reset` / `step` are ours. A Gymnasium adapter is later, with ROS 2 / Gazebo-as-robot, not this delivery’s physics.
- Viewer is optional and never the integrator: `VIEW=python` or `VIEW=gazebo`. Physics always `swarm.dynamics`.
- Gazebo is **not** installed by this repo. `assets/worlds/swarm.sdf` + `assets/drone/model.sdf` are shipped. If `gz`/`gazebo` is missing, `VIEW=gazebo` logs and falls back to the Python viewer.
- Tests: pytest. Config: YAML.

## One policy

There is **one** actor-critic. It must pass Tests 1, 2, and 3.

- Round-robin: worker `i` runs test `(i % 3) + 1`.
- Checkpoints: `models/latest.pt` every save; `models/best.pt` only when **combined** pass-gate score over all three tests improves.
- Optional rollbacks: `models/snapshot_test1.pt` (same architecture, same net, a point in time). Not three specialists.
- `make view TEST=1|2|3 POLICY=rl` loads `models/best.pt` into different tasks.

## Folder layout

```
src/swarm/
  dynamics/      quadrotor rigid body, motors, limits
  driver/        flight cmd → motors (PX4 / micro-ROS stand-in)
  env/           SwarmEnv: N drones, arena, collisions, bounds
  obs/           local observation (self + neighbor_set)
  tasks/         test_static_target, test_geometry, test_moving_target
  gates/         pass gates (eval/demo)
  baseline/      PID / geometric → flight cmd
  policy/        actor-critic + deep-sets neighbor encoder
  train/         PPO+GAE loop, workers, checkpoint
  viewer/        python_view, gazebo_view (state in, no physics)
  cli.py         argparse entry
configs/
  default.yaml   N, K, dt, arena, gates
  train.yaml     PPO, reward weights, workers
assets/drone/    visual mesh only (license in the folder)
models/          *.pt gitignored
tests/           test_<module>.py mirroring src/swarm
docs/
```

Entry points (pyproject scripts + Makefile):

- `make help`
- `make test`
- `make sim` / `make sim-resume` — Python 3D simulation (train)
- `make train VIEW=0|python|gazebo`
- `make view` / `make view-pid` / `make view-train` / `make view-prod` — play Test 1–3 (`MODE=train|prod|baseline`).
- `make eval MODE=prod|train|baseline` — run Tests 1–3 headless and write `results/submit/`.

## Naming

| Kind | Rule | Example |
| --- | --- | --- |
| Package | `swarm` | `from swarm.env.swarm_env import SwarmEnv` |
| Files | snake_case, one idea | `shared_target.py`, not `utils.py` |
| Classes | PascalCase | `SwarmEnv`, `Quadrotor`, `PassGate`, `NeighborEncoder` |
| Functions / fields | snake_case | `neighbor_set`, `shared_target`, `slot_positions` |
| Constants | UPPER_SNAKE | `CONTROL_HZ` |
| Config keys | same as Python | `num_drones`, `neighbor_k` |
| Tests | `tests/test_<module>.py` | `tests/test_gates.py` |

**Do not use in identifiers:** `agent`, `goal`, `quads_`, `formation` (say `geometry` + `slot`), `btp1` as the import name.

## Sim clock and body

| Quantity | v1 |
| --- | --- |
| Control / integrate | 100 Hz, dt = 0.01 s |
| Episode | 15 s |
| Arena | 10 × 10 × 10 m |
| Default N, K | 4, 3 |
| Action | `[roll, pitch, yaw_rate, climb]` in `[-1, 1]`. Driver → motors. Hover is zeros. |
| Dynamics | 3D rigid body, gravity, motor/omega limits |
| Motor lag, IMU noise | YAML flags, **default off** |

Spawn: random position in a box above the floor, random yaw, small random vel.

Visual mesh in `assets/drone/` is **draw-only**. If missing, viewer draws box+arms. Training does not download.

## Observation and action

Self (18): position relative to **shared_target** (tests 1, 3) or **slot** (test 2); world velocity (3); rotation matrix (9); body rates (3).

Neighbor set: K closest by distance + approaching speed; each (rel_pos, rel_vel) = 6. Pad zeros if fewer than K. No drone ids. No full-N list.

Action: `[roll, pitch, yaw_rate, climb]` per drone. Same layout as PX4 `VehicleAttitudeSetpoint` / MAVROS `AttitudeTarget`. Driver mixes motors.

## Pass gates (eval / demo)

Training does **not** end the episode on collision. Collision is a reward penalty so the policy can recover.

Eval / `make view`:

- Any drone–drone **square-body** overlap at the same altitude, floor hit, or leaving the arena → fail. Altitude pass-through is not a collision.
- Hold window: last **5 s**.
- Test 1: each drone within **40 cm of the standoff ring** around the shared target (not the point — four square bodies cannot sit on one coordinate); speed **< 25 cm/s**; roll/pitch **< 20°**.
- Test 2: each drone within **15 cm** of its slot; same speed/tilt.
- Test 3: lag to the **standoff ring** around the moving shared target **< 50 cm**.

Numbers live in `configs/default.yaml`.

## Reward (train only)

Sum, weights in `configs/train.yaml`:

1. Negative distance cost in centimetres. Floor 10 cm (one body). `e = excess_cm / 10`, cost `e + e²` with no ceiling, plus receding velocity and progress `max(e_now - e_prev, 0)`. Not 1 cm accuracy — the body is 10 cm.
2. Collision penalty (drone–drone, floor).
3. Smooth proximity: same-altitude XY crowding before a hit, plus coplanar neighbours in the climb band (altitude pass-through).
4. Hover aux (omega, thrust, tilt) only inside **one body (10 cm)** of station, so a far drone is not rewarded for stalling.

Do not report reward as the demo result.

## Learning

- Pure RL: obs → flight cmd. No PID inside the policy.
- Baseline: separate PID/geometric controller, same obs, same cmd. `POLICY=baseline|rl`.
- PPO + GAE, shared actor-critic, batch = workers × N drones. Critic is fitted on the PID clone before actor PPO. A pack that worsens Test 1 or Test 2, or that collides, is discarded.
- Neighbor encoder: deep sets (embed each neighbor, mean-pool). Attention is a later flag.
- TensorBoard: reward, collisions, gate errors for tests 1–3.
- Workers: default `os.cpu_count()`. Device: `cuda` if available else `cpu`.

## Viewer

`VIEW=python` — Python 3D window (drones, shared_target, geometry, mesh).
`VIEW=gazebo` — Gazebo draws the same state stream. Requires a local Gazebo install; skip if missing (log error, do not crash train).
`VIEW=0` — headless.

At most one viewed env (env 0). Other workers stay headless.

## Config surface

`configs/default.yaml`: `num_drones`, `neighbor_k`, `control_hz`, `episode_s`, `arena_cm`, gate thresholds, `motor_lag`, `sensor_noise`, `yaw_coeff`, `inertia_diag`. Lengths and speeds are centimetres.

`configs/train.yaml`: PPO lr, gamma, GAE λ, minibatch, reward weights, `num_workers`, `device` (`auto|cpu|cuda`). `total_steps: 0` = run until Ctrl+C.

CLI overrides YAML. Seed is required for a logged run.

## Tests (pytest)

External behavior only:

- Hover thrust balances gravity (no NaNs).
- Collision flag when Chebyshev XY < 20 cm at the same altitude.
- Observation length = 18 + 6K; K-cap holds.
- Slot and shared_target placement for the three tasks.
- Gate math on synthetic trajectories.
- Baseline `step` returns 4 cmds in range.

## Out of this TRD

- Gymnasium class, ROS 2 workspace, Gazebo as physics. `rclpy` / `px4_msgs` wait until a board is plugged in.
- Motor lag / noise on by default.
- Attention encoder, mixed random test sampling (we use round-robin).
- Physical drones.
- Three specialist networks.
