# 01 — Project Knowledge Base (Agent 1: Project Intelligence)

Tags: [V] verified from code/output (file given), [D] derived, [P] proposed/future, [T] team-reported.
Project root: `BTP/Drone-Swarm` (== GitHub GreatNerve/Drone-Swarm @ nishant, HEAD bf4eb92).
Secondary source of truth (team context): `Report/kb/reference_context/` (Webrookie0/BEETEEPEE@sumitkr/context).

## USER REPORTING DIRECTIVES (binding for Agents 2-5)

1. Final system = **pipeline version 3 (V3)** model: `models/previous-runs/run3-pipeline-v3/showcase/best.pt`.
2. Do **not** mention earlier versions (V1, V2), BTP1_v1, early failed runs, or any development diary.
3. Do **not** mention "baseline", "hand-coded controller", comparisons against it, the slot-only ablation,
   or the safety-filter/shield experiment. No comparison tables against a controller.
4. V4: mention only that a revised version (V4) was trained on a separate team machine and errors were
   observed during its training phase; therefore V3 remains the final validated model. [T] + [V:
   reference_context/history.md L76-78: V4 stage 1 completed, best 0.996023; V4 stage 2 has no completion
   marker, best 0.963885 at its start].
5. Tone: positive, constructive; no negative judgments. Limitations -> "future scope".
6. Orchestrator integrity constraints (not negotiable): report real V3 numbers (138/140, not 140/140);
   training method must truthfully state the demonstration-guided (imitation / DAgger) initialisation
   followed by PPO fine-tuning. The expert that provides demonstrations is described as a
   "sampling-based model-predictive expert planner" (it is `PredictiveController`, baseline.py), without
   calling it a baseline and without comparing results to it.

## PROJECT_OVERVIEW

- Software name: **SwarmFlow** — "follow, yield, recover" [V: README.md, docs/research-plan.md].
- Task: N=8 drones follow a moving reference target at 3 m stand-off, each holding its own persistent
  formation slot; when a non-cooperative spherical object crosses the flight path the drones leave their
  slots, avoid the object and each other, then return to **their own** slot relative to the moving target.
  No capture/pursuit objective; target path is scripted [V: research-plan.md].
- Approach: one **shared, decentralised** neural policy (parameter sharing), each drone acting on its own
  local observation; trained by imitation warm start (DART + DAgger) and PPO fine-tuning in a 3-stage
  curriculum; learned above a stabilised velocity-level flight model [V: train.py, curriculum.py].
- Stack: Python 3.12, NumPy, Gymnasium (spaces), PettingZoo (`ParallelEnv`), PyTorch; browser viewer in
  plain HTML/JS Canvas2D served by Python `ThreadingHTTPServer` [V: pyproject.toml, server.py, web/].
  Env `.venv`: torch 2.14.0, numpy 2.5.3 [V].

## SYSTEM_WORKFLOW

Training: Config + scenario sampler -> `SwarmEnv` (8 parallel instances) -> local observations
(self 17, neighbours 6x8, obstacles 6x8) -> shared actor (set-attention encoder) -> tanh-Gaussian
velocity action -> stabilised flight model (5 substeps) + swept collision check -> per-drone reward ->
rollout buffer -> reward scaling + GAE -> PPO update (+ imitation term) -> periodic validation on
selection seeds -> `best.pt` -> next curriculum stage -> held-out test (showcase) [V: train.py, curriculum.py].

Inference: checkpoint -> `LearnedController` -> deterministic action tanh(mean) per drone ->
environment step -> viewer / JSON reports [V: policy.py L132-135, L186-204; evaluate.py].

## IMPLEMENTATION_DETAILS (modules)

| Module | Role |
|---|---|
| config.py | frozen `Config` dataclass, all defaults, validation; geometries grid/circle/square/triangle; worlds empty/hills/canyon/pillars |
| scenarios.py | 11 scenarios, target motion, crossing-object generator, formation offsets, terrain |
| env.py | `SwarmEnv(ParallelEnv)`: dynamics, observations, swept collisions, reward, recovery, metrics, state capture/restore |
| geometry.py | `limit_norm`, `swept_distance` |
| baseline.py | `PredictiveController` (expert planner used as demonstrator) |
| policy.py | `SetEncoder`, `Encoder`, `ActorCritic`, `LearnedController` |
| train.py | cloning (DART+DAgger), `Worker` (with crash-state replay), GAE, `RewardScaler`, PPO loop, validation |
| evaluate.py | episode runner, aggregation, `selection_score`, `learning_checks` |
| curriculum.py | `STAGES`, `COMMON`, resumable pipeline, showcase (held-out test), model report for viewer |
| server.py, web/ | local viewer + REST API |
| cli.py | `python -m swarmflow view / eval / train / pipeline / compare` |
| tests/ | 3 files, 40 tests |

## ENVIRONMENT_AND_DYNAMICS [V: config.py, env.py L307-390, docs/implementation.md]

- dt = 0.05 s (20 Hz), 5 substeps (h = 0.01 s), episode T = 24 s (480 steps).
- Drones N=8 (2-32 allowed), radius 0.18 m, spacing 1.25 m (grid 3x3 minus one), altitude 3 m,
  stand-off 3 m, target speed 0.55 m/s.
- Action a in [-1,1]^3 -> u = limit_3(3a) (desired velocity, |u|<=3 m/s).
- Per substep: a_cmd = limit_5((u - v)/0.28); v <- v + (a_cmd + w) h; p <- p + v h (semi-implicit Euler).
  w = gust (0, 1.2, 0.2) m/s^2 for 7<=t<9 s in `gust` only.
- Swept (continuous) sphere-sphere clearance on every substep, drone-drone and drone-object; gap<=0 =>
  collision; ground (z<=radius) and world bounds (|x|,|y|>=60-r, z>=10-r) also fail. Failure terminates
  the whole team; no position repair.
- Slots: s_i(t) = q(t) - (3,0,0) + delta_i, world-aligned, identity immutable within an episode.
- Target: q(t) = (2 + 0.55 t, 0, 3); `turn`: y = 1.8 sin(0.22t), z = 3 + 0.35 sin(0.3t); `hold`: static.
- Crossing object: sphere, constant velocity v_o = v_q(t0) + v_r d, starts D = max(5, extent + r + 2) m
  away, active 2D/v_r s. crossing: t0~U(3,5.5), angle pi/2±0.35, v_r~U(1.4,1.8), r~U(0.6,0.8);
  head_on: angle pi±0.2, v_r~U(1.8,2.2), r~U(0.7,0.9); double_crossing: second object 6-10 s later,
  opposite side [V: scenarios.py L156-200].
- Benchmark scenarios (7): hold, follow, turn, crossing, head_on, double_crossing, gust
  [V: curriculum.SHOWCASE_SCENARIOS]. Other scenarios (viewer-only): slalom, dense, pilot, terrain_follow.

## OBSERVATIONS [V: env.py L240-259, L189-203]

- self (17): slot error s_i-p_i (3), own velocity (3), target velocity (3), previous command (3),
  slot offset delta_i (3), altitude (1), remaining episode fraction (1).
- neighbours (6x8) and obstacles (6x8): nearest within 7 m sensor range, rows = relative position (3),
  relative velocity (3), radius (1), validity mask (1); zero-padded and masked.
- Fixed normalisation in the network: self / [5,5,5,3,...,3,5,5,5,10,1]; rows / [7,7,7,6,6,6,1.5]
  [V: policy.py L48, L80].
- No global state, no future paths, no other agents' goals.

## ACTIONS

Continuous 3-D normalised desired velocity, Box(-1,1,(3,)) [V: env.py L48-49].

## REWARD [V: env.py L360-409]

e = ||p_i - s_i||, e_c = min(e,5); C_slot = e_c^2 if e_c<=1 else 2e_c-1;
b = ||v_i - v_q||; P = max(0, 0.45 - g_nn)^2 (nearest drone surface gap);
D = sum over active objects max(0, 1 - g_o/1 m)^2;
C = C_slot + 0.12 min(b,3)^2 + 6P + 3 min(D,2) + 0.005||u||^2 + 0.001 min(||Δu||/dt, 20)^2;
r_i = dt (1 - C) - 100·1[failure and drone i involved].
All costs bounded => surviving dominates crashing [V: test_reward_is_bounded_and_failure_is_always_worse].

## RL_COMPONENTS

- Network (width 128) [V: policy.py; checkpoint]: self MLP 17->128 (tanh); two SetEncoders
  (per-row MLP 7->64->128 tanh, learned query, 4-head masked multi-head attention pooling ->128);
  trunk 384->256->128 (tanh); actor head 128->3 (orthogonal init gain 0.01); learnable log-std (3),
  clamp [-3, 1]; separate critic encoder of identical architecture + value head 128->1.
  **Parameters (run-3 checkpoint, counted): actor 284,166; critic 283,905; total 568,071.** [V]
- Distribution: Gaussian in pre-squash space, action = tanh(x); log-prob corrected:
  log pi(a) = sum[log N(x) - 2(log 2 - x - softplus(-2x))] [V: policy.py L157-160].
- Default critic: local (IPPO). Optional central critic exists (off) [V].
- PPO [V: train.py L480-543]: L = L_clip(eps=0.2) + 0.5·L_V (L_V = 0.5·MSE) + beta(f)·L_BC, entropy 0;
  beta decays linearly 1.0 -> 0.5 over the stage; L_BC = MSE(tanh(mu(o)), a_expert) on 1,024 expert samples;
  5 epochs, minibatch 2,048, KL early stop 0.03, Adam (eps 1e-5), lr annealed to 10% linearly,
  separate grad clipping 0.5 actor/critic; critic warm-up 10 updates (actor frozen).
- Advantages: GAE gamma 0.99, lambda 0.95; bootstrap at time limit, never across failure; normalised.
- Reward scaling: divide by running std of discounted return [V: RewardScaler].
- Rollout: 8 envs x 256 steps x 8 drones = 16,384 transitions/update.
- Crash-state replay (stages 2-3): restore snapshot ~2 s before a collision (up to 2x/episode) [V: Worker].
- Imitation (per stage): round 0 expert flies with N(0,0.3) action noise, clean labels (DART); rounds 1-3
  student flies, expert labels (DAgger); aggregate; 15 epochs per round, Adam lr 5e-4, batch 1,024,
  MSE loss [V: train.clone, _fit_clone].

## SWARM_COMPONENTS

Persistent slot assignment; grid formation (others available in viewer); local neighbour sensing within
7 m, up to 6 neighbours; permutation-invariant attention pooling; drone-drone proximity penalty inside
0.45 m; object warning zone 1 m; formation recovery rule (below).

## EXPERT PLANNER (demonstrator) [V: baseline.py]

Sampling-based receding-horizon planner on the same local observation: 86 candidate velocities
(nominal v_q + 1.8 e, 4 tracking gains, target velocity, current velocity, zero, 26 directions x 3 speeds),
each simulated 2 s ahead (0.1 s steps) with the same response model; cost = tracking + deviation from
nominal + change from previous + 180·(soft-margin violation)^2 + 20000·(hard-margin violation)^2 + altitude
guard; argmin. **Run-3 demonstrator margins 0.6 / 0.25 m** [V: run3 config.json teacher_margins].

## TRAINING_PIPELINE — exact V3 (run 3) settings [V: models/previous-runs/run3-pipeline-v3/stage*/config.json, finished.json]

| Stage | Scenarios | Imitation | PPO steps | lr | crash replay | PPO wall time |
|---|---|---|---|---|---|---|
| 1 Formation keeping | hold, follow, turn | 4 rounds x 10,000 env steps | 1,000,000 | 3e-4 | off | 1,925 s |
| 2 Moving obstacles | + crossing, head_on | 4 x 20,000 | 2,000,000 | 1e-4 | on | 3,904 s |
| 3 Full benchmark | + double_crossing, gust | 4 x 20,000 | 2,000,000 | 1e-4 | on | 3,999 s |

Common: 8 envs, horizon 256, 5 epochs, minibatch 2,048, 10 selection seeds/scenario, init log-std -1.6,
min log-std -3, entropy 0, KL 0.03, bc 1.0->0.5, width 128. Stage starts from previous stage's best.pt.
Validation every 100k steps; best.pt kept. Total PPO 5M env steps = 40M drone decisions [D].
Total PPO wall time ≈ 9,828 s ≈ 2.7 h (excludes imitation rounds) [D from finished.json].
Final model selected at lineage step 700,416 with validation score 0.957 (stage 3) [V: SUMMARY.md].
Best validation per stage: 0.997 / 0.968 / 0.957 [V].
Training machine: a teammate's laptop (branch `sumit`); CPU model not recorded [MISSING, not needed].

## INFERENCE_PIPELINE

`LearnedController(path)` loads config+weights; deterministic tanh(mean); no filter behind the network.
Viewer: `.venv/bin/python -m swarmflow view --checkpoint models/previous-runs/run3-pipeline-v3/showcase/best.pt --port 8766`
then http://127.0.0.1:8766. Eval CLI: `python -m swarmflow eval --policy learned --checkpoint <pt>`.
Viewer UI: controller/scenario/geometry/compaction/world/drones/seed selectors; Run/Reset; object speed and
radius sliders; "Send a crossing object"/"Send from ahead"; 3-D orbit/axis canvas with drones, target,
slot markers, objects, trails; live metrics (slot error, min clearance, recovered events, flight state);
slot-error timeline; "Did the model learn correctly?" panel (validation score, training failure rate,
training slot RMSE curves, acceptance checks, held-out table) [V: web/index.html].

## EVALUATION_PROTOCOL [V: train.py L18-21, evaluate.py, curriculum.py]

Seed ranges: training 0-999; selection 5000+; development 1000+; held-out test 100000-100019.
Selection score = mean success - 0.25 mean min(RMSE,2) - 0.25 (1 - recovered fraction).
Recovery: every drone within 0.35 m of its own slot and 0.35 m/s of target velocity, continuously 1 s,
measured after the object leaves the scene. Acceptance checks: zero failures; >=95% events recovered
within 5 s; slot RMSE < 0.35 m on hold/follow/turn.

## VERIFIED_RESULTS — V3 held-out test (7 scenarios x 20 unseen seeds = 140 episodes) [V: run3 showcase/test_learned.json, summary.json]

| Scenario | Success | Slot RMSE (m) | Events recovered | Mean / max recovery (s) | Min clearance (m) |
|---|---|---|---|---|---|
| hold | 20/20 | 0.023 | – | – | 0.764 |
| follow | 20/20 | 0.027 | – | – | 0.764 |
| turn | 20/20 | 0.045 | – | – | 0.764 |
| gust | 20/20 | 0.176 | – | – | 0.764 |
| crossing | 20/20 | 0.274 | 20/20 | 1.88 / 3.07 | 0.183 |
| head_on | 18/20 | 0.289 | 18/20 | 2.43 / 4.12 | −0.041 (contact, seeds 100007, 100009) |
| double_crossing | 20/20 | 0.393 | 40/40 | 3.47 / 11.36 | 0.009 |

Totals: 138/140 episodes failure-free (98.6%); 6 of 7 scenarios 100%; every event in completed episodes
recovered (78/78); 70/80 within 5 s; precision check PASS (0.023/0.027/0.045 m, i.e. < 5 cm).
Test score 0.936 (selection formula). Recovery time includes the 1 s confirmation hold.
Unit tests: 40 collected (tests/). [Orchestrator ran pytest separately — see 05_qa_audit.md.]

## V4 (team-reported + context evidence)

Code changes in V4 (commit c4f7249): demonstrator margins 0.7/0.3 m, overlap-aware recovery measurement,
stages 2-3 shortened to 1M PPO steps [V: curriculum.py]. Training run on a separate team machine: stage 1
completed (best validation 0.996), stage 2 did not complete; errors observed during the training phase
[T + V: reference_context/history.md L76-78]. Not used as final model.

## PROJECT_LIMITATIONS -> FUTURE SCOPE (phrase constructively)

Velocity-level stabilised model (no rotor/attitude/downwash physics); single training seed; 8 drones in
training; spherical obstacles; ideal sensing by default; scripted target; residual close passes in the
head-on case; complete V4 training; multi-seed training; generalisation tests (4/12/16 drones, noise,
dense obstacles, terrain worlds already in viewer); PyBullet (gym-pybullet-drones) validation; hardware.

## OPEN_QUESTIONS

- Supervisor designation: user says "Prof. Jyotsna Singh"; NSUT site could not be parsed (JS-rendered).
  Use "Prof. Jyotsna Singh, Department of Electronics and Communication Engineering" — flag for check.
- Department: "Department of Electronics and Communication Engineering" (UEC roll prefix + demo report of
  same university). Flag for check.
- Individual contributions of members: not required by demo structure; omit.

## SOURCE_REFERENCES

Code: src/swarmflow/*.py; outputs: models/previous-runs/run3-pipeline-v3/**; docs: knowledge.md,
docs/implementation.md, docs/research-plan.md; team context: Report/kb/reference_context/*.md.
