# Graph Report - BTP1_v1  (2026-09-19)

## Corpus Check
- 65 files · ~17,687 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 343 nodes · 739 edges · 31 communities (16 shown, 15 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 50 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Agent session rules
- Decentralized policy
- loop.py
- ADRs
- no-push layout
- Module docs pattern
- TaskState
- cli.py
- SwarmEnv
- Quadrotor
- PythonView
- TRD — swarm package (BTP1)
- GeometricPid
- evaluate_episode
- build_obs
- baseline.md
- dynamics.md
- env.md
- gates.md
- obs.md
- policy.md
- tasks.md
- train.md
- viewer.md
- swarm
- load
- 0001 — Flight cmd is attitude, not motors
- report.md
- driver.md

## God Nodes (most connected - your core abstractions)
1. `SwarmEnv` - 42 edges
2. `PythonView` - 36 edges
3. `train()` - 25 edges
4. `load()` - 24 edges
5. `repo_root()` - 19 edges
6. `dump_pack()` - 17 edges
7. `TaskState` - 17 edges
8. `ActorCritic` - 16 edges
9. `_view()` - 15 edges
10. `GeometricPid` - 14 edges

## Surprising Connections (you probably didn't know these)
- `test_pid_warmstart_reduces_action_error()` --calls--> `load()`  [INFERRED]
  tests/test_policy.py → src/swarm/config.py
- `test_zero_cmd_is_hover_thrust()` --uses--> `Quadrotor`  [INFERRED]
  tests/test_driver.py → src/swarm/dynamics/quadrotor.py
- `test_baseline_cmds_in_range()` --uses--> `SwarmEnv`  [INFERRED]
  tests/test_baseline.py → src/swarm/env/swarm_env.py
- `test_pid_holds_standoff_around_shared_target()` --uses--> `SwarmEnv`  [INFERRED]
  tests/test_baseline.py → src/swarm/env/swarm_env.py
- `test_collision_flag_when_close()` --uses--> `SwarmEnv`  [INFERRED]
  tests/test_env.py → src/swarm/env/swarm_env.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Initial three tests** — docs_prd_prd_test1_static_target, docs_prd_prd_test2_geometry, docs_prd_prd_test3_moving_target [EXTRACTED 1.00]
- **Agent operating system** — agents_session_rules, agents_graphify_first [INFERRED 0.85]

## Communities (31 total, 15 thin omitted)

### Community 0 - "Agent session rules"
Cohesion: 0.25
Nodes (9): Graphify-first search, No AI authorship, Agent session rules, Shortest working diff, Loading and error states, Graphify query workflow, AGENTS.md wins, Code conventions (+1 more)

### Community 1 - "Decentralized policy"
Cohesion: 0.16
Nodes (15): Domain glossary, BTP1 overview, Baseline controller, Decentralized policy, Drone, Joystick target, Later work, Neighbor set (+7 more)

### Community 2 - "loop.py"
Cohesion: 0.09
Nodes (33): device, no_grad, Normal, ActorCritic, _atanh(), NeighborEncoder, tanh_logprob(), _append_metrics() (+25 more)

### Community 6 - "TaskState"
Cohesion: 0.14
Nodes (14): Task, TaskState, make_task(), _circle(), GeometryTask, ndarray, slot_positions(), MovingTargetTask (+6 more)

### Community 7 - "cli.py"
Cohesion: 0.11
Nodes (27): Namespace, _apply_live(), _eval(), _load_rl(), main(), _mode_policy_ckpt(), _view(), repo_root() (+19 more)

### Community 8 - "SwarmEnv"
Cohesion: 0.21
Nodes (4): ndarray, StepResult, SwarmEnv, _worker()

### Community 9 - "Quadrotor"
Cohesion: 0.20
Nodes (12): allocate(), _exp_so3(), hover_action(), mix_matrix(), _orthonormalize(), ndarray, Quadrotor, Rows: total thrust, roll, pitch, yaw. Cols: FR, RR, RL, FL. (+4 more)

### Community 10 - "PythonView"
Cohesion: 0.11
Nodes (6): _ensure_interactive(), _load_obj(), _motor_body(), ndarray, Path, PythonView

### Community 11 - "TRD — swarm package (BTP1)"
Cohesion: 0.11
Nodes (16): Architecture, Docs, Config surface, Folder layout, Learning, Naming, Observation and action, One policy (+8 more)

### Community 12 - "GeometricPid"
Cohesion: 0.12
Nodes (17): GeometricPid, ndarray, Track a ring of radius standoff around the aim, not the point itself., obs → flight cmd [roll, pitch, yaw_rate, climb]. Driver owns motors., _ring_error(), AttitudeDriver, _body_z(), cmd_scales() (+9 more)

### Community 13 - "evaluate_episode"
Cohesion: 0.36
Nodes (7): combined_score(), evaluate_episode(), _min_spacing(), ndarray, _tilt_deg(), _frame(), test_gate_pass_and_fail_on_synthetic()

### Community 14 - "build_obs"
Cohesion: 0.80
Nodes (4): build_obs(), neighbor_set(), ndarray, self_obs()

### Community 27 - "load"
Cohesion: 0.26
Nodes (11): Any, load(), Path, _read_yaml(), test_collision_flag_when_close(), test_collision_impulse_separates(), test_eval_episode_keeps_running_after_collision(), test_observation_length_and_k_cap() (+3 more)

### Community 28 - "0001 — Flight cmd is attitude, not motors"
Cohesion: 0.33
Nodes (5): 0001 — Flight cmd is attitude, not motors, Consequences, Context, Decision, Status

## Knowledge Gaps
- **43 isolated node(s):** `swarm`, `Docs`, `Status`, `Context`, `Decision` (+38 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SwarmEnv` connect `SwarmEnv` to `loop.py`, `TaskState`, `cli.py`, `Quadrotor`, `GeometricPid`, `load`?**
  _High betweenness centrality (0.128) - this node is a cross-community bridge._
- **Why does `PythonView` connect `PythonView` to `cli.py`?**
  _High betweenness centrality (0.125) - this node is a cross-community bridge._
- **Why does `load()` connect `load` to `loop.py`, `TaskState`, `cli.py`, `Quadrotor`, `GeometricPid`, `evaluate_episode`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `SwarmEnv` (e.g. with `_apply_live()` and `Quadrotor`) actually correct?**
  _`SwarmEnv` has 13 INFERRED edges - model-reasoned connections that need verification._
- **What connects `swarm`, `Docs`, `Status` to the rest of the system?**
  _43 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `loop.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09397163120567376 - nodes in this community are weakly interconnected._
- **Should `TaskState` be split into smaller, more focused modules?**
  _Cohesion score 0.1380952380952381 - nodes in this community are weakly interconnected._