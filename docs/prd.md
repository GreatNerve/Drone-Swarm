# PRD — Decentralized swarm control (BTP1)

Product requirements. Domain words: [CONTEXT.md](../CONTEXT.md). Numbers that belong in the TRD (thresholds, dt, network sizes) are not frozen here.

## Problem Statement

We need a showable B.Tech research demo of **decentralized** drone-swarm control learned with reinforcement learning — not a single drone, and not a central computer steering everyone.

The first demo must be repeatable: at least four quadrotors in 3D simulation, configurable N, no drone–drone collisions, three tests that get harder in a clear order. Training will iterate; the tests and pass gates stay fixed so we can tell when a run is actually better.

Without a non-learning baseline on the same tests, there is nothing to compare the policy to, and nothing to put on screen if training is still running.

## Solution

A 3D quadrotor swarm simulator with one **decentralized policy** copied to every **drone**. Each drone sees only itself, a **neighbor set** of K nearest drones, and the task spec (relative **shared target** or its **slot**). Each drone outputs only its own flight command (roll, pitch, yaw rate, climb).

Three tests, empty arena for the initial demo:

1. **Static shared target** — swarm hovers around one point with **standoff**, no collisions.
2. **Static geometry** — square / rectangle / circle; v1 **slots** are pre-assigned; hold the shape.
3. **Moving shared target** — Test 1 plus a moving target. Train on a scripted path; demo with a **joystick target**.

A **baseline controller** runs the same tests and the same **pass gates**. Wind, obstacles, moving geometry, unlabeled vertices, and physical drones are later work, not this delivery.

## User Stories

1. As a researcher, I want N ≥ 4 drones in one arena with N set in config, so the same tests scale without a rewrite.
2. As a researcher, I want every drone identical in body, observation layout, and policy, so the swarm is homogeneous.
3. As a researcher, I want one shared policy copied to each drone, so control is decentralized at run time.
4. As a researcher, I want no commander process that outputs all N actions, so the title of the project stays true.
5. As a researcher, I want each drone to observe only its own state plus K nearest neighbors (relative), so it does not rely on global swarm state.
6. As a researcher, I want K configurable (default 3 when N=4), so we can try poorer or richer local views later.
7. As a researcher, I want each drone to output a flight command for its own quadrotor, so the policy is end-to-end on the body we chose for v1.
8. As a researcher, I want 3D quadrotor dynamics from the first demo, so the tests are not a 2D toy we throw away.
9. As an evaluator, I want Test 1 to use a single static shared target, so I can see station-keeping and collision avoidance without a prescribed shape.
10. As an evaluator, I want Test 1 drones to hover around that target with standoff, so they do not stack on one point.
11. As an evaluator, I want Test 2 to offer square, rectangle, and circle geometries, so the formation claim is not one hardcoded square.
12. As an evaluator, I want Test 2 v1 to pre-assign slots, so the first demo is solvable without unlabeled assignment.
13. As a researcher, I want unlabeled self-arrangement left as later work, so v1 does not block on a harder research problem.
14. As an evaluator, I want the first Test 2 to be a static geometry, so holding slots is visible before the shape itself moves.
15. As a researcher, I want a later Test 2 iteration where the geometry translates/rotates and slots are tracked, so formation-following is in the roadmap.
16. As an evaluator, I want Test 3 to be Test 1 with a moving shared target, so tracking is an extension, not a new product.
17. As a demo operator, I want to move the Test 3 target live with a joystick, so the live show is interactive.
18. As a researcher, I want Test 3 training to use a scripted or noisy moving target, so thousands of episodes do not need a human.
19. As a researcher, I want the joystick demo to load the same tracking policy, so we do not train a separate “demo brain.”
20. As an evaluator, I want the initial demo arena empty (no obstacles, no wind), so the first video is about swarm coordination.
21. As a researcher, I want obstacles defined as bodies to avoid, added after the three tests pass, so the env can grow without changing Tests 1–2.
22. As a researcher, I want wind defined as an external factor (force disturbance), off by default, so robustness is a later toggle.
23. As an evaluator, I want any drone–drone collision to fail the episode under the pass gates, so “no collision” is not optional.
24. As an evaluator, I want leaving the arena to fail the episode, so drones cannot succeed by flying away.
25. As an evaluator, I want Test 1 pass to require error to the shared target under a threshold for a hold time, so a brief fly-by does not count.
26. As an evaluator, I want Test 2 pass to require each drone within a threshold of its slot for a hold time, so the shape is actually held.
27. As an evaluator, I want Test 3 pass to require tracking error under a cap while the target moves, so lag is measured.
28. As a researcher, I want reward used only for training, so we do not report “mean reward” as the demo result.
29. As an evaluator, I want a baseline controller on all three tests with the same gates, so there is a non-RL comparison.
30. As a demo operator, I want to run each test with either baseline or learned policy, so I can show something before training finishes.
31. As a researcher, I want both controllers to stay in the project, so the report can compare them.
32. As a demo operator, I want one command (or one viewer action) per test, with policy = baseline | learned, so a viva demo is short.
33. As a demo operator, I want a 3D viewer of the swarm, target, and geometry, so the tests are watchable.
34. As a demo operator, I want to change N, K, geometry kind, geometry size, seed, and policy without editing source, so a live demo can be retuned.
35. As a researcher, I want random feasible spawns at episode start, so the policy cannot memorize one start pose.
36. As a researcher, I want the neighbor set computed from true positions (sensing, not a radio protocol), so v1 has no message channel.
37. As a researcher, I want no explicit inter-drone messages in v1, so “decentralized” means local observation plus a shared policy.
38. As a researcher, I want training in one simulator with all N drones present, so we can learn from the joint episode while executing decentrally.
39. As a researcher, I want on-policy learning with shared weights, so every drone improves the same policy.
40. As a researcher, I want to iterate training against fixed tests and gates, so progress is visible run to run.
41. As an evaluator, I want logged gate results (pass/fail and the raw errors), so the report has numbers, not only a video.
42. As a researcher, I want config and logs enough to reproduce a run (seed, N, K, test id, policy path), so a result can be checked.
43. As a researcher, I want the observation for Tests 1 and 3 to include relative shared target and relative neighbors (position and velocity), so tracking and avoidance are locally observable.
44. As a researcher, I want the Test 2 observation to include the assigned slot in the drone’s frame, so v1 does not hide the assignment in a side channel.
45. As a researcher, I want command limits enforced in the environment, so the policy cannot cheat physics.
46. As a researcher, I want episode termination on crash, collision, or out-of-bounds, so bad behavior is not rewarded for lingering.
47. As a future hardware user, I want observations and actions to stay local and per-drone, so a later physical port is not designed out.
48. As a future hardware user, I do not want this PRD to include cameras, onboard SLAM, or real airframes, so BTP1 can finish in simulation.
49. As a researcher, I want moving-geometry Test 2, unlabeled slots, wind, obstacles, and pursuit-evasion listed as later, so they are not silent scope creep.
50. As an evaluator, I want the project report to describe the three tests in this document’s words, so we are not presenting someone else’s problem statement.

## Implementation Decisions

- **Modules (logical, not file paths)**
  - **Quadrotor dynamics**: 3D rigid-body quadrotor; motors and integration live here.
  - **Flight driver**: cmd `[roll, pitch, yaw_rate, climb]` → motors. Stand-in for PX4 / micro-ROS.
  - **Swarm environment**: N instances, shared arena, reset/step, spawns, collisions, bounds, episode end. Task plug-in chooses target or slots.
  - **Local observation**: builds per-drone obs: self, K neighbors relative, relative shared target or relative slot. Must not leak global indices of all N except via the K set.
  - **Tasks**: Test 1 static shared target; Test 2 static geometry + slots; Test 3 moving shared target (scripted in train, joystick in demo).
  - **Pass gates**: collision, bounds, hold-time error; emit pass/fail and scalars. Independent of the policy.
  - **Baseline controller**: PID/geometric; same env interface as the policy (obs in, cmd out).
  - **Shared policy**: one network, one set of weights, batched over N at train time, independent at run time.
  - **Trainer**: on-policy loop, shared weights, logs vs gates. Does not own rendering.
  - **Viewer**: 3D view; Test 3 demo reads joystick and writes the shared target the env already uses.
  - **Config**: N, K, geometry kind/size, seed, test id, policy source (baseline | checkpoint).

- **Control and learning**
  - Decentralized **execution**. Training may see the joint episode (standard for swarms). No central action head.
  - Homogeneous drones, shared policy.
  - Neighbor sensing is computed from state; v1 has no comms graph.

- **Tests vs later iterations**
  - Initial demo: empty arena; Test 2 static + pre-assigned slots; Test 3 train scripted / demo joystick.
  - Next iterations (not this PRD’s delivery): moving geometry, unlabeled vertices, wind, obstacles, then physical drones.

- **Baseline vs learned**
  - Same env, same tests, same gates. Demo can select either.

- **Success reporting**
  - Quote pass gates and errors. Do not quote reward as the result.

## Testing Decisions

- Good tests check **external behavior**: reset/step contracts, collision flags, observation locality (no extra drones beyond K), gate pass/fail on known trajectories, cmd limits. They do not assert network internals or trainer step counts.
- **Must have checks**: quadrotor integration sanity (hover thrust), collision detector, observation K-cap, Test 1/2/3 target or slot placement, gate math on synthetic trajectories, baseline can complete a trivial hover without NaNs.
- **Acceptance**: the three tests on N=4 in the viewer, baseline and (when trained) learned policy, logged gates.
- **Not the acceptance bar**: mean episode reward, loss curves alone, or “it looked fine once.”

## Out of Scope

- Physical drones, radios, cameras, motion capture, onboard firmware.
- Wind / external factors and obstacles in the initial demo.
- Moving geometry, unlabeled self-arrangement, mid-episode slot swap.
- Pursuit-evasion (learned or scripted evader).
- Centralized action (one brain for all cmds).
- Explicit messaging protocol between drones.
- 2D stand-in that replaces the 3D quadrotor for the graded demo.
- Treating reward as the published success metric.

## Further Notes

- Iterate training against this PRD. Do not change test meaning to make a curve look better; change it only with an explicit PRD edit.
- Keep actions per-drone and observations local so a later physical pass is still conceivable.
- TRD next: physics timestep, observation vector layout, gate numbers, algorithm, and simulator choice. No product code until that TRD exists.
