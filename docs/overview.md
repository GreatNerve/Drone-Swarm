# Overview

BTP1 is a B.Tech project: **decentralized control of drone swarms with reinforcement learning**.

We train one shared policy that runs independently on every drone. There is no central commander. The first showable result is a 3D simulated swarm (N ≥ 4) passing three tests, with a non-learning baseline on the same tests.

## In (this PRD)

- Simulated 3D quadrotors, attitude commands, local observations.
- Three tests: shared static target, static geometry with slots, moving shared target.
- Baseline controller and learned policy, same pass gates.
- Live Test 3 demo with a joystick target; training uses a scripted moving target.

## Out (this PRD)

- Physical drones (later; do not design them out).
- Wind, obstacles, moving geometry, unlabeled self-arrangement (later iterations).
- A central planner or a unique brain that outputs all N actions.

## Status

PRD: [prd.md](./prd.md). TRD: [trd.md](./trd.md). Package: `src/swarm/`.
