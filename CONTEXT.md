# BTP1

Domain language for this project. Implementation belongs in `docs/` and code, not here.

BTP1 is a B.Tech project on **decentralized control of drone swarms with reinforcement learning**, shown first as a simulated demo of three tests.

## Language

**Drone**:
A single simulated quadrotor. Homogeneous: every drone uses the same body, the same observation layout, and the same policy.
_Avoid_: UAV, agent (ambiguous with the learning algorithm), robot, vehicle.

**Swarm**:
N drones in one arena, N ≥ 4 and configurable.
_Avoid_: fleet, team (unless we mean the research group).

**Decentralized policy**:
One shared policy copied onto every drone. At run time there is no commander. Each drone outputs only its own flight command (roll, pitch, yaw rate, climb). The driver maps that to motors.
_Avoid_: central controller, planner, assignment server.

**Neighbor set**:
The K nearest other drones in the local observation. K is configurable (start K=3 when N=4).
_Avoid_: global state, full swarm vector.

**Shared target**:
One point in the arena that the whole swarm cares about (Test 1 and Test 3). Drones hover around it; they do not occupy the same point.
_Avoid_: waypoint (that is a slot on a geometry), goal (overloaded).

**Standoff**:
The required spacing of drones around a shared target so they do not stack or collide.
_Avoid_: orbit (implies a prescribed circle — that is geometry).

**Geometry**:
A prescribed shape — square, rectangle, or circle — used in Test 2.
_Avoid_: formation (use Geometry + Slot).

**Collision**:
Square XY footprint (Chebyshev `collision_cm`) plus height. Two drones at the same altitude whose squares overlap have collided. Passing over/under (`|dz|` larger than body height) is not a collision.
_Avoid_: sphere hit, “they clipped in the viewer”.

**Baseline controller**:
A non-learning controller (PID / geometric) that runs the same three tests and the same pass gates as the learned policy.
_Avoid_: expert, scripted demo (the tests are real; the controller is just not learned).

**Pass gate**:
A numeric pass/fail rule: collisions, bounds, and error held under a threshold for a hold time. Reward is for training only.
_Avoid_: “looks good”, mean reward as the demo criterion.

**External factor**:
A force disturbance (wind) on a drone. Not an obstacle. Off for the initial demo.
_Avoid_: noise (too vague), random event.

**Obstacle**:
A body in the world a drone must not hit. Off for the initial demo; planned for after Test 3 tracking works.
_Avoid_: external factor (different thing).

**Joystick target**:
A human-moved shared target used in the Test 3 *demo*. Training uses a scripted or noisy moving target so episodes do not need a human.
_Avoid_: evader, pursuit (later extra).

## Flagged ambiguities

**Agent**: Do not use. Say **Drone** for the vehicle and **policy** for the learner.

**Goal**: Do not use. Say **shared target** (Tests 1 and 3) or **slot** (Test 2).

**Formation**: Prefer **geometry** plus **slot**. “Hold formation” in speech means “hold slots on the geometry.”

## Example dialogue

Dev: For Test 1, do we assign each drone a point on a circle around the shared target?

Expert: No. Test 1 is one shared target plus standoff and no collisions. A prescribed circle is Test 2 geometry.

Dev: Then in Test 2 we only publish vertices and let them self-arrange?

Expert: Later. v1 Test 2 pre-assigns slots. The shape stays static for the first demo. A moving geometry is the next iteration.

Dev: Test 3 is a person flying the target with a joystick while we train?

Expert: Demo only. Train tracking on a scripted moving target. Same decentralized policy at demo time follows the joystick target.
