# env

`SwarmEnv`: N drones, one task, collisions, bounds, rewards, obs.

Public: `reset() -> obs (N, 18+6K)`, `step(cmds) -> StepResult`, `set_joystick_target`, `set_geometry_kind`, `set_test_id`.

`step` takes `[roll, pitch, yaw_rate, climb]` per drone. `AttitudeDriver` mixes motors; dynamics stay a plant.

Pair distances are vectorized. A hit is a **square XY footprint** overlapping at the same altitude (`body_overlap`); different altitude is a pass-through. Closing pairs get an impulse; floor hits bounce z.

Train episodes end on collision or out-of-bounds. Eval episodes always run to `episode_s` so pass gates see a full hold window.

Tests 1 and 3 reward a standoff ring around the shared target (same idea as the PID baseline). Test 2 rewards slot error. The plant and reward are in **centimetres**. Tracking accuracy is **one body (10 cm)**: error inside that band is on-station. Beyond one body the cost is `e + e²` with `e = excess_cm / 10`, uncapped, plus receding velocity and progress `max(e_now - e_prev, 0)`. Hover-aux fade uses the same 10 cm band.

Owns drone instances and episode clock. Depends on dynamics, obs, tasks.
