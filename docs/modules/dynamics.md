# dynamics

3D rigid-body quadrotor. Owns mass, inertia, X-config mixer, SO(3) integration. Length, speed, and accel are centimetres (`G = 981`).

Public: `Quadrotor.reset`, `step(action, dt)`, `hover_thrust`, `mix_matrix`, `allocate`.
Action `[-1,1]` → thrust. Mixer is shared with `AttitudeDriver`. Policy and PID do not call it.

`yaw_coeff` and `inertia_diag` are YAML knobs.

Depends on: numpy. Not on env or policy.
