# baseline

Geometric PID: same obs and 4-cmd as the policy. Outer loop only (standoff + repulsion → roll/pitch/climb). `AttitudeDriver` mixes motors.

Holds yaw_rate = 0. Tests 1 and 3 track a standoff ring around the shared target (Test 3 also feed-forwards target velocity from the obs residual). Neighbor repulsion is XY (capped, no 1/r² kick) plus a local altitude split so Test 2 paths can pass over each other.

Public: `GeometricPid.from_cfg(cfg, test_id=1)`, `step(obs) -> (N,4) in [-1,1]`.
