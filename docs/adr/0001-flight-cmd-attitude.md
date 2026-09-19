# 0001 — Flight cmd is attitude, not motors

## Status

Accepted.

## Context

Policy and PID used to output 4 motor commands. That fights PX4, MAVROS, and micro-ROS on ESP32/STM32, where the flight controller owns mixing. Clamping tilt in `Quadrotor.step` was rejected: Test 3 may need tilt, and physics should stay a plant.

## Decision

Env action is `[roll, pitch, yaw_rate, climb]` in `[-1, 1]`. `AttitudeDriver` maps cmd → motors. Same layout as PX4 `VehicleAttitudeSetpoint` / MAVROS `AttitudeTarget`. Coordinates stay in the task.

## Consequences

Old motor-space checkpoints are invalid. Retrain. A later ROS 2 / micro-ROS node republishes the same 4-vector; it does not fork dynamics.
