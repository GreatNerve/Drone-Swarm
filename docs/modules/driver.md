# driver

`AttitudeDriver`: flight cmd → 4 motor actions. Stand-in for a PX4 / MAVROS / micro-ROS mixer. Policy and PID never call `Quadrotor` motors.

Cmd in `[-1, 1]`: `[roll, pitch, yaw_rate, climb]`. Hover is zeros. `decode()` is rad, rad/s, cm/s. Sim is ENU, +z up.

Later hardware publishes the same cmd:

| Stack | Message |
| --- | --- |
| PX4 ROS 2 | `VehicleAttitudeSetpoint` (`q_d` from roll/pitch/yaw, `yaw_sp_move_rate`, `thrust_body` from climb) |
| MAVROS / Pixhawk | `mavros_msgs/AttitudeTarget` (orientation + `body_rate.z` + thrust, `type_mask=3`) |
| ESP32/STM32 micro-ROS | `std_msgs/Float32MultiArray` of the same 4 floats. Not `Twist` (`Twist` is velocity). |

Do not add `rclpy` / `px4_msgs` until a board is on the desk. Coordinates stay in the task (shared target / slot), not in the cmd.

Attitude torque uses `k_att`/`k_rate` scaled with the centimetre inertia (`×1e4`). Linear force still uses `mass_kg * (G + …)` with `G = 981`.

Public: `AttitudeDriver.from_cfg`, `decode`, `motors`, `cmd_scales`. Depends on dynamics `allocate`.
