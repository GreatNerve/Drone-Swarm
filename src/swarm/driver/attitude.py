from __future__ import annotations

import numpy as np

from swarm.dynamics.quadrotor import G, allocate

# ponytail: same limits as a typical PX4 MC tilt cap; yaml if a frame needs more
_CLIMB_CMPS = 200.0
_YAW_RATE = 1.5


def cmd_scales(cfg: dict | None = None) -> tuple[float, float, float]:
    """max_tilt_rad, max_climb_cmps, max_yaw_rate_rad_s."""
    tilt_deg = 18.0 if cfg is None else float(cfg.get("tilt_max_deg", 20.0)) * 0.9
    return float(np.deg2rad(tilt_deg)), _CLIMB_CMPS, _YAW_RATE


def hold_heading(rot: np.ndarray) -> np.ndarray:
    x_c = np.array([rot[0, 0], rot[1, 0], 0.0])
    n = float(np.linalg.norm(x_c))
    if n < 1e-4:
        return np.array([1.0, 0.0, 0.0])
    return x_c / n


class AttitudeDriver:
    """cmd [roll, pitch, yaw_rate, climb] in [-1, 1] → 4 motor actions.

    Stand-in for PX4 / MAVROS AttitudeTarget / micro-ROS firmware.
    Policy and PID never touch motors. Sim is ENU, +z up; a Pixhawk bridge
    flips to NED/FRD.
    """

    def __init__(
        self,
        mass_kg: float,
        arm_cm: float,
        max_thrust: float,
        yaw_coeff: float = 6.0,
        max_tilt: float | None = None,
        max_climb: float | None = None,
        max_yaw_rate: float | None = None,
        inertia_diag: np.ndarray | None = None,
    ):
        self.mass_kg = float(mass_kg)
        self.arm_cm = float(arm_cm)
        self.max_thrust = float(max_thrust)
        self.yaw_coeff = float(yaw_coeff)
        tilt, climb, yaw = cmd_scales()
        self.max_tilt = tilt if max_tilt is None else float(max_tilt)
        self.max_climb = climb if max_climb is None else float(max_climb)
        self.max_yaw_rate = yaw if max_yaw_rate is None else float(max_yaw_rate)
        self.inertia_diag = np.asarray(
            [23.0, 23.0, 40.0] if inertia_diag is None else inertia_diag, dtype=float
        )
        # torque gains: same numeric PD as the metre plant, times I_cm/I_m = 1e4
        self.k_att = 8.0e4
        self.k_rate = 0.45e4
        self.k_vz = 4.0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "AttitudeDriver":
        tilt, climb, yaw = cmd_scales(cfg)
        return cls(
            float(cfg["mass_kg"]),
            float(cfg["arm_cm"]),
            float(cfg["max_thrust"]),
            float(cfg.get("yaw_coeff", 6.0)),
            tilt,
            climb,
            yaw,
            cfg.get("inertia_diag"),
        )

    def decode(self, cmd: np.ndarray) -> tuple[float, float, float, float]:
        """roll_rad, pitch_rad, yaw_rate_rad_s, climb_cmps.

        PX4 VehicleAttitudeSetpoint: q_d(roll, pitch, yaw), yaw_sp_move_rate, thrust from climb.
        MAVROS AttitudeTarget: orientation + body_rate.z + thrust (type_mask 3).
        micro-ROS: same 4 floats (Float32MultiArray), not Twist (Twist is velocity).
        """
        cmd = np.clip(np.asarray(cmd, dtype=float).reshape(4), -1.0, 1.0)
        return (
            float(cmd[0]) * self.max_tilt,
            float(cmd[1]) * self.max_tilt,
            float(cmd[2]) * self.max_yaw_rate,
            float(cmd[3]) * self.max_climb,
        )

    def motors(self, rot: np.ndarray, omega: np.ndarray, vel: np.ndarray, cmd: np.ndarray) -> np.ndarray:
        roll, pitch, yaw_rate, climb = self.decode(cmd)
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        z_b = _body_z(roll, pitch, yaw)
        y_b = np.cross(z_b, hold_heading(rot))
        if np.linalg.norm(y_b) < 1e-4:
            y_b = np.array([0.0, 1.0, 0.0])
        y_b = y_b / (np.linalg.norm(y_b) + 1e-6)
        x_b = np.cross(y_b, z_b)
        rot_des = np.stack([x_b, y_b, z_b], axis=1)
        e_r = 0.5 * (rot_des.T @ rot - rot.T @ rot_des)
        att_err = np.array([e_r[2, 1], e_r[0, 2], e_r[1, 0]])
        omega_des = np.array([0.0, 0.0, yaw_rate])
        tau = -self.k_att * att_err - self.k_rate * (omega - omega_des)
        z_up = float(np.clip(rot[2, 2], 0.35, 1.0))
        f = self.mass_kg * (G + self.k_vz * (climb - float(vel[2]))) / z_up
        f = float(np.clip(f, 0.0, 4.0 * self.max_thrust))
        thrust = allocate(f, tau, self.arm_cm, self.yaw_coeff, self.max_thrust)
        return np.clip(2.0 * (thrust / self.max_thrust) - 1.0, -1.0, 1.0).astype(np.float32)


def _body_z(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    z = np.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])
    return z / (np.linalg.norm(z) + 1e-9)
