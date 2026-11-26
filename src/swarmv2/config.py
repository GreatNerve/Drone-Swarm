"""Configuration dataclasses. Vehicle parameters are deliberately unset until Phase 1 (decision D3)."""
from dataclasses import asdict, dataclass, field

FORMATIONS = ("DIAMOND", "COLUMN", "V_SHAPE", "LINE")
FLIGHT_MODES = ("STEADY", "AGILE")


@dataclass(frozen=True)
class VehicleConfig:
    """Rigid-body and actuator parameters of the D3 quadrotor, fixed from a cited source at Phase 1.

    Phase 0 only needs the size for drawing; physics must refuse to run until the rest is set.
    """
    radius_m: float = 0.18
    mass_kg: float | None = None
    arm_m: float | None = None
    inertia_kg_m2: tuple | None = None
    max_motor_thrust_n: float | None = None
    motor_lag_s: float | None = None
    drag_linear: tuple | None = None
    drag_quadratic: tuple | None = None
    source: str | None = None

    def complete(self):
        return all(value is not None for value in asdict(self).values())


@dataclass(frozen=True)
class TerrainConfig:
    size_m: float = 80.0
    resolution_m: float = 1.0
    max_height_m: float = 6.0
    canyon_depth_m: float = 5.0
    canyon_width_m: float = 7.0
    pillars: int = 14
    pillar_radius_m: tuple = (0.4, 0.9)
    city_x_m: tuple = (-37.0, -13.0)
    city_half_depth_m: float = 30.0
    street_half_width_m: float = 7.5
    building_height_m: tuple = (6.0, 24.0)
    building_footprint_m: tuple = (4.5, 8.0)
    gate_clearance_m: float = 9.0
    trees: int = 70

    @property
    def cells(self):
        return int(round(self.size_m / self.resolution_m)) + 1


@dataclass(frozen=True)
class WindConfig:
    mean_mps: tuple = (3.0, 1.0, 0.0)
    gust_sigma_mps: float = 1.5
    gust_tau_s: float = 2.0


@dataclass(frozen=True)
class DemoConfig:
    drones: int = 8
    formation: str = "DIAMOND"
    formation_scale_m: float = 1.6
    target_agl_m: float = 3.0
    cruise_speed_mps: float = 1.2
    frame_hz: float = 30.0
    reasoning_period_s: float = 1.5
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    wind: WindConfig = field(default_factory=WindConfig)

    def __post_init__(self):
        if not 2 <= self.drones <= 64:
            raise ValueError("drones must be between 2 and 64")
        if self.formation not in FORMATIONS:
            raise ValueError(f"formation must be one of {FORMATIONS}")


@dataclass(frozen=True)
class ViewerConfig:
    host: str = "127.0.0.1"
    port: int = 8770
