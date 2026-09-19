from swarm.tasks.base import Task, TaskState
from swarm.tasks.test_geometry import GeometryTask
from swarm.tasks.test_moving_target import MovingTargetTask
from swarm.tasks.test_static_target import StaticTargetTask


def make_task(test_id: int, cfg: dict) -> Task:
    n = int(cfg["num_drones"])
    target = cfg["shared_target"]
    if test_id == 1:
        return StaticTargetTask(n, target)
    if test_id == 2:
        z = float(target[2])
        return GeometryTask(
            n, str(cfg.get("geometry_kind", "square")), float(cfg["geometry_size_cm"]), z
        )
    if test_id == 3:
        return MovingTargetTask(n, target, radius_cm=200.0)
    raise ValueError(f"test_id must be 1, 2, or 3; got {test_id}")


__all__ = ["GeometryTask", "MovingTargetTask", "StaticTargetTask", "Task", "TaskState", "make_task"]
