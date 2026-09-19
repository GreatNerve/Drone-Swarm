from __future__ import annotations

import logging

from swarm.viewer.gazebo_view import GazeboView
from swarm.viewer.python_view import PythonView

log = logging.getLogger("swarm.viewer")


def make_viewer(kind: str, cfg: dict, demo: bool = False):
    if kind in {"0", "none", "off", ""}:
        return None
    if kind == "python":
        return PythonView(cfg, demo=demo)
    if kind == "gazebo":
        try:
            return GazeboView(cfg)
        except RuntimeError as exc:
            log.error("%s; falling back to python view", exc)
            return PythonView(cfg, demo=demo)
    log.error("unknown view %s", kind)
    return None


__all__ = ["GazeboView", "PythonView", "make_viewer"]
