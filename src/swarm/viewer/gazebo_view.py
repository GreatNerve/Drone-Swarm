from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

from swarm.config import repo_root

log = logging.getLogger("swarm.viewer")


class GazeboView:
    """Launch world if gz/gazebo is on PATH. Physics stays in swarm.dynamics."""

    def __init__(self, cfg: dict):
        self.exe = shutil.which("gz") or shutil.which("gazebo")
        if self.exe is None:
            raise RuntimeError("Gazebo not found on PATH")
        self.world = repo_root() / "assets" / "worlds" / "swarm.sdf"
        self._path = repo_root() / "models" / "gazebo_state.csv"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.proc: subprocess.Popen | None = None
        self._launch()
        log.info("Gazebo %s world=%s (draw-only)", self.exe, self.world)

    def _launch(self) -> None:
        if not self.world.is_file():
            log.error("world missing: %s", self.world)
            return
        name = Path(self.exe).stem.lower()
        if name in {"gz", "ign"}:
            cmd = [self.exe, "sim", "-r", str(self.world)]
        else:
            cmd = [self.exe, str(self.world)]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            log.error("cannot start Gazebo: %s", exc)
            self.proc = None

    def draw(self, snap: dict, stats: dict | None = None) -> None:
        pos = np.asarray(snap["pos"])
        try:
            with self._path.open("w", encoding="utf-8") as f:
                f.write("i,x,y,z\n")
                for i, p in enumerate(pos):
                    f.write(f"{i},{p[0]},{p[1]},{p[2]}\n")
        except OSError as exc:
            log.error("cannot write gazebo state: %s", exc)

    def poll(self) -> dict:
        return {}

    def is_open(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
