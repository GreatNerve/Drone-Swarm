import argparse
import sys

from .config import FORMATIONS, DemoConfig, ViewerConfig


def _safe_output():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None):
    _safe_output()
    parser = argparse.ArgumentParser(prog="swarmv2", description="Swarm-V2: 6-DoF terrain-aware drone swarm")
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view", help="Open the 3D viewer")
    view.add_argument("--demo", action="store_true",
                      help="Phase 0 scene: a scripted tilted swarm over procedural terrain with mock reasoning "
                           "(no physics yet)")
    view.add_argument("--port", type=int, default=ViewerConfig.port)
    view.add_argument("--seed", type=int, default=1000, help="Terrain and wind seed (default: validation seed 1000)")
    view.add_argument("--drones", type=int, default=DemoConfig.drones)
    view.add_argument("--formation", choices=FORMATIONS, default=DemoConfig.formation)
    view.add_argument("--no-browser", action="store_true")
    commands.add_parser("info", help="Show Python, PyTorch and the compute device")
    args = parser.parse_args(argv)
    try:
        if args.command == "info":
            from . import PHASE, __version__
            from .device import describe
            print(f"swarmv2 {__version__} (phase {PHASE})\n{describe()}")
            return
        if not args.demo:
            parser.error("Phase 0 has no simulator yet; use `swarmv2 view --demo`")
        from .viewer.demo import DemoScene
        from .viewer.server import serve
        scene = DemoScene(DemoConfig(drones=args.drones, formation=args.formation), seed=args.seed)
        serve(scene, ViewerConfig(port=args.port), open_browser=not args.no_browser)
    except ValueError as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        raise SystemExit(130)
