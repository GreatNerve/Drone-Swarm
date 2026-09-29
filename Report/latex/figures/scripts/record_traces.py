"""Record full-rate traces of the final (V3) model on held-out seed 100000 for every benchmark scenario."""
import json
from pathlib import Path

from swarmflow.env import SwarmEnv
from swarmflow.policy import LearnedController

ROOT = Path(__file__).resolve().parents[4]
CKPT = ROOT / "models/previous-runs/run3-pipeline-v3/showcase/best.pt"
OUT = Path(__file__).resolve().parent / "traces"
OUT.mkdir(exist_ok=True)


def record(controller, scenario, seed):
    env = SwarmEnv(controller.config, scenario)
    obs, _ = env.reset(seed=seed)
    frames = [{**env.snapshot(), "observation": {k: v.tolist() for k, v in obs["drone_0"].items()}}]
    while env.agents:
        obs, _, _, _, _ = env.step(controller(obs))
        snap = env.snapshot()
        if obs:
            snap["observation"] = {k: v.tolist() for k, v in obs["drone_0"].items()}
        frames.append(snap)
    return env.summary(), frames


def main():
    controller = LearnedController(CKPT)
    index = {}
    for scenario in ("hold", "follow", "turn", "gust", "crossing", "head_on", "double_crossing"):
        summary, frames = record(controller, scenario, 100000)
        (OUT / f"{scenario}_100000.json").write_text(json.dumps({"summary": summary, "frames": frames}))
        index[scenario] = {k: summary[k] for k in ("failure", "slot_rmse_m", "min_surface_clearance_m")}
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
