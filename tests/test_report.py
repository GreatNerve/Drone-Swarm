from pathlib import Path
import json

import numpy as np

from swarm.config import load, repo_root
from swarm.report import dump_pack, write_run_log


def _frame():
    n = 4
    return {
        "pos": np.zeros((n, 3)),
        "vel": np.zeros((n, 3)),
        "rot": np.repeat(np.eye(3)[None, ...], n, axis=0),
        "aim": np.zeros((n, 3)),
        "shared_target": np.zeros(3),
    }


def test_dump_pack_writes_result_and_data():
    cfg = load()
    hist = [_frame() for _ in range(8)]
    gate = {"test_id": 1, "passed": False, "error": 0.5, "reason": "ok"}
    out = dump_pack(
        cfg=cfg,
        mode="baseline",
        policy_name="baseline",
        ckpt=None,
        seed=0,
        gates=[gate],
        histories={1: hist},
        snaps=None,
        also_submit=True,
    )
    assert (out / "result.md").is_file()
    assert (out / "result.json").is_file()
    assert (out / "data" / "trajectory.csv").is_file()
    assert (out / "graphs" / "distance.png").is_file()
    submit = repo_root() / "results" / "submit"
    assert (submit / "result.md").is_file()
    csv = (out / "data" / "trajectory.csv").read_text(encoding="utf-8")
    assert "test_id" in csv
    assert Path(out).is_relative_to(repo_root() / "results")


def test_write_run_log_keeps_three(tmp_path: Path):
    cfg = load()
    hist = [_frame() for _ in range(12)]
    hist[-1]["pos"] = np.array([[0.0, 600.0, 200.0], [10.0, 0.0, 200.0], [20.0, 0.0, 200.0], [30.0, 0.0, 200.0]])
    hist[-1]["aim"] = np.zeros((4, 3))
    folder = tmp_path / "runs"
    paths = []
    for i in range(4):
        p = write_run_log(
            cfg=cfg,
            cmd="view",
            mode="baseline",
            policy_name="baseline",
            seed=i,
            episodes=[
                {
                    "test_id": 2,
                    "history": hist,
                    "collided": False,
                    "out_of_bounds": True,
                    "floor_hit": False,
                    "extra": {"shape": "rectangle"},
                }
            ],
            folder=folder,
        )
        assert p is not None and p.is_file()
        paths.append(p)
    kept = sorted(folder.glob("run_*.json"))
    assert len(kept) == 3
    assert paths[0] not in kept
    data = json.loads(kept[-1].read_text(encoding="utf-8"))
    assert data["issues"]
    assert any("far" in x or "out_of_bounds" in x for x in data["issues"])
