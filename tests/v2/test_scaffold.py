import re
from pathlib import Path

import numpy as np
import pytest

import swarmv2
from swarmv2 import seeds
from swarmv2.cli import main
from swarmv2.config import DemoConfig, VehicleConfig

PACKAGE = Path(swarmv2.__file__).parent


def test_v2_never_imports_v1():
    pattern = re.compile(r"^\s*(from|import)\s+swarmflow", re.MULTILINE)
    offenders = [path for path in PACKAGE.rglob("*.py") if pattern.search(path.read_text(encoding="utf-8"))]
    assert offenders == []


def test_seed_ranges_are_disjoint_and_cover_everything():
    assert seeds.split_of(0) == "train" and seeds.split_of(999) == "train"
    assert seeds.split_of(1000) == "validation" and seeds.split_of(1999) == "validation"
    assert seeds.split_of(2000) == "train" and seeds.split_of(99_999) == "train"
    assert seeds.split_of(100_000) == "test"
    with pytest.raises(ValueError):
        seeds.check(1500, "train")
    with pytest.raises(ValueError):
        seeds.split_of(-1)
    drawn = seeds.training_seeds(np.random.default_rng(0), 200_000)
    assert all(seeds.split_of(s) == "train" for s in np.unique(drawn))
    assert drawn.min() < 1000 and drawn.max() > 50_000


def test_vehicle_parameters_are_unset_until_phase_1():
    assert not VehicleConfig().complete()
    with pytest.raises(ValueError):
        DemoConfig(formation="CIRCLE")


def test_info_command_reports_the_device(capsys):
    main(["info"])
    out = capsys.readouterr().out
    assert "swarmv2" in out and "torch" in out and "auto device:" in out


def test_view_without_a_scene_source_is_refused():
    with pytest.raises(SystemExit):
        main(["view"])
