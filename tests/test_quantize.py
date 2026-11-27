import json
import shutil
from datetime import UTC, datetime

import numpy as np
import pytest
import torch

from swarmflow import Config, SwarmEnv
from swarmflow import quantize
from swarmflow.curriculum import model_report
from swarmflow.files import load_torch
from swarmflow.policy import LearnedController
from swarmflow.train import train


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("trained")
    train(Config(drones=4, duration=.2), ["hold"], total_steps=8, horizon=8, epochs=1, output=folder, envs=1,
          eval_seeds=1, width=32, clone_steps=4, clone_epochs=1)
    return folder / "best.pt"


def test_training_time_survives_copy_and_quantization(checkpoint, tmp_path):
    trained_at = model_report(checkpoint)["last_trained_at"]
    assert datetime.fromisoformat(trained_at).tzinfo == UTC
    copied = shutil.copyfile(checkpoint, tmp_path / "copy.pt")
    assert model_report(copied)["last_trained_at"] == trained_at

    data = load_torch(copied)
    variant = tmp_path / "fp16.pt"
    torch.save(quantize.build_variant(data, "fp16"), variant)
    assert model_report(variant)["last_trained_at"] == trained_at

    del data["last_trained_at"]
    legacy = tmp_path / "legacy.pt"
    torch.save(data, legacy)
    assert model_report(legacy)["last_trained_at"] is None


def test_every_preset_is_smaller_loads_everywhere_and_flies_like_the_original(checkpoint, tmp_path):
    report = quantize.quantize_checkpoint(checkpoint, list(quantize.VARIANTS), tmp_path, seeds=1, scenarios=["hold"])
    rows = {row["variant"]: row for row in report["variants"]}
    assert set(rows) == {"original", *quantize.VARIANTS}
    full = checkpoint.stat().st_size
    for name in quantize.VARIANTS:
        path = tmp_path / name / "model.pt"
        assert path.stat().st_size < full
        data = torch.load(path, weights_only=True)
        assert data["format"] == "swarmflow-v4-actor" and data["variant"] == name
        assert rows[name]["episodes"] == 1
    assert rows["fp32"]["max_action_error"] == 0
    assert rows["fp16"]["max_action_error"] < 1e-2
    assert rows["int8"]["max_action_error"] < 0.1
    assert rows["int8"]["size_bytes"] < rows["fp16"]["size_bytes"] < rows["fp32"]["size_bytes"]
    assert load_torch(tmp_path / "int8" / "model.pt")["weights"]["trunk.0.weight"].dtype == torch.int8
    assert json.loads((tmp_path / "report.json").read_text())["variants"][0]["variant"] == "original"
    assert "| int8 |" in (tmp_path / "QUANTIZATION.md").read_text()


def test_quantized_model_drives_the_swarm_and_reports_its_precision(checkpoint, tmp_path):
    path = tmp_path / "model.pt"
    torch.save(quantize.build_variant(load_torch(checkpoint), "int8-keep-head", checkpoint), path)
    controller = LearnedController(path)
    env = SwarmEnv(controller.config, "hold")
    observations, _ = env.reset(seed=4)
    actions = controller(observations)
    assert set(actions) == set(observations) and all(np.all(np.abs(a) <= 1) for a in actions.values())
    assert controller.metadata["precision"] == "int8"
    assert controller.metadata["source_checkpoint"].endswith("best.pt")


def test_int8_dynamic_falls_back_to_float_without_a_quantized_engine(checkpoint, monkeypatch):
    data = quantize.build_variant(load_torch(checkpoint), "int8-dynamic", checkpoint)
    monkeypatch.setattr(quantize, "_int8_engine", lambda: None)
    model = quantize.load_actor(data)
    assert model.compute == "float32"
    reference = quantize.load_actor(quantize.build_variant(load_torch(checkpoint), "int8", checkpoint))
    example = quantize._example_input(model)
    with torch.no_grad():
        torch.testing.assert_close(model.act(example), reference.act(example))


def test_actor_only_files_cannot_be_trained_or_requantized(checkpoint, tmp_path):
    path = tmp_path / "fp16.pt"
    torch.save(quantize.build_variant(load_torch(checkpoint), "fp16", checkpoint), path)
    with pytest.raises(ValueError, match="actor-only"):
        train(Config(drones=4, duration=.2), ["hold"], total_steps=8, output=tmp_path / "again", resume=path)
    with pytest.raises(ValueError, match="full"):
        quantize.quantize_checkpoint(path, ["int8"], tmp_path / "q", seeds=0)
    with pytest.raises(ValueError, match="Unknown"):
        quantize.quantize_checkpoint(checkpoint, ["int4"], tmp_path / "q", seeds=0)
