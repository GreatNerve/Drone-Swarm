import json
import signal

import numpy as np
import pytest
import torch

from swarmflow import Config
from swarmflow import train as training
from swarmflow.curriculum import train_stage
from swarmflow.files import load_torch, read_jsonl
from swarmflow.policy import ActorCritic

CONFIG = Config(drones=4, duration=.2)
TINY = {"horizon": 8, "epochs": 1, "envs": 1, "eval_seeds": 1, "width": 32, "critic_warmup": 0}


def _rows(folder, kind):
    return [row for row in read_jsonl(folder / "training.jsonl") if row["type"] == kind]


def _evaluations(folder):
    return [(row["reason"], row["environment_steps"]) for row in read_jsonl(folder / "evaluations.jsonl")]


def test_stop_during_cloning_continues_at_the_next_round(tmp_path, monkeypatch):
    real, calls = training._fit_clone, []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "_fit_clone", interrupted)
    with pytest.raises(KeyboardInterrupt):
        training.train(CONFIG, ["hold"], total_steps=16, output=tmp_path, clone_steps=4, clone_epochs=1,
                       dagger_rounds=2, **TINY)
    saved = load_torch(tmp_path / "latest.pt")
    assert saved["progress"]["phase"] == "cloning" and saved["progress"]["clone_rounds_done"] == 1
    assert (tmp_path / "teacher-data" / "round0.pt").exists()
    monkeypatch.setattr(training, "_fit_clone", real)
    training.train(CONFIG, ["hold"], total_steps=1, output=tmp_path, continue_run=True)
    assert [row["round"] for row in _rows(tmp_path, "behavior_cloning")] == [0, 1, 2]
    assert _evaluations(tmp_path) == [("start", 0), ("final", 16)]
    assert (tmp_path / "finished.json").exists() and not (tmp_path / "teacher-data").exists()


def test_ctrl_c_stops_after_the_update_and_the_same_call_continues_exactly(tmp_path, monkeypatch):
    real_step, steps = training.Worker.step, []

    def step(self, actions):
        steps.append(1)
        if len(steps) == 3:
            signal.raise_signal(signal.SIGINT)
        return real_step(self, actions)

    monkeypatch.setattr(training.Worker, "step", step)
    arguments = dict(total_steps=32, output=tmp_path, clone_steps=4, clone_epochs=1, eval_every=16, **TINY)
    with pytest.raises(KeyboardInterrupt) as stopped:
        training.train(CONFIG, ["hold"], **arguments)
    assert isinstance(stopped.value, training.TrainingStopped)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    saved = load_torch(tmp_path / "latest.pt")
    assert saved["environment_steps"] == 8 and saved["progress"]["updates"] == 1
    teacher = load_torch(tmp_path / "teacher-data" / "round0.pt")

    def rebuilt(*args, **kwargs):
        raise AssertionError("the saved teacher dataset must be reused")

    monkeypatch.setattr(training.Worker, "step", real_step)
    monkeypatch.setattr(training, "demonstrations", rebuilt)
    seen = []
    real_tensors = training._demo_tensors
    monkeypatch.setattr(training, "_demo_tensors", lambda data, labels, *a: seen.append(
        np.concatenate(labels)) or real_tensors(data, labels, *a))
    training.train(CONFIG, ["hold"], continue_run=True, total_steps=1, output=tmp_path)
    np.testing.assert_array_equal(seen[0], teacher["labels"].numpy())
    ppo = _rows(tmp_path, "ppo")
    assert [row["environment_steps"] for row in ppo] == [8, 16, 24, 32]
    assert [row["update"] for row in ppo] == [1, 2, 3, 4]
    assert _evaluations(tmp_path) == [("start", 0), ("periodic", 16), ("periodic", 32)]
    finished = json.loads((tmp_path / "finished.json").read_text())
    assert finished["environment_steps"] == 32 and finished["updates_this_run"] == 4


def test_a_hard_kill_between_logging_and_saving_never_duplicates_rows(tmp_path, monkeypatch):
    real = training.save_torch_atomic

    def killed(data, path, *args, **kwargs):
        if path.name == "latest.pt" and data["environment_steps"] == 16:
            raise SystemExit("killed")
        return real(data, path, *args, **kwargs)

    monkeypatch.setattr(training, "save_torch_atomic", killed)
    with pytest.raises(SystemExit):
        training.train(CONFIG, ["hold"], total_steps=32, output=tmp_path, eval_every=16, **TINY)
    monkeypatch.setattr(training, "save_torch_atomic", real)
    training.train(CONFIG, ["hold"], total_steps=1, output=tmp_path, continue_run=True)
    assert [row["environment_steps"] for row in _rows(tmp_path, "ppo")] == [8, 16, 24, 32]
    assert _evaluations(tmp_path) == [("start", 0), ("periodic", 16), ("periodic", 32)]


def test_a_stop_between_the_last_update_and_finishing_is_completed_on_the_next_run(tmp_path):
    training.train(CONFIG, ["hold"], total_steps=8, output=tmp_path, **TINY)
    (tmp_path / "finished.json").unlink()
    data = load_torch(tmp_path / "latest.pt")
    data["progress"]["last_validated"] = 0
    torch.save(data, tmp_path / "latest.pt")
    training.train(CONFIG, ["hold"], total_steps=1, output=tmp_path, continue_run=True)
    assert _evaluations(tmp_path)[-1] == ("final", 8)
    assert json.loads((tmp_path / "finished.json").read_text())["environment_steps"] == 8


def test_checkpoints_from_before_exact_resume_still_continue(tmp_path):
    training.train(CONFIG, ["hold"], total_steps=8, output=tmp_path, **TINY)
    data = load_torch(tmp_path / "latest.pt")
    del data["progress"]
    data["training"]["target_environment_steps"] = 16
    torch.save(data, tmp_path / "latest.pt")
    training.train(CONFIG, ["hold"], total_steps=1, output=tmp_path, continue_run=True)
    assert [row["environment_steps"] for row in _rows(tmp_path, "ppo")] == [8, 16]


def test_a_locked_latest_checkpoint_does_not_stop_training(tmp_path, monkeypatch, capsys):
    real, failures = training.save_torch_atomic, []

    def locked(data, path, *args, **kwargs):
        if path.name == "latest.pt" and len(failures) < 2:
            failures.append(1)
            raise PermissionError(13, "locked by another process")
        return real(data, path, *args, **kwargs)

    monkeypatch.setattr(training, "save_torch_atomic", locked)
    training.train(CONFIG, ["hold"], total_steps=24, output=tmp_path, **TINY)
    assert "could not save latest.pt" in capsys.readouterr().out
    assert load_torch(tmp_path / "latest.pt")["environment_steps"] == 24


def test_non_finite_updates_are_rolled_back_then_stop_cleanly(tmp_path, monkeypatch):
    real, calls = ActorCritic.evaluate, []

    def poisoned(self, observation, raw, n_agents=None):
        calls.append(1)
        logp, entropy, value = real(self, observation, raw, n_agents)
        return (logp * float("nan"), entropy, value) if len(calls) == 1 else (logp, entropy, value)

    monkeypatch.setattr(ActorCritic, "evaluate", poisoned)
    training.train(CONFIG, ["hold"], total_steps=16, output=tmp_path, **TINY)
    ppo = _rows(tmp_path, "ppo")
    assert ppo[0]["skipped_minibatches"] == 1 and ppo[1]["skipped_minibatches"] == 0
    assert ppo[1]["lr"] == pytest.approx(3e-4 * 0.1 * 0.5)
    monkeypatch.setattr(ActorCritic, "evaluate",
                        lambda self, o, r, n=None: tuple(t * float("nan") for t in real(self, o, r, n)))
    with pytest.raises(RuntimeError, match="run the same command to continue"):
        training.train(CONFIG, ["hold"], total_steps=64, output=tmp_path / "bad", **TINY)


def test_pipeline_stage_interrupted_while_cloning_is_continued_not_restarted(tmp_path, monkeypatch):
    models = tmp_path / "models"
    stage = dict(config=CONFIG, clone_steps=4, clone_epochs=1, dagger_rounds=1, total_steps=8, eval_every=8,
                 snapshot_every=0, **TINY)
    real, calls = training._fit_clone, []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "_fit_clone", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train_stage(1, 0, models, **stage)
    monkeypatch.setattr(training, "_fit_clone", real)
    train_stage(1, 0, models, **stage)
    folder = models / "stage1-seed0"
    assert [row["round"] for row in _rows(folder, "behavior_cloning")] == [0, 1]
    assert (folder / "finished.json").exists()
    assert not list(models.glob("stage1-seed0-old-*"))
