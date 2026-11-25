import json
import math
import os
import sys
import threading
import time

import pytest

from swarmflow import files
from swarmflow.curriculum import training_chain
from swarmflow.files import (append_jsonl, local_path, read_jsonl, replace_with_retry, sanitize, write_json_atomic)


def test_replace_waits_out_a_transient_lock(tmp_path, monkeypatch):
    source, target = tmp_path / "new", tmp_path / "old"
    source.write_text("new")
    target.write_text("old")
    real, calls = os.replace, []

    def flaky(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(13, "The process cannot access the file")
        real(a, b)

    monkeypatch.setattr(files.os, "replace", flaky)
    replace_with_retry(source, target, timeout=5)
    assert target.read_text() == "new" and len(calls) == 3


def test_replace_gives_up_on_a_permanent_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        replace_with_retry(tmp_path / "missing", tmp_path / "target", timeout=1)


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows blocks replacing an open file")
def test_replace_succeeds_once_another_handle_closes(tmp_path):
    target = tmp_path / "best.pt"
    target.write_bytes(b"old")
    source = tmp_path / "best.pt.tmp"
    source.write_bytes(b"new")
    handle = open(target, "rb")
    threading.Timer(0.5, handle.close).start()
    began = time.monotonic()
    replace_with_retry(source, target, timeout=10)
    assert target.read_bytes() == b"new" and time.monotonic() - began >= 0.3


def test_json_never_fails_on_nan(tmp_path):
    path = tmp_path / "row.json"
    write_json_atomic(path, {"loss": float("nan"), "values": [1.0, float("inf")], "ok": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"loss": None, "values": [1.0, None], "ok": 2}
    assert sanitize({"x": math.nan})["x"] is None


def test_jsonl_survives_a_partial_last_line(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text('{"update": 1}\n{"update": 2, "loss', encoding="utf-8")
    append_jsonl(path, {"update": 3})
    assert read_jsonl(path) == [{"update": 1}, {"update": 3}]
    assert path.read_text(encoding="utf-8").endswith('{"update": 3}\n')


def test_paths_written_on_windows_read_back_everywhere(tmp_path, monkeypatch):
    assert local_path("models\\stage2-seed0\\best.pt").parent.name == "stage2-seed0"
    monkeypatch.chdir(tmp_path)
    for number, source in ((1, None), (2, "models\\stage1-seed0\\best.pt")):
        folder = tmp_path / "models" / f"stage{number}-seed0"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"stage": number, "initialized_from": source}))
    chain = training_chain("models/stage2-seed0")
    assert [folder.name for folder in chain] == ["stage1-seed0", "stage2-seed0"]
