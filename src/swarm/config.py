from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read config {path}: {exc}") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise RuntimeError(f"config {path} must be a mapping")
    return data


def load(train: bool = False) -> dict[str, Any]:
    root = repo_root()
    cfg = _read_yaml(root / "configs" / "default.yaml")
    if train:
        cfg.update(_read_yaml(root / "configs" / "train.yaml"))
    return cfg
