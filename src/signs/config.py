"""One configuration object for a run, read from YAML.

A config file may name a `base` file. The base is loaded first and the
file's own keys are merged over it, so each experiment file lists only
what it changes. Command-line overrides such as `train.epochs=10` are
applied last.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AugmentConfig:
    # No horizontal flip on purpose: a mirrored "keep right" is a different sign.
    rotation_degrees: float = 15.0
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2


@dataclass
class DataConfig:
    data_dir: str = "data/hf"
    image_size: int = 64
    batch_size: int = 64
    val_fraction: float = 0.2
    # The split seed stays fixed when the training seed changes, so every
    # seed of an experiment is scored on the same validation images.
    split_seed: int = 42
    num_workers: int = 0
    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass
class ModelConfig:
    name: str = "mobilenet_v2"
    pretrained: bool = True
    freeze_features: bool = True


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3


@dataclass
class Config:
    run_name: str = "mobilenet_frozen"
    seed: int = 42
    runs_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.run_name


def _merge(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    base_name = raw.pop("base", None)
    if base_name is None:
        return raw
    base_path = (path.parent / base_name).resolve()
    if base_path == path.resolve():
        raise ValueError(f"{path} names itself as its base")
    return _merge(_read_yaml(base_path), raw)


def _build(cls, values: dict, where: str):
    hints = typing.get_type_hints(cls)
    unknown = set(values) - set(hints)
    if unknown:
        raise ValueError(f"unknown config key(s) in {where or 'config'}: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        if dataclasses.is_dataclass(hints[name]):
            prefix = f"{where}.{name}" if where else name
            kwargs[name] = _build(hints[name], value or {}, prefix)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _parse_override(text: str) -> dict:
    if "=" not in text:
        raise ValueError(f"override {text!r} must look like key.subkey=value")
    dotted, raw_value = text.split("=", 1)
    value = yaml.safe_load(raw_value)
    tree: dict = {}
    node = tree
    keys = dotted.strip().split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
    return tree


def from_dict(values: dict) -> Config:
    return _build(Config, values, "")


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Read a config file (or the defaults), then apply `key=value` overrides."""
    values = _read_yaml(Path(path)) if path is not None else {}
    for text in overrides or []:
        values = _merge(values, _parse_override(text))
    return from_dict(values)


def save_config(config: Config, path: Path) -> None:
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
