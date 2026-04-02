"""YAML configuration loader with dot-notation access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Hierarchical configuration loaded from YAML.

    Supports dot-notation access (config.model.backbone), dictionary-style
    access (config["model"]["backbone"]), deep merging, and serialization
    back to YAML.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            value = self._data[key]
        except KeyError:
            raise AttributeError(
                f"Config has no attribute '{key}'. "
                f"Available keys: {list(self._data.keys())}"
            )
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data})"

    def __iter__(self):
        return iter(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """Safe access with default value."""
        try:
            value = self._data[key]
        except KeyError:
            return default
        if isinstance(value, dict):
            return Config(value)
        return value

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert back to plain dict."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, dict):
                result[key] = Config(value).to_dict()
            else:
                result[key] = value
        return result

    def merge(self, other: Config) -> Config:
        """Deep merge. Values from `other` take precedence."""
        merged = _deep_merge(self._data, other._data)
        return Config(merged)

    def save(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dicts. Override values take precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return a Config object.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return Config(data)


def load_and_merge_configs(*paths: str | Path) -> Config:
    """Load multiple YAML files and merge them left-to-right.

    Later configs override earlier ones for duplicate keys.
    """
    if not paths:
        raise ValueError("At least one config path is required")

    config = load_config(paths[0])
    for path in paths[1:]:
        other = load_config(path)
        config = config.merge(other)

    return config
