from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[2]


ROOT = repo_root()


class Config(Mapping):
    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    def __getattr__(self, item: str) -> Any:
        try:
            value = self._data[item]
        except KeyError as exc:
            raise AttributeError(f"no config key {item!r}") from exc
        return Config(value) if isinstance(value, Mapping) else value

    def __getitem__(self, item: str) -> Any:
        value = self._data[item]
        return Config(value) if isinstance(value, Mapping) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (v.to_dict() if isinstance(v, Config) else v)
            for k, v in ((k, self[k]) for k in self._data)
        }

    def path(self, key: str) -> Path:
        raw = Path(str(self.paths[key]))
        p = raw if raw.is_absolute() else ROOT / raw
        p.mkdir(parents=True, exist_ok=True)
        return p


def _coerce(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_dotted_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key.subkey=value")
        dotted, raw = item.split("=", 1)
        cursor = data
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"override {dotted!r} traverses a non-mapping key")
        cursor[parts[-1]] = _coerce(raw.strip())
    return data


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    cfg_path = Path(path) if path else ROOT / "conf" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    env_overrides = [
        f"{key[len('CABCAST_'):].lower().replace('__', '.')}={value}"
        for key, value in os.environ.items()
        if key.startswith("CABCAST_")
    ]
    if env_overrides:
        data = apply_dotted_overrides(data, env_overrides)
    if overrides:
        data = apply_dotted_overrides(data, list(overrides))
    return Config(data)


@lru_cache(maxsize=1)
def default_config() -> Config:
    return load_config()
