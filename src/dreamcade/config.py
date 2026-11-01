"""Typed configuration loaded from TOML files in ``configs/``.

Unknown keys are an error: a typo in a config silently falling back to a default is
exactly the kind of bug that wastes a GPU run.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dreamcade.data.generate import GenSpec


@dataclass(frozen=True, slots=True)
class DungeonTrainConfig:
    """EDM diffusion world model for the dungeon."""

    data: str
    out: str
    seed: int = 0
    context: int = 4  # number of past frames (and actions) the model sees
    channels: tuple[int, ...] = (96, 192, 256)
    blocks: int = 2  # residual blocks per level
    attn_levels: tuple[int, ...] = (2,)  # levels (0 = finest) with self-attention
    emb_dim: int = 256
    batch: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-2
    warmup: int = 500
    max_steps: int = 100_000
    max_minutes: float = 50.0  # wall-clock budget; training stops cleanly when reached
    ema: float = 0.999
    ctx_noise_max: float = 0.3  # context frames get N(0, s^2) noise with s ~ U(0, max); 0 disables
    ctx_noise_zero_prob: float = 0.1
    holdout_episodes: int = 16  # last episodes of the last shard, never trained on
    log_every: int = 50
    ckpt_every: int = 2000
    bf16: bool = True
    data_on_gpu: bool = True
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class BreakoutTrainConfig:
    """Per-pixel palette classifier for Breakout."""

    data: str
    out: str
    seed: int = 0
    context: int = 4
    channels: tuple[int, ...] = (48, 96, 192)
    batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup: int = 200
    max_steps: int = 50_000
    max_minutes: float = 6.0
    ema: float = 0.999
    ctx_flip_max: float = 0.0  # context pixels re-coloured at random with prob p ~ U(0, max)
    changed_weight: float = 10.0  # loss weight on pixels that differ from the last context frame
    holdout_episodes: int = 8
    log_every: int = 50
    ckpt_every: int = 2000
    bf16: bool = True
    data_on_gpu: bool = True
    device: str = "auto"


def _build[T](cls: type[T], raw: dict[str, Any], source: Path) -> T:
    names = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(raw) - set(names))
    if unknown:
        msg = f"{source}: unknown keys {unknown}; valid keys are {sorted(names)}"
        raise ValueError(msg)
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    try:
        return cls(**values)
    except TypeError as err:
        msg = f"{source}: {err}"
        raise ValueError(msg) from err


def load_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file with a clear error when it is missing."""
    if not path.exists():
        msg = f"config file {path} does not exist"
        raise FileNotFoundError(msg)
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_gen(path: Path) -> GenSpec:
    """A ``[gen]`` table -> :class:`GenSpec`."""
    raw = load_toml(path)
    if "gen" not in raw:
        msg = f"{path}: missing [gen] table"
        raise ValueError(msg)
    return _build(GenSpec, raw["gen"], path)


def load_dungeon(path: Path) -> DungeonTrainConfig:
    """A ``[train]`` table -> :class:`DungeonTrainConfig`."""
    return _build(DungeonTrainConfig, load_toml(path).get("train", {}), path)


def load_breakout(path: Path) -> BreakoutTrainConfig:
    """A ``[train]`` table -> :class:`BreakoutTrainConfig`."""
    return _build(BreakoutTrainConfig, load_toml(path).get("train", {}), path)


def config_from_dict[T](cls: type[T], raw: dict[str, Any]) -> T:
    """Rebuild a config stored in a checkpoint (lists become tuples, unknown keys are errors)."""
    return _build(cls, raw, Path("<checkpoint>"))
