"""Episode shards on disk: plain ``.npy`` files that are memory-mapped for training.

A shard is a directory holding ``E`` equally long episodes::

    frames.npy   uint8 [E, T + 1, *frame_shape]   observation before each action, plus the last
    actions.npy  uint8 [E, T]                     action taken after frames[:, t]
    states.npy   int64 [E, T + 1, S]              engine state behind every frame (for probes/eval)
    meta.json    game, policy mix and seed

Fixed-length episodes keep indexing trivial: window ``(e, t)`` is frames ``t .. t + n``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class Shard:
    """A memory-mapped shard (read-only views)."""

    frames: npt.NDArray[np.uint8]
    actions: npt.NDArray[np.uint8]
    states: npt.NDArray[np.int64]
    meta: dict[str, object]

    @property
    def episodes(self) -> int:
        """Number of episodes."""
        return int(self.frames.shape[0])

    @property
    def steps(self) -> int:
        """Actions per episode."""
        return int(self.actions.shape[1])


def write_shard(
    directory: Path,
    frames: npt.NDArray[np.uint8],
    actions: npt.NDArray[np.uint8],
    states: npt.NDArray[np.int64],
    meta: dict[str, object],
) -> None:
    """Validate shapes and write one shard atomically enough for a single writer."""
    if frames.dtype != np.uint8 or actions.dtype != np.uint8:
        msg = f"frames/actions must be uint8, got {frames.dtype}/{actions.dtype}"
        raise TypeError(msg)
    e, t = actions.shape
    if frames.shape[:2] != (e, t + 1) or states.shape[:2] != (e, t + 1):
        msg = (
            f"inconsistent shard shapes: frames {frames.shape}, actions {actions.shape}, "
            f"states {states.shape}"
        )
        raise ValueError(msg)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "frames.npy", frames)
    np.save(directory / "actions.npy", actions)
    np.save(directory / "states.npy", states.astype(np.int64))
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def read_shard(directory: Path) -> Shard:
    """Memory-map a shard written by :func:`write_shard`."""
    missing = [
        n for n in ("frames.npy", "actions.npy", "states.npy", "meta.json") if not (directory / n).exists()
    ]
    if missing:
        msg = f"{directory} is not a shard (missing {', '.join(missing)}); run `dreamcade gen` first"
        raise FileNotFoundError(msg)
    frames = np.load(directory / "frames.npy", mmap_mode="r")
    actions = np.load(directory / "actions.npy", mmap_mode="r")
    states = np.load(directory / "states.npy", mmap_mode="r")
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    return Shard(frames=frames, actions=actions, states=states, meta=meta)


def list_shards(root: Path) -> list[Path]:
    """Shard directories under ``root`` in a stable order."""
    shards = sorted(p.parent for p in root.glob("shard_*/meta.json"))
    if not shards:
        msg = f"no shards under {root}; run `dreamcade gen` first"
        raise FileNotFoundError(msg)
    return shards
