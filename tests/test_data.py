"""Shard generation, storage and coverage statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dreamcade.config import load_gen
from dreamcade.data.generate import BREAKOUT_STATE_FIELDS, GenSpec, coverage, generate
from dreamcade.data.shards import list_shards, read_shard, write_shard

ROOT = Path(__file__).resolve().parents[1]


def test_dungeon_shards_shapes(tmp_path: Path) -> None:
    generate(GenSpec("dungeon", shards=2, episodes_per_shard=3, steps=12, seed=0), tmp_path, workers=1)
    shards = [read_shard(p) for p in list_shards(tmp_path)]
    assert len(shards) == 2
    s = shards[0]
    assert s.frames.shape == (3, 13, 64, 64, 3) and s.frames.dtype == np.uint8
    assert s.actions.shape == (3, 12) and int(s.actions.max()) < 7
    assert s.states.shape == (3, 13, 3)
    stats = coverage(tmp_path)
    assert stats["frames"] == 2 * 3 * 13
    hist = stats["action_hist"]
    assert isinstance(hist, list)
    assert abs(sum(hist) - 1) < 1e-3


def test_breakout_shards_shapes(tmp_path: Path) -> None:
    generate(GenSpec("breakout", shards=1, episodes_per_shard=2, steps=30, seed=0), tmp_path, workers=1)
    s = read_shard(list_shards(tmp_path)[0])
    assert s.frames.shape == (2, 31, 64, 64)
    assert s.states.shape == (2, 31, len(BREAKOUT_STATE_FIELDS))
    assert int(s.frames.max()) < 8


def test_generation_is_deterministic(tmp_path: Path) -> None:
    spec = GenSpec("dungeon", shards=1, episodes_per_shard=2, steps=8, seed=3)
    generate(spec, tmp_path / "a", workers=1)
    generate(spec, tmp_path / "b", workers=1)
    a, b = read_shard(tmp_path / "a" / "shard_000"), read_shard(tmp_path / "b" / "shard_000")
    assert np.array_equal(a.frames, b.frames) and np.array_equal(a.actions, b.actions)


def test_write_shard_validates(tmp_path: Path) -> None:
    frames = np.zeros((2, 5, 64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="inconsistent"):
        write_shard(
            tmp_path, frames, np.zeros((2, 3), dtype=np.uint8), np.zeros((2, 5, 1), dtype=np.int64), {}
        )


def test_missing_shards_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dreamcade gen"):
        list_shards(tmp_path)


def test_gen_configs_parse() -> None:
    for name in ("gen_dungeon.toml", "gen_breakout.toml", "gen_tiny.toml"):
        spec = load_gen(ROOT / "configs" / name)
        assert spec.shards >= 1 and spec.steps > 0
