"""CPU smoke tests: tiny training runs (with resume) and ONNX export parity."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from dreamcade.config import BreakoutTrainConfig, DungeonTrainConfig, load_breakout, load_dungeon
from dreamcade.data.generate import GenSpec, generate
from dreamcade.export import export
from dreamcade.train import breakout as train_breakout
from dreamcade.train import dungeon as train_dungeon

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("data")
    generate(
        GenSpec("dungeon", shards=1, episodes_per_shard=3, steps=16, seed=0), root / "dungeon", workers=1
    )
    generate(
        GenSpec("breakout", shards=1, episodes_per_shard=3, steps=24, seed=0), root / "breakout", workers=1
    )
    return {"dungeon": root / "dungeon", "breakout": root / "breakout"}


def _records(path: Path) -> list[dict[str, float]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_dungeon_training_smoke_and_resume(tiny_data: dict[str, Path], tmp_path: Path) -> None:
    cfg = DungeonTrainConfig(
        data=str(tiny_data["dungeon"]),
        out=str(tmp_path / "run"),
        context=2,
        channels=(32, 64),
        attn_levels=(1,),
        emb_dim=64,
        batch=2,
        max_steps=4,
        holdout_episodes=1,
        log_every=2,
        ckpt_every=2,
        bf16=False,
        device="cpu",
    )
    train_dungeon.train(cfg)
    first = _records(tmp_path / "run" / "metrics.jsonl")
    assert any("train/loss" in r and "_time" in r for r in first)
    assert any("val/psnr_1step" in r for r in first)
    train_dungeon.train(dataclasses.replace(cfg, max_steps=6))
    steps = [r["step"] for r in _records(tmp_path / "run" / "metrics.jsonl") if "train/loss" in r]
    assert steps[-1] == 6 and 2 in steps


def test_breakout_training_smoke(tiny_data: dict[str, Path], tmp_path: Path) -> None:
    cfg = BreakoutTrainConfig(
        data=str(tiny_data["breakout"]),
        out=str(tmp_path / "run"),
        channels=(16, 32),
        batch=4,
        max_steps=4,
        holdout_episodes=1,
        log_every=2,
        ckpt_every=2,
        bf16=False,
        device="cpu",
        ctx_flip_max=0.05,
    )
    train_breakout.train(cfg)
    recs = _records(tmp_path / "run" / "metrics.jsonl")
    assert any("val/frame_exact" in r for r in recs)


def test_onnx_export_parity(tiny_data: dict[str, Path], tmp_path: Path) -> None:
    pytest.importorskip("onnxscript")
    d = DungeonTrainConfig(
        data=str(tiny_data["dungeon"]),
        out=str(tmp_path / "d"),
        context=2,
        channels=(32, 64),
        attn_levels=(1,),
        emb_dim=64,
        batch=2,
        max_steps=1,
        holdout_episodes=1,
        bf16=False,
        device="cpu",
    )
    b = BreakoutTrainConfig(
        data=str(tiny_data["breakout"]),
        out=str(tmp_path / "b"),
        channels=(16, 32),
        batch=2,
        max_steps=1,
        holdout_episodes=1,
        bf16=False,
        device="cpu",
    )
    train_dungeon.train(d)
    train_breakout.train(b)
    out = tmp_path / "web"
    export("dungeon", Path(d.out), out)
    export("breakout", Path(b.out), out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dungeon"]["parity"]["max_abs_diff"] < 1e-3
    assert manifest["breakout"]["parity"]["pixel_mismatch"] == 0.0
    joined = b"".join((out / p).read_bytes() for p in manifest["dungeon"]["parts"])
    assert len(joined) == manifest["dungeon"]["bytes"]


def test_pod_and_laptop_configs_parse() -> None:
    assert load_dungeon(ROOT / "configs" / "dungeon_pod.toml").context == 4
    assert load_dungeon(ROOT / "configs" / "dungeon_laptop.toml").batch == 4
    assert load_breakout(ROOT / "configs" / "breakout_cn.toml").ctx_flip_max > 0


def test_unknown_config_key_is_an_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('[train]\ndata = "x"\nout = "y"\nlearning_rate = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_dungeon(bad)
