"""Command-line wiring and the static server's guard rails."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dreamcade.cli import build_parser
from dreamcade.serve import serve

ROOT = Path(__file__).resolve().parents[1]


def test_play_defaults_to_the_repo_web_folder() -> None:
    args = build_parser().parse_args(["play"])
    assert args.port == 8095 and args.host == "127.0.0.1"
    assert (Path(args.web) / "index.html").exists()


def test_train_requires_a_known_game() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "pong", "--config", "x.toml"])


def test_serve_refuses_a_folder_without_the_player(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"no index\.html"):
        serve(tmp_path, "127.0.0.1", 0)


def test_shipped_manifest_matches_its_parts() -> None:
    manifest_path = ROOT / "web" / "models" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("models not exported in this checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for game in ("dungeon", "breakout"):
        entry = manifest[game]
        size = sum((manifest_path.parent / p).stat().st_size for p in entry["parts"])
        assert size == entry["bytes"], f"{game} parts do not add up to the manifest size"
        assert all((manifest_path.parent / p).stat().st_size < 20 * 1024 * 1024 for p in entry["parts"])
