"""Python <-> JavaScript parity via golden trajectories."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dreamcade.golden import breakout_golden, dungeon_golden, fnv1a

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"


def test_golden_files_are_reproducible() -> None:
    """The committed goldens are exactly what the Python engines produce today."""
    for name, make in (("breakout", breakout_golden), ("dungeon", dungeon_golden)):
        stored = json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))
        fresh = make(len(stored["actions"]), stored["seed"])
        assert fresh == stored, f"{name} golden drifted: run `dreamcade golden --out tests/golden`"


def test_goldens_cover_interesting_events() -> None:
    stored = json.loads((GOLDEN / "breakout.json").read_text(encoding="utf-8"))
    scores = [s["score"] for s in stored["states"]]
    lives = [s["lives"] for s in stored["states"]]
    assert max(scores) > 0, "golden should break bricks"
    assert min(lives) < 3, "golden should lose a life"
    dungeon = json.loads((GOLDEN / "dungeon.json").read_text(encoding="utf-8"))
    assert len({tuple(s) for s in dungeon["states"]}) > 100


def test_fnv1a_known_value() -> None:
    import numpy as np

    assert fnv1a(np.frombuffer(b"a", dtype=np.uint8)) == 0xE40C292C


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_ports_match_python() -> None:
    proc = subprocess.run(
        ["node", str(ROOT / "tests" / "js" / "parity.mjs")], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "breakout: 601 frames identical" in proc.stdout
    assert "dungeon: 301 frames identical" in proc.stdout
