"""Golden trajectories that pin the JavaScript ports to the Python engines.

Each file holds an action sequence, the engine state after every step and a 32-bit FNV-1a
hash of every rendered frame. ``tests/js/parity.mjs`` replays the actions through the JS
port and must reproduce every state and hash exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from dreamcade.data.breakout_policies import BreakoutPlayer
from dreamcade.data.dungeon_policies import Explorer
from dreamcade.dungeon import engine as dengine
from dreamcade.dungeon.render import render as drender
from dreamcade.game import engine as bengine
from dreamcade.game.render import render as brender

FNV_OFFSET = 2166136261
FNV_PRIME = 16777619


def fnv1a(data: npt.NDArray[np.uint8]) -> int:
    """32-bit FNV-1a of a byte array (vectorised over chunks would change the result; keep it scalar)."""
    h = FNV_OFFSET
    for byte in data.reshape(-1).tobytes():
        h = ((h ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return h


def breakout_golden(steps: int, seed: int) -> dict[str, object]:
    """Play ``steps`` ticks with the mixed-style player."""
    rng = np.random.default_rng(seed)
    state = bengine.new_game()
    player = BreakoutPlayer(rng)
    actions, states, hashes = [], [state.to_dict()], [fnv1a(brender(state))]
    for _ in range(steps):
        action = int(player.act(state))
        state, _ = bengine.step(state, action)
        actions.append(action)
        states.append(state.to_dict())
        hashes.append(fnv1a(brender(state)))
    return {"game": "breakout", "seed": seed, "actions": actions, "states": states, "hashes": hashes}


def dungeon_golden(steps: int, seed: int) -> dict[str, object]:
    """Explore for ``steps`` ticks from a random spawn."""
    rng = np.random.default_rng(seed)
    state = dengine.spawn(rng, 1)
    policy = Explorer(rng, 1)
    moved = np.ones(1, dtype=np.bool_)
    actions: list[int] = []
    states = [state.as_array()[0].tolist()]
    hashes = [fnv1a(drender(state)[0])]
    for _ in range(steps):
        act = policy.act(state, moved)
        state, moved = dengine.step(state, act)
        actions.append(int(act[0]))
        states.append(state.as_array()[0].tolist())
        hashes.append(fnv1a(drender(state)[0]))
    return {"game": "dungeon", "seed": seed, "actions": actions, "states": states, "hashes": hashes}


def write_golden(out: Path, steps: int, seed: int) -> None:
    """Write ``breakout.json`` and ``dungeon.json`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("breakout", breakout_golden(steps * 2, seed)),
        ("dungeon", dungeon_golden(steps, seed)),
    ):
        (out / f"{name}.json").write_text(
            json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n"
        )
