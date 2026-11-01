"""Write episode shards for both games and summarise their coverage."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from dreamcade.data.breakout_policies import BreakoutPlayer
from dreamcade.data.dungeon_policies import Explorer
from dreamcade.data.shards import list_shards, read_shard, write_shard
from dreamcade.dungeon import engine as dengine
from dreamcade.dungeon.render import render as drender
from dreamcade.dungeon.world import ROOM_NAMES, SIZE, WORLD
from dreamcade.game import engine as bengine
from dreamcade.game.constants import NUM_ACTIONS as BREAKOUT_ACTIONS
from dreamcade.game.constants import PADDLE_MAX_X, PADDLE_MIN_X
from dreamcade.game.render import render as brender

#: Columns of the Breakout ``states.npy`` array.
BREAKOUT_STATE_FIELDS = (
    "paddle_x",
    "ball_x",
    "ball_y",
    "vx",
    "vy",
    "mode",
    "score",
    "lives",
    "bricks_left",
    "events",
)
DUNGEON_STATE_FIELDS = ("x", "y", "a")


@dataclass(frozen=True, slots=True)
class GenSpec:
    """How much data to write for one game."""

    game: str
    shards: int
    episodes_per_shard: int
    steps: int
    seed: int


def dungeon_shard(spec: GenSpec, index: int, out: Path) -> None:
    """One dungeon shard: ``episodes_per_shard`` agents explored in lockstep."""
    rng = np.random.default_rng([spec.seed, index])
    b, t = spec.episodes_per_shard, spec.steps
    state = dengine.spawn(rng, b)
    policy = Explorer(rng, b)
    frames = np.empty((b, t + 1, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((b, t), dtype=np.uint8)
    states = np.empty((b, t + 1, 3), dtype=np.int64)
    moved = np.ones(b, dtype=np.bool_)
    modes = np.zeros(4, dtype=np.int64)
    for step in range(t + 1):
        frames[:, step] = drender(state)
        states[:, step] = state.as_array()
        if step == t:
            break
        act = policy.act(state, moved)
        modes += np.bincount(policy.mode, minlength=4)
        actions[:, step] = act
        state, moved = dengine.step(state, act)
    meta: dict[str, object] = {
        "game": "dungeon",
        "seed": spec.seed,
        "index": index,
        "state_fields": list(DUNGEON_STATE_FIELDS),
        "mode_steps": {m: int(c) for m, c in zip(("walk", "follow", "spin", "goto"), modes, strict=True)},
    }
    write_shard(out / f"shard_{index:03d}", frames, actions, states, meta)


def breakout_shard(spec: GenSpec, index: int, out: Path) -> None:
    """One Breakout shard of independent games, each driven by a style-switching player."""
    rng = np.random.default_rng([spec.seed, index])
    e, t = spec.episodes_per_shard, spec.steps
    frames = np.empty((e, t + 1, 64, 64), dtype=np.uint8)
    actions = np.empty((e, t), dtype=np.uint8)
    states = np.zeros((e, t + 1, len(BREAKOUT_STATE_FIELDS)), dtype=np.int64)
    styles = np.zeros(4, dtype=np.int64)
    for ep in range(e):
        start_x = int(rng.integers(PADDLE_MIN_X, PADDLE_MAX_X + 1))
        state = bengine.new_game(start_x)
        player = BreakoutPlayer(rng)
        event = bengine.Event.NONE
        for step in range(t + 1):
            frames[ep, step] = brender(state)
            states[ep, step] = (
                state.paddle_x,
                state.ball_x,
                state.ball_y,
                state.vx,
                state.vy,
                int(state.mode),
                state.score,
                state.lives,
                state.bricks_left,
                int(event),
            )
            if step == t:
                break
            action = player.act(state)
            styles[int(player.style)] += 1
            actions[ep, step] = int(action)
            state, event = bengine.step(state, action)
    meta: dict[str, object] = {
        "game": "breakout",
        "seed": spec.seed,
        "index": index,
        "state_fields": list(BREAKOUT_STATE_FIELDS),
        "style_steps": {
            s: int(c) for s, c in zip(("tracker", "random", "sticky", "loser"), styles, strict=True)
        },
    }
    write_shard(out / f"shard_{index:03d}", frames, actions, states, meta)


WRITERS: dict[str, Callable[[GenSpec, int, Path], None]] = {
    "dungeon": dungeon_shard,
    "breakout": breakout_shard,
}


def generate(spec: GenSpec, out: Path, workers: int) -> None:
    """Write ``spec.shards`` shards in parallel, then the coverage summary."""
    if spec.game not in WRITERS:
        msg = f"unknown game {spec.game!r}; choose from {sorted(WRITERS)}"
        raise ValueError(msg)
    writer = WRITERS[spec.game]
    if workers <= 1:
        for i in range(spec.shards):
            writer(spec, i, out)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for f in [pool.submit(writer, spec, i, out) for i in range(spec.shards)]:
                f.result()
    stats = coverage(out)
    (out / "coverage.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def _histogram(values: npt.NDArray[np.int64], n: int) -> list[float]:
    counts = np.bincount(values.reshape(-1).astype(np.int64), minlength=n)[:n]
    return [round(float(c) / max(1, int(counts.sum())), 5) for c in counts]


def coverage(root: Path) -> dict[str, object]:
    """Summary statistics over every shard under ``root``."""
    shards = [read_shard(p) for p in list_shards(root)]
    game = str(shards[0].meta["game"])
    actions = np.concatenate([np.asarray(s.actions).reshape(-1) for s in shards])
    states = np.concatenate([np.asarray(s.states).reshape(-1, s.states.shape[-1]) for s in shards])
    frames = sum(int(s.frames.shape[0] * s.frames.shape[1]) for s in shards)
    stats: dict[str, object] = {
        "game": game,
        "shards": len(shards),
        "frames": frames,
        "transitions": int(actions.size),
    }
    if game == "dungeon":
        cx, cy = states[:, 0] >> 8, states[:, 1] >> 8
        visits = np.zeros((SIZE, SIZE), dtype=np.int64)
        np.add.at(visits, (cy, cx), 1)
        free = WORLD.walls == 0
        rooms = WORLD.rooms[cy, cx]
        stats |= {
            "action_hist": _histogram(actions.astype(np.int64), dengine.NUM_ACTIONS),
            "heading_hist": _histogram(states[:, 2], 64),
            "free_tiles_visited": round(float((visits[free] > 0).mean()), 4),
            "min_visits_per_free_tile": int(visits[free].min()),
            "room_fraction": {ROOM_NAMES[k]: round(float((rooms == k).mean()), 4) for k in ROOM_NAMES},
            "visits": visits.tolist(),
        }
    else:
        ev = states[:, 9]
        names = [e.name for e in bengine.Event if e.name and e.value]
        stats |= {
            "action_hist": _histogram(actions.astype(np.int64), BREAKOUT_ACTIONS),
            "mode_hist": _histogram(states[:, 5], 3),
            "events": {str(n).lower(): int(((ev & bengine.Event[str(n)].value) > 0).sum()) for n in names},
            "bricks_left_hist": _histogram(states[:, 8], 49),
            "max_score": int(states[:, 6].max()),
        }
    return stats
