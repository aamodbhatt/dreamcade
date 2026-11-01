"""Dungeon engine and renderer: determinism, invariants and the browser world file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from dreamcade.dungeon import engine
from dreamcade.dungeon.render import HEIGHT, WIDTH, render
from dreamcade.dungeon.world import FIXED, NUM_ANGLES, WORLD, world_to_json

ROOT = Path(__file__).resolve().parents[1]


def test_world_json_is_current() -> None:
    on_disk = (ROOT / "web" / "assets" / "dungeon_world.json").read_text(encoding="utf-8")
    assert on_disk == world_to_json(), "run `dreamcade world` to refresh web/assets/dungeon_world.json"


def test_outer_ring_is_solid() -> None:
    w = WORLD.walls
    assert (w[0] > 0).all() and (w[-1] > 0).all() and (w[:, 0] > 0).all() and (w[:, -1] > 0).all()


def test_render_shape_and_determinism() -> None:
    state = engine.spawn(np.random.default_rng(0), 5)
    a, b = render(state), render(state.copy())
    assert a.shape == (5, HEIGHT, WIDTH, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert a.std() > 10, "frames should not be blank"


def test_batched_render_matches_single() -> None:
    state = engine.spawn(np.random.default_rng(1), 4)
    batch = render(state)
    for i in range(4):
        single = engine.DungeonState.from_array(state.as_array()[i : i + 1])
        assert np.array_equal(render(single)[0], batch[i])


def test_turning_is_periodic() -> None:
    state = engine.spawn(np.random.default_rng(2), 3)
    s = state
    for _ in range(NUM_ANGLES):
        s, _ = engine.step(s, np.full(3, engine.Action.TURN_RIGHT))
    assert np.array_equal(s.as_array(), state.as_array())


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(0, engine.NUM_ACTIONS - 1), min_size=1, max_size=120), st.integers(0, 2**31 - 1))
def test_agents_never_enter_walls(actions: list[int], seed: int) -> None:
    state = engine.spawn(np.random.default_rng(seed), 4)
    for a in actions:
        state, _ = engine.step(state, np.full(4, a))
        for ox in (-engine.RADIUS, engine.RADIUS - 1):
            for oy in (-engine.RADIUS, engine.RADIUS - 1):
                cells = WORLD.walls[(state.y + oy) // FIXED, (state.x + ox) // FIXED]
                assert (cells == 0).all()


def test_blocked_moves_are_reported() -> None:
    # Face west from the left edge of the red hall: the wall is half a tile away.
    state = engine.DungeonState(
        np.array([1 * FIXED + 80]), np.array([3 * FIXED + 128]), np.array([NUM_ANGLES // 2])
    )
    moved_any = True
    for _ in range(3):
        state, moved = engine.step(state, np.array([engine.Action.FORWARD]))
        moved_any = bool(moved[0])
    assert not moved_any


def test_step_rejects_bad_actions() -> None:
    state = engine.spawn(np.random.default_rng(0), 2)
    try:
        engine.step(state, np.array([0, 9]))
    except ValueError as err:
        assert "actions must be" in str(err)
    else:
        raise AssertionError("expected ValueError")
