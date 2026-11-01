"""Vectorised, integer-exact dungeon movement.

State is three integer arrays: ``x`` and ``y`` in fixed point (``FIXED`` units per tile) and a
heading index ``a`` in ``[0, NUM_ANGLES)``. All arithmetic is integer, so a batch of B agents
stepped here matches the scalar JavaScript port bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np
import numpy.typing as npt

from dreamcade.dungeon.world import FIXED, NUM_ANGLES, ORB_COLLIDE, WORLD, World

I64 = npt.NDArray[np.int64]


class Action(IntEnum):
    """Discrete controls. Strafing is split into two actions so every action is one key."""

    NOOP = 0
    FORWARD = 1
    BACK = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    STRAFE_LEFT = 5
    STRAFE_RIGHT = 6


NUM_ACTIONS: Final = len(Action)
MOVE_SPEED: Final = 32  # fixed-point units per step (1/8 tile)
STRAFE_SPEED: Final = 24
RADIUS: Final = 64  # player half-extent for wall collision (1/4 tile)


@dataclass(slots=True)
class DungeonState:
    """A batch of agents. Arrays are int64 with shape ``[B]``."""

    x: I64
    y: I64
    a: I64

    def copy(self) -> DungeonState:
        """Deep copy (the engine mutates nothing, but policies may)."""
        return DungeonState(self.x.copy(), self.y.copy(), self.a.copy())

    def as_array(self) -> I64:
        """``[B, 3]`` integer view used by datasets."""
        return np.stack([self.x, self.y, self.a], axis=1)

    @classmethod
    def from_array(cls, arr: npt.ArrayLike) -> DungeonState:
        """Inverse of :meth:`as_array`."""
        data = np.asarray(arr, dtype=np.int64).reshape(-1, 3)
        return cls(data[:, 0].copy(), data[:, 1].copy(), data[:, 2].copy())


def _blocked(world: World, px: I64, py: I64) -> npt.NDArray[np.bool_]:
    """True where the player's bounding box at ``(px, py)`` overlaps a wall tile."""
    blocked = np.zeros(px.shape, dtype=np.bool_)
    for ox in (-RADIUS, RADIUS - 1):
        for oy in (-RADIUS, RADIUS - 1):
            cx = (px + ox) >> 8
            cy = (py + oy) >> 8
            blocked |= world.walls[cy, cx] != 0
    return blocked


def _hits_orb(world: World, px: I64, py: I64) -> npt.NDArray[np.bool_]:
    hit = np.zeros(px.shape, dtype=np.bool_)
    for ox, oy in world.orbs_xy:
        dx = px - ox
        dy = py - oy
        hit |= dx * dx + dy * dy < ORB_COLLIDE * ORB_COLLIDE
    return hit


def step(
    state: DungeonState, actions: npt.ArrayLike, world: World = WORLD
) -> tuple[DungeonState, npt.NDArray[np.bool_]]:
    """Advance every agent one tick.

    Returns the new state and a ``moved`` mask (False when a translation was fully blocked),
    which exploration policies use to notice walls.
    """
    act = np.asarray(actions, dtype=np.int64)
    if act.shape != state.x.shape:
        msg = f"actions shape {act.shape} does not match batch shape {state.x.shape}"
        raise ValueError(msg)
    if np.any((act < 0) | (act >= NUM_ACTIONS)):
        msg = f"actions must be in [0, {NUM_ACTIONS})"
        raise ValueError(msg)

    a = state.a.copy()
    a = np.where(act == Action.TURN_LEFT, (a - 1) & (NUM_ANGLES - 1), a)
    a = np.where(act == Action.TURN_RIGHT, (a + 1) & (NUM_ANGLES - 1), a)

    cos = world.cos[state.a]
    sin = world.sin[state.a]
    # Forward is (cos, sin); strafing right is the heading rotated by +90 degrees: (-sin, cos).
    fwd = np.where(act == Action.FORWARD, MOVE_SPEED, 0) - np.where(act == Action.BACK, MOVE_SPEED, 0)
    side = np.where(act == Action.STRAFE_RIGHT, STRAFE_SPEED, 0) - np.where(
        act == Action.STRAFE_LEFT, STRAFE_SPEED, 0
    )
    dx = (cos * fwd - sin * side) >> 14
    dy = (sin * fwd + cos * side) >> 14

    # Axis-separated collision lets the agent slide along walls instead of sticking.
    nx = state.x + dx
    nx = np.where(_blocked(world, nx, state.y), state.x, nx)
    ny = state.y + dy
    ny = np.where(_blocked(world, nx, ny), state.y, ny)
    orb = _hits_orb(world, nx, ny)
    nx = np.where(orb, state.x, nx)
    ny = np.where(orb, state.y, ny)

    translating = (fwd != 0) | (side != 0)
    moved = ~translating | (nx != state.x) | (ny != state.y)
    return DungeonState(nx, ny, a), moved


def spawn(rng: np.random.Generator, batch: int, world: World = WORLD) -> DungeonState:
    """Random valid starting states: tile centres of free cells with random headings."""
    ys, xs = np.nonzero(world.walls == 0)
    centres_x = xs.astype(np.int64) * FIXED + FIXED // 2
    centres_y = ys.astype(np.int64) * FIXED + FIXED // 2
    valid = ~_hits_orb(world, centres_x, centres_y)
    idx = rng.choice(np.flatnonzero(valid), size=batch)
    a = rng.integers(0, NUM_ANGLES, size=batch, dtype=np.int64)
    return DungeonState(centres_x[idx].copy(), centres_y[idx].copy(), a)
