"""Vectorised exploration policies for the dungeon.

A world model only learns the parts of the world its data visits, and it only learns to
turn around if the data turns around. Each agent therefore switches between four
behaviours every few dozen steps:

* ``WALK``   momentum random walk (actions are sticky, forward-heavy)
* ``FOLLOW`` wall following: walk forward, turn away when a move is blocked
* ``SPIN``   turn in place, which is exactly the "what is behind me" case
* ``GOTO``   shortest-path navigation to a random room, so rooms are revisited from new sides
"""

from __future__ import annotations

import math
from collections import deque
from enum import IntEnum
from typing import Final

import numpy as np
import numpy.typing as npt

from dreamcade.dungeon.engine import NUM_ACTIONS, Action, DungeonState
from dreamcade.dungeon.world import FIXED, NUM_ANGLES, ROOMS, SIZE, WORLD, World

I64 = npt.NDArray[np.int64]


class Mode(IntEnum):
    """Behaviour of one agent."""

    WALK = 0
    FOLLOW = 1
    SPIN = 2
    GOTO = 3


MODE_PROBS: Final = np.array([0.35, 0.2, 0.15, 0.3])
WALK_PROBS: Final = np.array([0.04, 0.46, 0.08, 0.13, 0.13, 0.08, 0.08])
UNREACHABLE: Final = 10_000


def distance_fields(world: World = WORLD) -> I64:
    """BFS step distance from every tile to each room's centre tile: ``[num_rooms, SIZE, SIZE]``."""
    fields = np.full((len(ROOMS), SIZE, SIZE), UNREACHABLE, dtype=np.int64)
    for k, (x0, y0, x1, y1) in enumerate(ROOMS.values()):
        goal = ((x0 + x1) // 2, (y0 + y1) // 2)
        fields[k, goal[1], goal[0]] = 0
        queue = deque([goal])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if world.walls[ny, nx] == 0 and fields[k, ny, nx] == UNREACHABLE:
                    fields[k, ny, nx] = fields[k, y, x] + 1
                    queue.append((nx, ny))
    return fields


class Explorer:
    """Stateful batch policy. ``act`` is called once per step with the engine feedback."""

    def __init__(self, rng: np.random.Generator, batch: int, world: World = WORLD) -> None:
        self._rng = rng
        self._world = world
        self._fields = distance_fields(world)
        self.mode = rng.choice(len(Mode), size=batch, p=MODE_PROBS).astype(np.int64)
        self._left = rng.integers(20, 120, size=batch)
        self._prev = np.full(batch, Action.FORWARD, dtype=np.int64)
        self._spin_dir = rng.choice([Action.TURN_LEFT, Action.TURN_RIGHT], size=batch).astype(np.int64)
        self._target = rng.integers(0, len(ROOMS), size=batch)
        self._turn_left = np.zeros(batch, dtype=np.int64)

    def _switch(self, which: npt.NDArray[np.bool_]) -> None:
        n = int(which.sum())
        if n == 0:
            return
        self.mode[which] = self._rng.choice(len(Mode), size=n, p=MODE_PROBS)
        self._left[which] = self._rng.integers(20, 150, size=n)
        spin = which & (self.mode == Mode.SPIN)
        self._left[spin] = self._rng.integers(8, 70, size=int(spin.sum()))
        self._spin_dir[which] = self._rng.choice([Action.TURN_LEFT, Action.TURN_RIGHT], size=n)
        self._target[which] = self._rng.integers(0, len(ROOMS), size=n)

    def _goto(self, state: DungeonState) -> I64:
        cx = state.x >> 8
        cy = state.y >> 8
        field = self._fields[self._target]
        idx = np.arange(cx.shape[0])
        best = field[idx, cy, cx]
        tx, ty = cx.copy(), cy.copy()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            d = field[idx, cy + dy, cx + dx]
            better = d < best
            best = np.where(better, d, best)
            tx = np.where(better, cx + dx, tx)
            ty = np.where(better, cy + dy, ty)
        gx = tx * FIXED + FIXED // 2 - state.x
        gy = ty * FIXED + FIXED // 2 - state.y
        want = np.array(
            [
                round(math.atan2(float(y), float(x)) / (2 * math.pi) * NUM_ANGLES) % NUM_ANGLES
                for x, y in zip(gx, gy, strict=True)
            ],
            dtype=np.int64,
        )
        diff = (want - state.a) % NUM_ANGLES
        act = np.full(cx.shape, Action.FORWARD, dtype=np.int64)
        act = np.where((diff >= 2) & (diff < NUM_ANGLES // 2), Action.TURN_RIGHT, act)
        act = np.where((diff >= NUM_ANGLES // 2) & (diff <= NUM_ANGLES - 2), Action.TURN_LEFT, act)
        arrived = (best == 0) | (best >= UNREACHABLE)
        self._left = np.where(arrived & (self.mode == Mode.GOTO), 0, self._left)
        return act

    def act(self, state: DungeonState, moved: npt.NDArray[np.bool_]) -> I64:
        """Choose the next action for every agent."""
        self._left -= 1
        self._switch(self._left <= 0)
        b = state.x.shape[0]
        rng = self._rng

        fresh = rng.choice(NUM_ACTIONS, size=b, p=WALK_PROBS)
        keep = (rng.random(b) < 0.85) & moved
        walk = np.where(keep, self._prev, fresh)

        # Wall following: after a bump, keep turning for a few steps, then go straight again.
        bumped = ~moved
        self._turn_left = np.where(
            bumped & (self._turn_left <= 0), rng.integers(3, 12, size=b), self._turn_left
        )
        turning = self._turn_left > 0
        self._turn_left = np.maximum(self._turn_left - 1, 0)
        follow = np.where(turning, Action.TURN_LEFT, Action.FORWARD)
        follow = np.where(~turning & (rng.random(b) < 0.06), Action.TURN_RIGHT, follow)

        goto = self._goto(state)
        act = np.select(
            [self.mode == Mode.WALK, self.mode == Mode.FOLLOW, self.mode == Mode.SPIN],
            [walk, follow, self._spin_dir],
            default=goto,
        ).astype(np.int64)
        # A little action noise everywhere keeps every (state, action) pair in the data.
        noise = rng.random(b) < 0.05
        act = np.where(noise, rng.integers(0, NUM_ACTIONS, size=b), act)
        self._prev = act
        return act
