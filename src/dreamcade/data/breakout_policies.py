"""Breakout behaviour policies.

A good player alone never shows the model a lost life, and a random one rarely breaks a
brick, so every episode mixes four styles and switches between them:

* ``TRACKER`` follows the ball with a random aim offset (long rallies, fast balls, clears)
* ``RANDOM``  uniform actions
* ``STICKY``  repeats its previous action 90% of the time (long sweeps across the field)
* ``LOSER``   moves away from the ball (lost lives, game over, restarts)
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import numpy as np

from dreamcade.game.constants import NUM_ACTIONS, PADDLE_W, Action, Mode
from dreamcade.game.engine import GameState


class Style(IntEnum):
    """Policy style."""

    TRACKER = 0
    RANDOM = 1
    STICKY = 2
    LOSER = 3


STYLE_PROBS: Final = (0.5, 0.15, 0.2, 0.15)


class BreakoutPlayer:
    """One scripted player; ``act`` maps a state to an action."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._style = Style.TRACKER
        self._left = 0
        self._aim = 0
        self._prev = Action.NOOP

    @property
    def style(self) -> Style:
        """The current style (for coverage statistics)."""
        return self._style

    def _switch(self) -> None:
        self._style = Style(int(self._rng.choice(len(Style), p=STYLE_PROBS)))
        self._left = int(self._rng.integers(60, 400))
        self._aim = int(self._rng.integers(-PADDLE_W // 2, PADDLE_W // 2 + 1))

    def act(self, state: GameState) -> Action:
        """Pick the next action."""
        self._left -= 1
        if self._left <= 0:
            self._switch()
        rng = self._rng
        if state.mode != Mode.PLAY and self._style != Style.RANDOM and rng.random() < 0.15:
            action = Action.FIRE
        elif self._style == Style.RANDOM:
            action = Action(int(rng.integers(NUM_ACTIONS)))
        elif self._style == Style.STICKY:
            action = self._prev if rng.random() < 0.9 else Action(int(rng.integers(NUM_ACTIONS)))
        else:
            centre = state.paddle_x + PADDLE_W // 2 + self._aim
            error = state.ball_x - centre
            if self._style == Style.LOSER:
                error = -error
            if state.mode != Mode.PLAY or abs(error) <= 1 or rng.random() < 0.1:
                action = Action.NOOP
            else:
                action = Action.RIGHT if error > 0 else Action.LEFT
        self._prev = action
        return action
