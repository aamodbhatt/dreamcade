"""Geometry, palette and action space of the dreamcade Breakout.

Everything is expressed in integer pixel units on a 64x64 canvas so the Python engine
and its JavaScript port can agree bit-for-bit. The same numbers are mirrored in
``web/js/game.js``; the golden-trajectory tests fail if the two ever drift apart.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

# Canvas -------------------------------------------------------------------------------
WIDTH: Final = 64
HEIGHT: Final = 64


class Color(IntEnum):
    """Palette indices. A frame is a ``uint8[64, 64]`` array of these values."""

    BACKGROUND = 0
    WALL = 1  # side/top walls and the score digits
    PADDLE = 2  # paddle and the lives pips
    BALL = 3
    BRICK_RED = 4
    BRICK_ORANGE = 5
    BRICK_YELLOW = 6
    BRICK_GREEN = 7


NUM_COLORS: Final = len(Color)

#: RGB for each palette index (used by plots, GIFs and mirrored in the web player).
PALETTE_RGB: Final[tuple[tuple[int, int, int], ...]] = (
    (10, 10, 24),
    (142, 146, 168),
    (64, 196, 255),
    (250, 250, 250),
    (239, 71, 111),
    (255, 159, 28),
    (255, 214, 10),
    (6, 214, 160),
)


class Action(IntEnum):
    """Discrete controls, in the same order as the ALE minimal Breakout action set."""

    NOOP = 0
    FIRE = 1
    LEFT = 2
    RIGHT = 3


NUM_ACTIONS: Final = len(Action)


class Mode(IntEnum):
    """High-level game phase. Every phase is recoverable from two consecutive frames."""

    SERVE = 0  # ball rests on the paddle until FIRE
    PLAY = 1  # ball in flight
    GAME_OVER = 2  # no ball on screen; FIRE starts a new game


# Walls and play field -----------------------------------------------------------------
HUD_HEIGHT: Final = 8  # rows 0..7 hold the score and lives
WALL_THICKNESS: Final = 2
TOP_WALL_Y: Final = HUD_HEIGHT  # rows 8..9
PLAY_X0: Final = WALL_THICKNESS  # first free column
PLAY_X1: Final = WIDTH - WALL_THICKNESS - 1  # last free column (61)
PLAY_Y0: Final = TOP_WALL_Y + WALL_THICKNESS  # first free row (10)

# Bricks -------------------------------------------------------------------------------
BRICK_ROWS: Final = 4
BRICK_COLS: Final = 12
NUM_BRICKS: Final = BRICK_ROWS * BRICK_COLS
BRICK_W: Final = 4
BRICK_H: Final = 2
BRICK_PITCH_X: Final = 5  # 1-pixel gap between bricks
BRICK_PITCH_Y: Final = 3
BRICK_X0: Final = PLAY_X0
BRICK_Y0: Final = 16
#: Row colours from top to bottom.
BRICK_ROW_COLORS: Final = (Color.BRICK_RED, Color.BRICK_ORANGE, Color.BRICK_YELLOW, Color.BRICK_GREEN)
#: Hitting a brick in a row with index below this switches the ball to fast mode.
FAST_ROWS: Final = 2

# Paddle and ball ----------------------------------------------------------------------
PADDLE_W: Final = 8
PADDLE_H: Final = 2
PADDLE_Y: Final = 58
PADDLE_SPEED: Final = 3
PADDLE_MIN_X: Final = PLAY_X0
PADDLE_MAX_X: Final = PLAY_X1 - PADDLE_W + 1  # 54
PADDLE_START_X: Final = (PLAY_X0 + PLAY_X1 + 1 - PADDLE_W) // 2  # 28, centred
BALL_SIZE: Final = 2
SERVE_OFFSET_X: Final = (PADDLE_W - BALL_SIZE) // 2  # ball sits centred on the paddle
SERVE_Y: Final = PADDLE_Y - BALL_SIZE  # 56
SLOW_SPEED: Final = 1
FAST_SPEED: Final = 2
MAX_DX: Final = 2

# HUD ----------------------------------------------------------------------------------
START_LIVES: Final = 3
MAX_SCORE: Final = 999
SCORE_DIGITS: Final = 3
DIGIT_W: Final = 3
DIGIT_H: Final = 5
DIGIT_X0: Final = 2
DIGIT_PITCH: Final = 4
DIGIT_Y: Final = 1
PIP_W: Final = 4
PIP_H: Final = 2
PIP_Y: Final = 3
PIP_RIGHT: Final = PLAY_X1  # right edge of the first pip
PIP_PITCH: Final = 5
