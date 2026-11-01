"""Rasterise a :class:`GameState` into a 64x64 frame of palette indices."""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from dreamcade.game.constants import (
    BALL_SIZE,
    BRICK_COLS,
    BRICK_H,
    BRICK_ROW_COLORS,
    BRICK_W,
    DIGIT_H,
    DIGIT_PITCH,
    DIGIT_W,
    DIGIT_X0,
    DIGIT_Y,
    HEIGHT,
    NUM_BRICKS,
    PADDLE_H,
    PADDLE_W,
    PADDLE_Y,
    PIP_H,
    PIP_PITCH,
    PIP_RIGHT,
    PIP_W,
    PIP_Y,
    SCORE_DIGITS,
    TOP_WALL_Y,
    WALL_THICKNESS,
    WIDTH,
    Color,
    Mode,
)
from dreamcade.game.engine import GameState, brick_rect
from dreamcade.game.font import GLYPHS

Frame = npt.NDArray[np.uint8]


def _static_layer() -> Frame:
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    frame[TOP_WALL_Y : TOP_WALL_Y + WALL_THICKNESS, :] = Color.WALL
    frame[TOP_WALL_Y:, :WALL_THICKNESS] = Color.WALL
    frame[TOP_WALL_Y:, WIDTH - WALL_THICKNESS :] = Color.WALL
    return frame


def _brick_layers() -> tuple[npt.NDArray[np.int16], Frame]:
    """Per-pixel brick index (or -1) and the colour each brick pixel takes when present."""
    ids = np.full((HEIGHT, WIDTH), -1, dtype=np.int16)
    colors = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for index in range(NUM_BRICKS):
        x, y = brick_rect(index)
        ids[y : y + BRICK_H, x : x + BRICK_W] = index
        colors[y : y + BRICK_H, x : x + BRICK_W] = BRICK_ROW_COLORS[index // BRICK_COLS]
    return ids, colors


STATIC_LAYER: Final = _static_layer()
BRICK_IDS, BRICK_COLORS = _brick_layers()
_BRICK_MASK: Final = BRICK_IDS >= 0
_BRICK_IDS_CLIPPED: Final = np.where(_BRICK_MASK, BRICK_IDS, 0)


def pip_left(index: int) -> int:
    """Leftmost column of the ``index``-th lives pip (pips grow right-to-left)."""
    return PIP_RIGHT - PIP_W + 1 - index * PIP_PITCH


def digit_left(index: int) -> int:
    """Leftmost column of the ``index``-th score digit (most significant first)."""
    return DIGIT_X0 + index * DIGIT_PITCH


def render(state: GameState) -> Frame:
    """Draw ``state``. Layer order: walls, bricks, HUD, paddle, ball (clipped at the bottom)."""
    frame = STATIC_LAYER.copy()

    alive = np.frombuffer(state.bricks, dtype=np.uint8).astype(np.bool_)
    visible = _BRICK_MASK & alive[_BRICK_IDS_CLIPPED]
    frame[visible] = BRICK_COLORS[visible]

    for index, char in enumerate(f"{state.score:0{SCORE_DIGITS}d}"):
        x = digit_left(index)
        region = frame[DIGIT_Y : DIGIT_Y + DIGIT_H, x : x + DIGIT_W]
        region[GLYPHS[int(char)]] = Color.WALL

    for index in range(state.lives):
        x = pip_left(index)
        frame[PIP_Y : PIP_Y + PIP_H, x : x + PIP_W] = Color.PADDLE

    frame[PADDLE_Y : PADDLE_Y + PADDLE_H, state.paddle_x : state.paddle_x + PADDLE_W] = Color.PADDLE

    if state.mode != Mode.GAME_OVER:
        y0, x0 = state.ball_y, state.ball_x
        frame[y0 : min(y0 + BALL_SIZE, HEIGHT), x0 : x0 + BALL_SIZE] = Color.BALL
    return frame
