"""Deterministic, frame-observable Breakout rules.

Design invariant: the full game state is recoverable from the last two rendered frames.

* Positions, bricks, score, lives and the paddle are drawn on screen.
* Velocity is never hidden: collisions are resolved by *look-ahead reflection* (the velocity
  is reflected before the ball moves), so the displacement between two consecutive frames
  is exactly the velocity the next step starts from.
* The phase is visible too: in ``SERVE`` the ball rests on the paddle with zero vertical
  displacement (impossible in flight, where ``vy`` is never 0) and ``GAME_OVER`` has no
  ball at all. ``FIRE`` is mutually exclusive with moving, so a launch frame is never
  blurred by paddle motion.

``dreamcade.game.observe.infer_state`` implements the inverse mapping and the test suite
checks ``infer_state(render(s_prev), render(s)) == s`` on random reachable states.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntFlag

from dreamcade.game.constants import (
    BALL_SIZE,
    BRICK_COLS,
    BRICK_H,
    BRICK_PITCH_X,
    BRICK_PITCH_Y,
    BRICK_ROWS,
    BRICK_W,
    BRICK_X0,
    BRICK_Y0,
    FAST_ROWS,
    FAST_SPEED,
    HEIGHT,
    MAX_SCORE,
    NUM_BRICKS,
    PADDLE_H,
    PADDLE_MAX_X,
    PADDLE_MIN_X,
    PADDLE_SPEED,
    PADDLE_START_X,
    PADDLE_W,
    PADDLE_Y,
    PLAY_X0,
    PLAY_X1,
    PLAY_Y0,
    SERVE_OFFSET_X,
    SERVE_Y,
    SLOW_SPEED,
    START_LIVES,
    Action,
    Mode,
)

FULL_WALL = bytes([1]) * NUM_BRICKS


class Event(IntFlag):
    """What happened during one step. Used for dataset coverage statistics."""

    NONE = 0
    WALL_BOUNCE = 1
    PADDLE_HIT = 2
    BRICK_HIT = 4
    LIFE_LOST = 8
    GAME_OVER = 16
    RESET = 32
    LAUNCH = 64
    LEVEL_CLEAR = 128
    FAST_MODE = 256  # the ball switched from slow to fast during this step


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete game state.

    ``bricks`` holds one byte per brick (1 = present), row-major from the top row.
    In ``SERVE`` the velocity is ``(0, 0)`` and the ball sits on the paddle; in
    ``GAME_OVER`` the ball is parked at ``(0, 0)`` and not drawn. These canonical values
    make states comparable with ``==``.
    """

    paddle_x: int
    ball_x: int
    ball_y: int
    vx: int
    vy: int
    mode: Mode
    bricks: bytes
    score: int
    lives: int

    def __post_init__(self) -> None:
        if len(self.bricks) != NUM_BRICKS:
            msg = f"bricks must have {NUM_BRICKS} entries, got {len(self.bricks)}"
            raise ValueError(msg)
        if not PADDLE_MIN_X <= self.paddle_x <= PADDLE_MAX_X:
            msg = f"paddle_x {self.paddle_x} outside [{PADDLE_MIN_X}, {PADDLE_MAX_X}]"
            raise ValueError(msg)
        if not 0 <= self.score <= MAX_SCORE:
            msg = f"score {self.score} outside [0, {MAX_SCORE}]"
            raise ValueError(msg)
        if not 0 <= self.lives <= START_LIVES:
            msg = f"lives {self.lives} outside [0, {START_LIVES}]"
            raise ValueError(msg)
        if (self.mode == Mode.GAME_OVER) != (self.lives == 0):
            msg = "GAME_OVER must coincide with zero lives"
            raise ValueError(msg)

    @property
    def bricks_left(self) -> int:
        """Number of bricks still standing."""
        return sum(self.bricks)

    def to_dict(self) -> dict[str, int | str | list[int]]:
        """JSON-friendly representation shared with the JavaScript port."""
        return {
            "paddleX": self.paddle_x,
            "ballX": self.ball_x,
            "ballY": self.ball_y,
            "vx": self.vx,
            "vy": self.vy,
            "mode": int(self.mode),
            "bricks": list(self.bricks),
            "score": self.score,
            "lives": self.lives,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GameState:
        """Inverse of :meth:`to_dict`, validating every field."""

        def as_int(key: str) -> int:
            value = data[key]
            if not isinstance(value, int) or isinstance(value, bool):
                msg = f"field {key!r} must be an int, got {value!r}"
                raise TypeError(msg)
            return value

        bricks = data["bricks"]
        if not isinstance(bricks, list) or not all(b in (0, 1) for b in bricks):
            msg = "field 'bricks' must be a list of 0/1 values"
            raise TypeError(msg)
        return cls(
            paddle_x=as_int("paddleX"),
            ball_x=as_int("ballX"),
            ball_y=as_int("ballY"),
            vx=as_int("vx"),
            vy=as_int("vy"),
            mode=Mode(as_int("mode")),
            bricks=bytes(bricks),
            score=as_int("score"),
            lives=as_int("lives"),
        )


def brick_rect(index: int) -> tuple[int, int]:
    """Top-left pixel of brick ``index`` (row-major from the top row)."""
    row, col = divmod(index, BRICK_COLS)
    return BRICK_X0 + col * BRICK_PITCH_X, BRICK_Y0 + row * BRICK_PITCH_Y


def serving_state(paddle_x: int, bricks: bytes, score: int, lives: int) -> GameState:
    """A state with the ball resting on the paddle, waiting for FIRE."""
    return GameState(
        paddle_x=paddle_x,
        ball_x=paddle_x + SERVE_OFFSET_X,
        ball_y=SERVE_Y,
        vx=0,
        vy=0,
        mode=Mode.SERVE,
        bricks=bricks,
        score=score,
        lives=lives,
    )


def game_over_state(paddle_x: int, bricks: bytes, score: int) -> GameState:
    """The canonical state shown after the last life is lost."""
    return GameState(
        paddle_x=paddle_x,
        ball_x=0,
        ball_y=0,
        vx=0,
        vy=0,
        mode=Mode.GAME_OVER,
        bricks=bricks,
        score=score,
        lives=0,
    )


def new_game(paddle_x: int = PADDLE_START_X) -> GameState:
    """A fresh game: full wall, score 0, three lives, ball on the paddle."""
    return serving_state(paddle_x, FULL_WALL, 0, START_LIVES)


def _overlaps(ax: int, ay: int, aw: int, ah: int, bx: int, by: int, bw: int, bh: int) -> bool:
    return ax <= bx + bw - 1 and bx <= ax + aw - 1 and ay <= by + bh - 1 and by <= ay + ah - 1


def _hits_side_wall(x: int) -> bool:
    return x < PLAY_X0 or x + BALL_SIZE - 1 > PLAY_X1


def _brick_at(bricks: bytes, x: int, y: int) -> int:
    """Index of the standing brick overlapped by a ball at ``(x, y)``, or -1.

    Bricks are separated by one-pixel gaps and the ball is 2x2, so at most one brick can
    overlap the ball; this is what keeps the score a +1 counter.
    """
    row_lo = max(0, (y - BRICK_Y0) // BRICK_PITCH_Y)
    row_hi = min(BRICK_ROWS - 1, (y + BALL_SIZE - 1 - BRICK_Y0) // BRICK_PITCH_Y)
    col_lo = max(0, (x - BRICK_X0) // BRICK_PITCH_X)
    col_hi = min(BRICK_COLS - 1, (x + BALL_SIZE - 1 - BRICK_X0) // BRICK_PITCH_X)
    for row in range(row_lo, row_hi + 1):
        for col in range(col_lo, col_hi + 1):
            index = row * BRICK_COLS + col
            if not bricks[index]:
                continue
            bx, by = brick_rect(index)
            if _overlaps(x, y, BALL_SIZE, BALL_SIZE, bx, by, BRICK_W, BRICK_H):
                return index
    return -1


def _english(offset: int, incoming_vx: int) -> int:
    """Horizontal velocity after a paddle hit, from where the ball lands on the paddle.

    ``offset`` is the ball's left column relative to the paddle's left column, in
    ``[-1, PADDLE_W - 1]``; the centre (offset 3) keeps the incoming direction.
    """
    if offset <= 0:
        return -2
    if offset <= 2:
        return -1
    if offset == 3:
        return 1 if incoming_vx > 0 else -1
    if offset <= 5:
        return 1
    return 2


def _is_blocked(bricks: bytes, x: int, y: int) -> bool:
    return _hits_side_wall(x) or y < PLAY_Y0 or _brick_at(bricks, x, y) >= 0


def _advance_ball(state: GameState, paddle_x: int) -> tuple[GameState, Event]:
    x, y, vx, vy = state.ball_x, state.ball_y, state.vx, state.vy
    events = Event.NONE
    bricks, score = state.bricks, state.score

    if _hits_side_wall(x + vx):
        vx = -vx
        events |= Event.WALL_BOUNCE
    if y + vy < PLAY_Y0:
        vy = -vy
        events |= Event.WALL_BOUNCE

    ball_above_paddle = y + BALL_SIZE - 1 < PADDLE_Y
    if (
        vy > 0
        and ball_above_paddle
        and _overlaps(x + vx, y + vy, BALL_SIZE, BALL_SIZE, paddle_x, PADDLE_Y, PADDLE_W, PADDLE_H)
    ):
        vx = _english(x + vx - paddle_x, vx)
        vy = -vy
        events |= Event.PADDLE_HIT
        if _hits_side_wall(x + vx):
            vx = -vx

    hit = _brick_at(bricks, x + vx, y + vy)
    if hit >= 0:
        bricks = bricks[:hit] + b"\x00" + bricks[hit + 1 :]
        score = min(MAX_SCORE, score + 1)
        events |= Event.BRICK_HIT
        bx, by = brick_rect(hit)
        x_overlap_now = x <= bx + BRICK_W - 1 and bx <= x + BALL_SIZE - 1
        y_overlap_now = y <= by + BRICK_H - 1 and by <= y + BALL_SIZE - 1
        if x_overlap_now and not y_overlap_now:
            vy = -vy
        elif y_overlap_now and not x_overlap_now:
            vx = -vx
        else:
            vx, vy = -vx, -vy
        if hit // BRICK_COLS < FAST_ROWS and abs(vy) == SLOW_SPEED:
            vy = FAST_SPEED if vy > 0 else -FAST_SPEED
            events |= Event.FAST_MODE
        if _is_blocked(bricks, x + vx, y + vy):
            # Rare pocket geometry: retrace the incoming path, which is known to be free.
            vx, vy = -state.vx, -state.vy
            events &= ~Event.FAST_MODE

    x, y = x + vx, y + vy

    if y >= HEIGHT:
        lives = state.lives - 1
        events |= Event.LIFE_LOST
        if lives == 0:
            return game_over_state(paddle_x, bricks, score), events | Event.GAME_OVER
        return serving_state(paddle_x, bricks, score, lives), events

    if not any(bricks):
        return serving_state(paddle_x, FULL_WALL, score, state.lives), events | Event.LEVEL_CLEAR

    next_state = replace(
        state, paddle_x=paddle_x, ball_x=x, ball_y=y, vx=vx, vy=vy, bricks=bricks, score=score
    )
    return next_state, events


def step(state: GameState, action: Action | int) -> tuple[GameState, Event]:
    """Advance the game by one tick. Pure: ``state`` is not modified."""
    action = Action(action)
    paddle_x = state.paddle_x
    if action == Action.LEFT:
        paddle_x = max(PADDLE_MIN_X, paddle_x - PADDLE_SPEED)
    elif action == Action.RIGHT:
        paddle_x = min(PADDLE_MAX_X, paddle_x + PADDLE_SPEED)

    if state.mode == Mode.GAME_OVER:
        if action == Action.FIRE:
            return serving_state(paddle_x, FULL_WALL, 0, START_LIVES), Event.RESET
        return replace(state, paddle_x=paddle_x), Event.NONE

    if state.mode == Mode.SERVE:
        if action == Action.FIRE:
            # Serve towards the centre of the field: the direction is a visible function of
            # the paddle position, so a model can predict it from the current frame.
            vx = 1 if paddle_x < PADDLE_START_X else -1
            vy = -SLOW_SPEED
            launched = replace(
                state,
                paddle_x=paddle_x,
                ball_x=paddle_x + SERVE_OFFSET_X + vx,
                ball_y=SERVE_Y + vy,
                vx=vx,
                vy=vy,
                mode=Mode.PLAY,
            )
            return launched, Event.LAUNCH
        return serving_state(paddle_x, state.bricks, state.score, state.lives), Event.NONE

    return _advance_ball(state, paddle_x)
