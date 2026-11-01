// Port of src/dreamcade/game/{constants,engine}.py. Integer-only, so it matches Python
// bit-for-bit; tests/js/parity.mjs replays the golden trajectories to prove it.

export const WIDTH = 64;
export const HEIGHT = 64;
export const Color = Object.freeze({
  BACKGROUND: 0, WALL: 1, PADDLE: 2, BALL: 3, BRICK_RED: 4, BRICK_ORANGE: 5, BRICK_YELLOW: 6, BRICK_GREEN: 7,
});
export const PALETTE_RGB = [
  [10, 10, 24], [142, 146, 168], [64, 196, 255], [250, 250, 250],
  [239, 71, 111], [255, 159, 28], [255, 214, 10], [6, 214, 160],
];
export const Action = Object.freeze({ NOOP: 0, FIRE: 1, LEFT: 2, RIGHT: 3 });
export const Mode = Object.freeze({ SERVE: 0, PLAY: 1, GAME_OVER: 2 });
export const Event = Object.freeze({
  NONE: 0, WALL_BOUNCE: 1, PADDLE_HIT: 2, BRICK_HIT: 4, LIFE_LOST: 8, GAME_OVER: 16,
  RESET: 32, LAUNCH: 64, LEVEL_CLEAR: 128, FAST_MODE: 256,
});

export const HUD_HEIGHT = 8;
export const WALL_THICKNESS = 2;
export const TOP_WALL_Y = HUD_HEIGHT;
export const PLAY_X0 = WALL_THICKNESS;
export const PLAY_X1 = WIDTH - WALL_THICKNESS - 1;
export const PLAY_Y0 = TOP_WALL_Y + WALL_THICKNESS;
export const BRICK_ROWS = 4;
export const BRICK_COLS = 12;
export const NUM_BRICKS = BRICK_ROWS * BRICK_COLS;
export const BRICK_W = 4;
export const BRICK_H = 2;
export const BRICK_PITCH_X = 5;
export const BRICK_PITCH_Y = 3;
export const BRICK_X0 = PLAY_X0;
export const BRICK_Y0 = 16;
export const BRICK_ROW_COLORS = [Color.BRICK_RED, Color.BRICK_ORANGE, Color.BRICK_YELLOW, Color.BRICK_GREEN];
const FAST_ROWS = 2;
export const PADDLE_W = 8;
export const PADDLE_H = 2;
export const PADDLE_Y = 58;
const PADDLE_SPEED = 3;
const PADDLE_MIN_X = PLAY_X0;
const PADDLE_MAX_X = PLAY_X1 - PADDLE_W + 1;
export const PADDLE_START_X = Math.floor((PLAY_X0 + PLAY_X1 + 1 - PADDLE_W) / 2);
export const BALL_SIZE = 2;
const SERVE_OFFSET_X = Math.floor((PADDLE_W - BALL_SIZE) / 2);
const SERVE_Y = PADDLE_Y - BALL_SIZE;
const SLOW_SPEED = 1;
const FAST_SPEED = 2;
export const START_LIVES = 3;
const MAX_SCORE = 999;

const fullWall = () => new Uint8Array(NUM_BRICKS).fill(1);

export function brickRect(index) {
  const row = Math.floor(index / BRICK_COLS);
  const col = index % BRICK_COLS;
  return [BRICK_X0 + col * BRICK_PITCH_X, BRICK_Y0 + row * BRICK_PITCH_Y];
}

function servingState(paddleX, bricks, score, lives) {
  return { paddleX, ballX: paddleX + SERVE_OFFSET_X, ballY: SERVE_Y, vx: 0, vy: 0, mode: Mode.SERVE, bricks, score, lives };
}

function gameOverState(paddleX, bricks, score) {
  return { paddleX, ballX: 0, ballY: 0, vx: 0, vy: 0, mode: Mode.GAME_OVER, bricks, score, lives: 0 };
}

/** A fresh game: full wall, score 0, three lives, ball resting on the paddle. */
export function newGame(paddleX = PADDLE_START_X) {
  return servingState(paddleX, fullWall(), 0, START_LIVES);
}

const overlaps = (ax, ay, aw, ah, bx, by, bw, bh) =>
  ax <= bx + bw - 1 && bx <= ax + aw - 1 && ay <= by + bh - 1 && by <= ay + ah - 1;

const hitsSideWall = (x) => x < PLAY_X0 || x + BALL_SIZE - 1 > PLAY_X1;

function brickAt(bricks, x, y) {
  const rowLo = Math.max(0, Math.floor((y - BRICK_Y0) / BRICK_PITCH_Y));
  const rowHi = Math.min(BRICK_ROWS - 1, Math.floor((y + BALL_SIZE - 1 - BRICK_Y0) / BRICK_PITCH_Y));
  const colLo = Math.max(0, Math.floor((x - BRICK_X0) / BRICK_PITCH_X));
  const colHi = Math.min(BRICK_COLS - 1, Math.floor((x + BALL_SIZE - 1 - BRICK_X0) / BRICK_PITCH_X));
  for (let row = rowLo; row <= rowHi; row++) {
    for (let col = colLo; col <= colHi; col++) {
      const index = row * BRICK_COLS + col;
      if (!bricks[index]) continue;
      const [bx, by] = brickRect(index);
      if (overlaps(x, y, BALL_SIZE, BALL_SIZE, bx, by, BRICK_W, BRICK_H)) return index;
    }
  }
  return -1;
}

function english(offset, incomingVx) {
  if (offset <= 0) return -2;
  if (offset <= 2) return -1;
  if (offset === 3) return incomingVx > 0 ? 1 : -1;
  if (offset <= 5) return 1;
  return 2;
}

const isBlocked = (bricks, x, y) => hitsSideWall(x) || y < PLAY_Y0 || brickAt(bricks, x, y) >= 0;

function advanceBall(state, paddleX) {
  let { ballX: x, ballY: y, vx, vy } = state;
  let events = Event.NONE;
  let bricks = state.bricks;
  let score = state.score;

  if (hitsSideWall(x + vx)) { vx = -vx; events |= Event.WALL_BOUNCE; }
  if (y + vy < PLAY_Y0) { vy = -vy; events |= Event.WALL_BOUNCE; }

  const ballAbovePaddle = y + BALL_SIZE - 1 < PADDLE_Y;
  if (vy > 0 && ballAbovePaddle && overlaps(x + vx, y + vy, BALL_SIZE, BALL_SIZE, paddleX, PADDLE_Y, PADDLE_W, PADDLE_H)) {
    vx = english(x + vx - paddleX, vx);
    vy = -vy;
    events |= Event.PADDLE_HIT;
    if (hitsSideWall(x + vx)) vx = -vx;
  }

  const hit = brickAt(bricks, x + vx, y + vy);
  if (hit >= 0) {
    bricks = bricks.slice();
    bricks[hit] = 0;
    score = Math.min(MAX_SCORE, score + 1);
    events |= Event.BRICK_HIT;
    const [bx, by] = brickRect(hit);
    const xOverlapNow = x <= bx + BRICK_W - 1 && bx <= x + BALL_SIZE - 1;
    const yOverlapNow = y <= by + BRICK_H - 1 && by <= y + BALL_SIZE - 1;
    if (xOverlapNow && !yOverlapNow) vy = -vy;
    else if (yOverlapNow && !xOverlapNow) vx = -vx;
    else { vx = -vx; vy = -vy; }
    if (Math.floor(hit / BRICK_COLS) < FAST_ROWS && Math.abs(vy) === SLOW_SPEED) {
      vy = vy > 0 ? FAST_SPEED : -FAST_SPEED;
      events |= Event.FAST_MODE;
    }
    if (isBlocked(bricks, x + vx, y + vy)) {
      vx = -state.vx;
      vy = -state.vy;
      events &= ~Event.FAST_MODE;
    }
  }

  x += vx;
  y += vy;

  if (y >= HEIGHT) {
    const lives = state.lives - 1;
    events |= Event.LIFE_LOST;
    if (lives === 0) return { state: gameOverState(paddleX, bricks, score), events: events | Event.GAME_OVER };
    return { state: servingState(paddleX, bricks, score, lives), events };
  }
  if (!bricks.some((b) => b)) {
    return { state: servingState(paddleX, fullWall(), score, state.lives), events: events | Event.LEVEL_CLEAR };
  }
  return { state: { ...state, paddleX, ballX: x, ballY: y, vx, vy, bricks, score }, events };
}

/** Advance one tick. Pure: `state` is not modified. Returns `{state, events}`. */
export function step(state, action) {
  let paddleX = state.paddleX;
  if (action === Action.LEFT) paddleX = Math.max(PADDLE_MIN_X, paddleX - PADDLE_SPEED);
  else if (action === Action.RIGHT) paddleX = Math.min(PADDLE_MAX_X, paddleX + PADDLE_SPEED);

  if (state.mode === Mode.GAME_OVER) {
    if (action === Action.FIRE) return { state: servingState(paddleX, fullWall(), 0, START_LIVES), events: Event.RESET };
    return { state: { ...state, paddleX }, events: Event.NONE };
  }
  if (state.mode === Mode.SERVE) {
    if (action === Action.FIRE) {
      const vx = paddleX < PADDLE_START_X ? 1 : -1;
      const vy = -SLOW_SPEED;
      return {
        state: { ...state, paddleX, ballX: paddleX + SERVE_OFFSET_X + vx, ballY: SERVE_Y + vy, vx, vy, mode: Mode.PLAY },
        events: Event.LAUNCH,
      };
    }
    return { state: servingState(paddleX, state.bricks, state.score, state.lives), events: Event.NONE };
  }
  return advanceBall(state, paddleX);
}
