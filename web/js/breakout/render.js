// Port of src/dreamcade/game/render.py: a state -> 64x64 palette-index frame.
import {
  BALL_SIZE, BRICK_COLS, BRICK_H, BRICK_ROW_COLORS, BRICK_W, Color, HEIGHT, Mode, NUM_BRICKS,
  PADDLE_H, PADDLE_W, PADDLE_Y, TOP_WALL_Y, WALL_THICKNESS, WIDTH, brickRect,
} from "./engine.js";

// Five 3-bit rows per digit, MSB = leftmost pixel (mirrors font.glyph_rows_as_bits()).
const GLYPHS = [
  [7, 5, 5, 5, 7], [2, 6, 2, 2, 7], [7, 1, 7, 4, 7], [7, 1, 7, 1, 7], [5, 5, 7, 1, 1],
  [7, 4, 7, 1, 7], [7, 4, 7, 5, 7], [7, 1, 2, 2, 2], [7, 5, 7, 5, 7], [7, 5, 7, 1, 7],
];
const DIGIT_X0 = 2;
const DIGIT_PITCH = 4;
const DIGIT_Y = 1;
const PIP_W = 4;
const PIP_H = 2;
const PIP_Y = 3;
const PIP_RIGHT = WIDTH - WALL_THICKNESS - 1;
const PIP_PITCH = 5;

function fillRect(frame, x0, y0, w, h, color) {
  for (let y = Math.max(0, y0); y < Math.min(HEIGHT, y0 + h); y++) {
    for (let x = Math.max(0, x0); x < Math.min(WIDTH, x0 + w); x++) frame[y * WIDTH + x] = color;
  }
}

/** Draw a state into a new Uint8Array(64*64) of palette indices. */
export function render(state) {
  const frame = new Uint8Array(WIDTH * HEIGHT);
  fillRect(frame, 0, TOP_WALL_Y, WIDTH, WALL_THICKNESS, Color.WALL);
  fillRect(frame, 0, TOP_WALL_Y, WALL_THICKNESS, HEIGHT - TOP_WALL_Y, Color.WALL);
  fillRect(frame, WIDTH - WALL_THICKNESS, TOP_WALL_Y, WALL_THICKNESS, HEIGHT - TOP_WALL_Y, Color.WALL);
  for (let i = 0; i < NUM_BRICKS; i++) {
    if (!state.bricks[i]) continue;
    const [x, y] = brickRect(i);
    fillRect(frame, x, y, BRICK_W, BRICK_H, BRICK_ROW_COLORS[Math.floor(i / BRICK_COLS)]);
  }
  const digits = String(state.score).padStart(3, "0");
  for (let i = 0; i < 3; i++) {
    const glyph = GLYPHS[Number(digits[i])];
    const x0 = DIGIT_X0 + i * DIGIT_PITCH;
    for (let r = 0; r < 5; r++) {
      for (let c = 0; c < 3; c++) {
        if (glyph[r] & (4 >> c)) frame[(DIGIT_Y + r) * WIDTH + x0 + c] = Color.WALL;
      }
    }
  }
  for (let i = 0; i < state.lives; i++) {
    fillRect(frame, PIP_RIGHT - PIP_W + 1 - i * PIP_PITCH, PIP_Y, PIP_W, PIP_H, Color.PADDLE);
  }
  fillRect(frame, state.paddleX, PADDLE_Y, PADDLE_W, PADDLE_H, Color.PADDLE);
  if (state.mode !== Mode.GAME_OVER) fillRect(frame, state.ballX, state.ballY, BALL_SIZE, BALL_SIZE, Color.BALL);
  return frame;
}
