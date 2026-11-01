// Keyboard state -> per-game actions, plus the arcade-style "attract mode" autopilot.
import { Action as BreakoutAction, Mode, PADDLE_W } from "./breakout/engine.js";
import { Action as DungeonAction } from "./dungeon/engine.js";

const held = new Set();
const pressed = new Set();
const GAME_KEYS = new Set([
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE", "Space",
]);

export function listen(target, onActivity) {
  target.addEventListener("keydown", (e) => {
    if (!GAME_KEYS.has(e.code)) return;
    e.preventDefault();
    held.add(e.code);
    pressed.add(e.code);
    onActivity();
  });
  target.addEventListener("keyup", (e) => held.delete(e.code));
  target.addEventListener("blur", () => held.clear());
}

const any = (...codes) => codes.some((c) => held.has(c) || pressed.has(c));

/** The action for this tick from the keyboard (taps between ticks still count once). */
export function keyboardAction(game) {
  let action;
  if (game === "breakout") {
    if (any("Space", "ArrowUp", "KeyW")) action = BreakoutAction.FIRE;
    else if (any("ArrowLeft", "KeyA")) action = BreakoutAction.LEFT;
    else if (any("ArrowRight", "KeyD")) action = BreakoutAction.RIGHT;
    else action = BreakoutAction.NOOP;
  } else if (any("ArrowUp", "KeyW")) action = DungeonAction.FORWARD;
  else if (any("ArrowDown", "KeyS")) action = DungeonAction.BACK;
  else if (any("KeyQ")) action = DungeonAction.STRAFE_LEFT;
  else if (any("KeyE")) action = DungeonAction.STRAFE_RIGHT;
  else if (any("ArrowLeft", "KeyA")) action = DungeonAction.TURN_LEFT;
  else if (any("ArrowRight", "KeyD")) action = DungeonAction.TURN_RIGHT;
  else action = DungeonAction.NOOP;
  pressed.clear();
  return action;
}

const F = DungeonAction.FORWARD;
const L = DungeonAction.TURN_LEFT;
const R = DungeonAction.TURN_RIGHT;
const SL = DungeonAction.STRAFE_LEFT;
// A tour from the hub: east down the corridor, north into the blue vault, a look at its orb,
// a full spin (the hard case: everything behind you must be remembered), then onwards.
const DUNGEON_TOUR = [
  [F, 54], [L, 16], [F, 46], [R, 16], [L, 32], [F, 20], [R, 64], [SL, 8], [R, 16], [F, 12], [L, 32],
];

/** Scripted inputs for the attract mode. Breakout plays like the dataset's tracker. */
export class Autopilot {
  constructor(game) {
    this.game = game;
    this.i = 0;
    this.left = DUNGEON_TOUR[0][1];
  }

  next(realState) {
    if (this.game === "breakout") {
      if (realState.mode !== Mode.PLAY) return BreakoutAction.FIRE;
      const error = realState.ballX - (realState.paddleX + PADDLE_W / 2 - 1);
      if (Math.abs(error) <= 1) return BreakoutAction.NOOP;
      return error > 0 ? BreakoutAction.RIGHT : BreakoutAction.LEFT;
    }
    if (this.left === 0) {
      this.i = (this.i + 1) % DUNGEON_TOUR.length;
      this.left = DUNGEON_TOUR[this.i][1];
    }
    this.left -= 1;
    return DUNGEON_TOUR[this.i][0];
  }
}
