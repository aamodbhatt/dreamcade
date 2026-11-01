// Port of src/dreamcade/dungeon/engine.py (one agent). Integer arithmetic only.

export const Action = Object.freeze({
  NOOP: 0, FORWARD: 1, BACK: 2, TURN_LEFT: 3, TURN_RIGHT: 4, STRAFE_LEFT: 5, STRAFE_RIGHT: 6,
});
export const NUM_ACTIONS = 7;
const MOVE_SPEED = 32;
const STRAFE_SPEED = 24;
const RADIUS = 64;

function blocked(world, px, py) {
  for (const ox of [-RADIUS, RADIUS - 1]) {
    for (const oy of [-RADIUS, RADIUS - 1]) {
      const cx = (px + ox) >> 8;
      const cy = (py + oy) >> 8;
      if (world.walls[cy * world.size + cx] !== 0) return true;
    }
  }
  return false;
}

function hitsOrb(world, px, py) {
  const r = world.orbCollide;
  for (const [ox, oy] of world.orbs) {
    const dx = px - ox;
    const dy = py - oy;
    if (dx * dx + dy * dy < r * r) return true;
  }
  return false;
}

/** Advance one tick. Returns `{state: {x, y, a}, moved}`. */
export function step(world, state, action) {
  const mask = world.numAngles - 1;
  let a = state.a;
  if (action === Action.TURN_LEFT) a = (a - 1) & mask;
  if (action === Action.TURN_RIGHT) a = (a + 1) & mask;
  const cos = world.cos[state.a];
  const sin = world.sin[state.a];
  const fwd = (action === Action.FORWARD ? MOVE_SPEED : 0) - (action === Action.BACK ? MOVE_SPEED : 0);
  const side = (action === Action.STRAFE_RIGHT ? STRAFE_SPEED : 0) - (action === Action.STRAFE_LEFT ? STRAFE_SPEED : 0);
  const dx = (cos * fwd - sin * side) >> 14;
  const dy = (sin * fwd + cos * side) >> 14;

  let nx = state.x + dx;
  if (blocked(world, nx, state.y)) nx = state.x;
  let ny = state.y + dy;
  if (blocked(world, nx, ny)) ny = state.y;
  if (hitsOrb(world, nx, ny)) {
    nx = state.x;
    ny = state.y;
  }
  const translating = fwd !== 0 || side !== 0;
  const moved = !translating || nx !== state.x || ny !== state.y;
  return { state: { x: nx, y: ny, a }, moved };
}

/** Where a new game starts: the hub, looking east down the corridor. */
export function startState(world) {
  return { x: 9 * world.fixed + world.fixed / 2, y: 10 * world.fixed + world.fixed / 2, a: 0 };
}
