// Port of src/dreamcade/dungeon/render.py. Every floating-point expression mirrors the
// Python one operation for operation (same order, explicit Math.floor), which is what makes
// the frames pixel-identical; tests/js/parity.mjs checks golden frame hashes.

export const WIDTH = 64;
export const HEIGHT = 64;
const PLANE = 0.66;
const FOCAL = WIDTH / (2 * PLANE);
const FOG = 0.02;
const SIDE_SHADE = 0.8;
const CHECKER_DARK = 0.75;
const ORB_HEIGHT = 0.45;
const ORB_CORE = 0.6;
const NO_DIR = 1e30;

/** Render one state into a Uint8ClampedArray of 64*64*3 RGB bytes. */
export function render(world, state, out = new Uint8ClampedArray(WIDTH * HEIGHT * 3)) {
  const { size, tex: TEX, walls, rooms, textures, floorTint, ceiling } = world;
  const posX = state.x / world.fixed;
  const posY = state.y / world.fixed;
  const dirX = world.cos[state.a] / world.trigOne;
  const dirY = world.sin[state.a] / world.trigOne;
  const planeX = -dirY * PLANE;
  const planeY = dirX * PLANE;
  const zbuf = new Float64Array(WIDTH);

  for (let c = 0; c < WIDTH; c++) {
    const cam = (2.0 * c + 1.0) / WIDTH - 1.0;
    const rayX = dirX + planeX * cam;
    const rayY = dirY + planeY * cam;
    let mapX = Math.floor(posX);
    let mapY = Math.floor(posY);
    const deltaX = rayX === 0 ? NO_DIR : Math.abs(1 / rayX);
    const deltaY = rayY === 0 ? NO_DIR : Math.abs(1 / rayY);
    const stepX = rayX < 0 ? -1 : 1;
    const stepY = rayY < 0 ? -1 : 1;
    let sideX = rayX < 0 ? (posX - mapX) * deltaX : (mapX + 1.0 - posX) * deltaX;
    let sideY = rayY < 0 ? (posY - mapY) * deltaY : (mapY + 1.0 - posY) * deltaY;
    let side = 0;
    for (let i = 0; i < 4 * size; i++) {
      if (sideX < sideY) {
        sideX += deltaX;
        mapX += stepX;
        side = 0;
      } else {
        sideY += deltaY;
        mapY += stepY;
        side = 1;
      }
      if (walls[mapY * size + mapX] !== 0) break;
    }
    const perp = side === 0 ? sideX - deltaX : sideY - deltaY;
    zbuf[c] = perp;

    const texId = walls[mapY * size + mapX] - 1;
    let wallX = side === 0 ? posY + perp * rayY : posX + perp * rayX;
    wallX = wallX - Math.floor(wallX);
    let texX = Math.floor(wallX * TEX);
    if ((side === 0 && rayX > 0) || (side === 1 && rayY < 0)) texX = TEX - 1 - texX;
    const lineH = FOCAL / perp;
    const top = HEIGHT / 2 - lineH / 2;
    const shade = (side === 1 ? SIDE_SHADE : 1.0) / (1 + perp * perp * FOG);

    for (let r = 0; r < HEIGHT; r++) {
      const row = r + 0.5;
      const o = (r * WIDTH + c) * 3;
      const v = (row - top) / lineH;
      if (v >= 0 && v < 1) {
        const texY = Math.min(Math.floor(v * TEX), TEX - 1);
        const t = ((texId * TEX + texY) * TEX + texX) * 3;
        out[o] = Math.floor(textures[t] * shade);
        out[o + 1] = Math.floor(textures[t + 1] * shade);
        out[o + 2] = Math.floor(textures[t + 2] * shade);
        continue;
      }
      const horizon = row - HEIGHT / 2;
      const rowDist = (FOCAL * 0.5) / Math.abs(horizon);
      const fog = 1 / (1 + rowDist * rowDist * FOG);
      if (horizon > 0) {
        const fx = posX + rowDist * rayX;
        const fy = posY + rowDist * rayY;
        const cellX = Math.min(Math.max(Math.floor(fx), 0), size - 1);
        const cellY = Math.min(Math.max(Math.floor(fy), 0), size - 1);
        const checker = (Math.floor(fx * 2) + Math.floor(fy * 2)) & 1;
        const tint = floorTint[rooms[cellY * size + cellX]];
        const k = checker === 1 ? CHECKER_DARK : 1.0;
        out[o] = Math.floor(tint[0] * k * fog);
        out[o + 1] = Math.floor(tint[1] * k * fog);
        out[o + 2] = Math.floor(tint[2] * k * fog);
      } else {
        out[o] = Math.floor(ceiling[0] * fog);
        out[o + 1] = Math.floor(ceiling[1] * fog);
        out[o + 2] = Math.floor(ceiling[2] * fog);
      }
    }
  }

  // Orbs, depth-tested per pixel against walls and previously drawn orbs.
  const depthBuf = new Float64Array(WIDTH * HEIGHT);
  for (let r = 0; r < HEIGHT; r++) for (let c = 0; c < WIDTH; c++) depthBuf[r * WIDTH + c] = zbuf[c];
  for (const [ox, oy, red, green, blue] of world.orbs) {
    const sx = ox / world.fixed - posX;
    const sy = oy / world.fixed - posY;
    const depth = dirX * sx + dirY * sy;
    if (!(depth > 0.05)) continue;
    const lateral = dirX * sy - dirY * sx;
    const cx = (WIDTH / 2) * (1 + lateral / (PLANE * depth));
    const cy = HEIGHT / 2 + ((0.5 - ORB_HEIGHT) * FOCAL) / depth;
    const radius = (world.orbRadius * FOCAL) / depth;
    const r2 = radius * radius;
    const fog = 1 + depth * depth * FOG;
    for (let r = 0; r < HEIGHT; r++) {
      const ddy = r + 0.5 - cy;
      for (let c = 0; c < WIDTH; c++) {
        const ddx = c + 0.5 - cx;
        const d2 = ddx * ddx + ddy * ddy;
        const i = r * WIDTH + c;
        if (!(d2 < r2) || !(depth < depthBuf[i])) continue;
        const k = (1 - (ORB_CORE * d2) / r2) / fog;
        out[i * 3] = Math.floor(red * k);
        out[i * 3 + 1] = Math.floor(green * k);
        out[i * 3 + 2] = Math.floor(blue * k);
        depthBuf[i] = depth;
      }
    }
  }
  return out;
}
