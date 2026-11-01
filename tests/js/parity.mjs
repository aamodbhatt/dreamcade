// Replays the Python golden trajectories through the JavaScript ports.
// Usage: node tests/js/parity.mjs   (exit code 0 = bit/pixel identical)
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as breakout from "../../web/js/breakout/engine.js";
import { render as renderBreakout } from "../../web/js/breakout/render.js";
import * as dungeon from "../../web/js/dungeon/engine.js";
import { render as renderDungeon } from "../../web/js/dungeon/render.js";
import { parseWorld } from "../../web/js/dungeon/world.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const load = (p) => JSON.parse(readFileSync(join(root, p), "utf8"));

function fnv1a(bytes) {
  let h = 2166136261;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

const toPy = (s) => ({
  paddleX: s.paddleX, ballX: s.ballX, ballY: s.ballY, vx: s.vx, vy: s.vy,
  mode: s.mode, bricks: Array.from(s.bricks), score: s.score, lives: s.lives,
});

let failures = 0;
function check(name, golden, replay) {
  const { states, hashes } = replay(golden);
  for (let t = 0; t < golden.hashes.length; t++) {
    if (JSON.stringify(states[t]) !== JSON.stringify(golden.states[t]) || hashes[t] !== golden.hashes[t]) {
      console.error(`${name}: diverged at step ${t}\n  js:     ${JSON.stringify(states[t])} ${hashes[t]}\n  python: ${JSON.stringify(golden.states[t])} ${golden.hashes[t]}`);
      failures += 1;
      return;
    }
  }
  console.log(`${name}: ${golden.hashes.length} frames identical`);
}

check("breakout", load("tests/golden/breakout.json"), (g) => {
  let state = breakout.newGame();
  const states = [toPy(state)];
  const hashes = [fnv1a(renderBreakout(state))];
  for (const a of g.actions) {
    state = breakout.step(state, a).state;
    states.push(toPy(state));
    hashes.push(fnv1a(renderBreakout(state)));
  }
  return { states, hashes };
});

const world = parseWorld(load("web/assets/dungeon_world.json"));
check("dungeon", load("tests/golden/dungeon.json"), (g) => {
  const [x, y, a] = g.states[0];
  let state = { x, y, a };
  const states = [[x, y, a]];
  const hashes = [fnv1a(renderDungeon(world, state))];
  for (const act of g.actions) {
    state = dungeon.step(world, state, act).state;
    states.push([state.x, state.y, state.a]);
    hashes.push(fnv1a(renderDungeon(world, state)));
  }
  return { states, hashes };
});

process.exit(failures === 0 ? 0 : 1);
