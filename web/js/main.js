// dreamcade player: the real game and the neural dream, fed the same inputs in lockstep.
import * as breakout from "./breakout/engine.js";
import { render as renderBreakout } from "./breakout/render.js";
import * as dungeon from "./dungeon/engine.js";
import { render as renderDungeon } from "./dungeon/render.js";
import { loadWorld } from "./dungeon/world.js";
import { BreakoutDream } from "./dream/breakout_dream.js";
import { DungeonDream } from "./dream/dungeon_dream.js";
import { createSession, fetchModel } from "./dream/runtime.js";
import { asRgb, drawDiff, drawRgb } from "./display.js";
import { Autopilot, keyboardAction, listen } from "./input.js";

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const TICK_MS = { dungeon: 1000 / 15, breakout: 1000 / 30 };
const IDLE_TO_ATTRACT_MS = 20000;

// Exposed for automation (capture scripts read the frame counter and backend).
// `?step=manual` (frame-step mode, used by the recorder) advances one tick per `dreamcade.next()`.
const manualStep = params.get("step") === "manual";
let releaseTick = null;
const status = {
  game: null, frame: 0, backend: null, ready: false, error: null, fps: 0, modelMs: 0,
  next() {
    if (releaseTick) releaseTick();
  },
};
globalThis.dreamcade = status;

const app = {
  game: params.get("game") === "breakout" ? "breakout" : "dungeon",
  view: ["dream", "real", "split", "diff"].includes(params.get("view")) ? params.get("view") : "split",
  attract: params.get("attract") === "1",
  manifest: null,
  world: null,
  sessions: {},
  real: null,
  dream: null,
  autopilot: null,
  lastInput: performance.now(),
  generation: 0,
  frameTimes: [],
};

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

async function session(game) {
  if (!app.sessions[game]) {
    const entry = app.manifest[game];
    if (!entry) throw new Error(`no ${game} model in web/models/manifest.json; run dreamcade export ${game}`);
    setStatus(`downloading the ${game} dream (${(entry.bytes / 1e6).toFixed(1)} MB)...`);
    const bytes = await fetchModel("models/", entry, (f) => setStatus(`downloading the ${game} dream... ${Math.round(f * 100)}%`));
    setStatus("compiling the network...");
    app.sessions[game] = await createSession(bytes, params.get("backend"));
  }
  return app.sessions[game];
}

function realFrame() {
  return app.game === "dungeon" ? renderDungeon(app.world, app.real) : asRgb(renderBreakout(app.real), true);
}

/** Start (or restart) a game: fresh real state, dream memory filled with real frames. */
async function start() {
  const generation = ++app.generation;
  status.ready = false;
  status.frame = 0;
  status.game = app.game;
  document.body.dataset.game = app.game;
  const { session: s, backend } = await session(app.game);
  if (generation !== app.generation) return;
  status.backend = backend;
  $("backend").textContent = backend === "webgpu" ? "WebGPU" : "WASM";
  const meta = app.manifest[app.game];
  app.autopilot = new Autopilot(app.game);
  if (app.game === "dungeon") {
    app.real = dungeon.startState(app.world);
    app.dream = new DungeonDream(s, meta);
    app.dream.steps = Number($("steps").value);
    const frame = renderDungeon(app.world, app.real);
    app.dream.reset(Array(meta.context).fill(frame), Array(meta.context).fill(dungeon.Action.NOOP));
  } else {
    app.real = breakout.newGame();
    app.dream = new BreakoutDream(s, meta);
    app.dream.reset(Array(meta.context).fill(renderBreakout(app.real)));
  }
  const rgb = realFrame();
  draw(rgb, rgb);
  setStatus(app.attract ? "attract mode: press any game key to take over" : "your move");
  status.ready = true;
  loop(generation);
}

function draw(realRgb, dreamRgb) {
  drawRgb($("real"), realRgb);
  drawRgb($("dream"), dreamRgb);
  const psnr = drawDiff($("diff"), realRgb, dreamRgb);
  $("psnr").textContent = Number.isFinite(psnr) ? `${psnr.toFixed(1)} dB` : "identical";
}

async function loop(generation) {
  while (generation === app.generation) {
    if (manualStep) await new Promise((resolve) => { releaseTick = resolve; });
    const t0 = performance.now();
    if (!app.attract && t0 - app.lastInput > IDLE_TO_ATTRACT_MS && params.get("attract") !== "0") {
      app.attract = true;
      setStatus("attract mode: press any game key to take over");
    }
    const action = app.attract ? app.autopilot.next(app.real) : keyboardAction(app.game);
    if (app.game === "dungeon") app.real = dungeon.step(app.world, app.real, action).state;
    else app.real = breakout.step(app.real, action).state;
    const dreamed = await app.dream.step(action);
    if (generation !== app.generation) return;
    draw(realFrame(), app.game === "dungeon" ? dreamed : asRgb(dreamed, true));
    status.frame += 1;
    status.modelMs = app.dream.lastMs;
    const now = performance.now();
    app.frameTimes.push(now);
    while (app.frameTimes.length > 30) app.frameTimes.shift();
    if (app.frameTimes.length > 1) {
      status.fps = (1000 * (app.frameTimes.length - 1)) / (now - app.frameTimes[0]);
      $("fps").textContent = status.fps.toFixed(1);
    }
    $("ms").textContent = `${app.dream.lastMs.toFixed(0)} ms`;
    $("frame").textContent = String(status.frame);
    const wait = TICK_MS[app.game] - (performance.now() - t0);
    await new Promise((r) => setTimeout(r, Math.max(0, wait)));
  }
}

function setView(view) {
  app.view = view;
  document.body.dataset.view = view;
  for (const b of document.querySelectorAll("[data-set-view]")) b.classList.toggle("active", b.dataset.setView === view);
}

function setGame(game) {
  app.game = game;
  for (const b of document.querySelectorAll("[data-set-game]")) b.classList.toggle("active", b.dataset.setGame === game);
  start().catch(fail);
}

function fail(err) {
  status.error = String(err);
  setStatus(String(err.message || err), true);
  console.error(err);
}

async function main() {
  setView(app.view);
  for (const b of document.querySelectorAll("[data-set-view]")) b.addEventListener("click", () => setView(b.dataset.setView));
  for (const b of document.querySelectorAll("[data-set-game]")) b.addEventListener("click", () => setGame(b.dataset.setGame));
  $("steps").value = params.get("steps") || "3";
  $("steps-value").textContent = $("steps").value;
  $("steps").addEventListener("input", () => {
    $("steps-value").textContent = $("steps").value;
    if (app.dream instanceof DungeonDream) app.dream.steps = Number($("steps").value);
  });
  $("corrupt").addEventListener("click", () => {
    app.dream?.corrupt();
    setStatus("memory corrupted: watch the dream try to recover");
  });
  $("resync").addEventListener("click", () => start().catch(fail));
  $("attract").addEventListener("click", () => {
    app.attract = !app.attract;
    app.lastInput = performance.now();
  });
  listen(window, () => {
    app.lastInput = performance.now();
    if (app.attract) {
      app.attract = false;
      setStatus("your move");
    }
  });
  const res = await fetch("models/manifest.json");
  if (!res.ok) throw new Error("web/models/manifest.json missing: export the models first");
  app.manifest = await res.json();
  app.world = await loadWorld("assets/dungeon_world.json");
  setGame(app.game);
}

main().catch(fail);
