// The dungeon dream: EDM Euler sampling around the exported denoiser, in the browser.
import { gaussianSource, tensor } from "./runtime.js";

const PIXELS = 64 * 64;

/** Karras noise levels (identical to dreamcade.models.edm.karras_sigmas). */
export function karrasSigmas(steps, sigmaMin, sigmaMax, rho) {
  if (steps === 1) return [sigmaMax, 0];
  const lo = sigmaMin ** (1 / rho);
  const hi = sigmaMax ** (1 / rho);
  const out = [];
  for (let i = 0; i < steps; i++) out.push((hi + (i / (steps - 1)) * (lo - hi)) ** rho);
  out.push(0);
  return out;
}

export class DungeonDream {
  constructor(session, meta) {
    this.session = session;
    this.meta = meta;
    this.n = meta.context;
    this.steps = meta.defaultSteps;
    this.ctxSigma = meta.defaultCtxSigma;
    this.ctx = new Float32Array(this.n * 3 * PIXELS); // clean context frames in [-1, 1]
    this.actions = new BigInt64Array(this.n);
    this.randn = gaussianSource(1234);
    this.lastMs = 0;
  }

  /** Fill the memory with real frames (RGB bytes) and their actions. */
  reset(frames, actions) {
    frames.forEach((rgb, i) => this.ctx.set(toPlanar(rgb), i * 3 * PIXELS));
    actions.forEach((a, i) => { this.actions[i] = BigInt(a); });
  }

  /** "Corrupt memory": drown every remembered frame in heavy noise. */
  corrupt(strength = 0.8) {
    for (let i = 0; i < this.ctx.length; i++) this.ctx[i] = Math.max(-1, Math.min(1, this.ctx[i] + strength * this.randn()));
  }

  /** Dream the next frame for `action`; returns RGB bytes. */
  async step(action) {
    const t0 = performance.now();
    this.actions.copyWithin(0, 1);
    this.actions[this.n - 1] = BigInt(action);
    const ctx = new Float32Array(this.ctx.length);
    for (let i = 0; i < ctx.length; i++) ctx[i] = this.ctx[i] + this.ctxSigma * this.randn();
    const sigmas = karrasSigmas(this.steps, this.meta.sigmaMin, this.meta.sigmaMax, this.meta.rho);
    let x = new Float32Array(3 * PIXELS);
    for (let i = 0; i < x.length; i++) x[i] = this.randn() * sigmas[0];
    const feeds = {
      ctx: tensor("float32", ctx, [1, 3 * this.n, 64, 64]),
      actions: tensor("int64", this.actions.slice(), [1, this.n]),
      ctx_sigma: tensor("float32", new Float32Array([this.ctxSigma]), [1]),
    };
    for (let k = 0; k < sigmas.length - 1; k++) {
      const sigma = sigmas[k];
      const next = sigmas[k + 1];
      const res = await this.session.run({
        ...feeds,
        x: tensor("float32", x, [1, 3, 64, 64]),
        sigma: tensor("float32", new Float32Array([sigma]), [1]),
      });
      const denoised = res.out.data;
      if (next === 0) {
        x = Float32Array.from(denoised);
      } else {
        for (let i = 0; i < x.length; i++) x[i] = x[i] + ((x[i] - denoised[i]) / sigma) * (next - sigma);
      }
    }
    for (let i = 0; i < x.length; i++) x[i] = Math.max(-1, Math.min(1, x[i]));
    this.ctx.copyWithin(0, 3 * PIXELS);
    this.ctx.set(x, (this.n - 1) * 3 * PIXELS);
    this.lastMs = performance.now() - t0;
    return toInterleaved(x);
  }
}

function toPlanar(rgb) {
  const out = new Float32Array(3 * PIXELS);
  for (let p = 0; p < PIXELS; p++) {
    for (let c = 0; c < 3; c++) out[c * PIXELS + p] = rgb[p * 3 + c] / 127.5 - 1;
  }
  return out;
}

function toInterleaved(x) {
  const out = new Uint8ClampedArray(3 * PIXELS);
  for (let p = 0; p < PIXELS; p++) {
    for (let c = 0; c < 3; c++) out[p * 3 + c] = Math.round((x[c * PIXELS + p] + 1) * 127.5);
  }
  return out;
}
