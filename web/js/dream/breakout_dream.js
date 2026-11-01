// The Breakout dream: one forward pass per frame, palette indices in and out.
import { tensor, uniformSource } from "./runtime.js";

const PIXELS = 64 * 64;

export class BreakoutDream {
  constructor(session, meta) {
    this.session = session;
    this.n = meta.context;
    this.ctx = new Float32Array(this.n * PIXELS);
    this.rand = uniformSource(99);
    this.lastMs = 0;
  }

  /** Fill the memory with real palette frames. */
  reset(frames) {
    frames.forEach((f, i) => this.ctx.set(Float32Array.from(f), i * PIXELS));
  }

  /** "Corrupt memory": recolour 15% of remembered pixels at random. */
  corrupt(fraction = 0.15) {
    for (let i = 0; i < this.ctx.length; i++) {
      if (this.rand() < fraction) this.ctx[i] = Math.floor(this.rand() * 8);
    }
  }

  /** Predict the next frame for `action`; returns a Uint8Array of palette indices. */
  async step(action) {
    const t0 = performance.now();
    const res = await this.session.run({
      frames: tensor("float32", this.ctx.slice(), [1, this.n, 64, 64]),
      action: tensor("int64", new BigInt64Array([BigInt(action)]), [1]),
    });
    const frame = Uint8Array.from(res.out.data);
    this.ctx.copyWithin(0, PIXELS);
    this.ctx.set(Float32Array.from(frame), (this.n - 1) * PIXELS);
    this.lastMs = performance.now() - t0;
    return frame;
  }
}
