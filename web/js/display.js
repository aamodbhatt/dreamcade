// Drawing 64x64 frames (palette or RGB) and their difference onto canvases.
import { PALETTE_RGB } from "./breakout/engine.js";

const PIXELS = 64 * 64;

function toImage(ctx, fill) {
  const img = ctx.createImageData(64, 64);
  for (let p = 0; p < PIXELS; p++) {
    const [r, g, b] = fill(p);
    img.data[p * 4] = r;
    img.data[p * 4 + 1] = g;
    img.data[p * 4 + 2] = b;
    img.data[p * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** Convert a frame to interleaved RGB bytes (palette frames are looked up). */
export function asRgb(frame, palette) {
  if (!palette) return frame;
  const out = new Uint8ClampedArray(PIXELS * 3);
  for (let p = 0; p < PIXELS; p++) out.set(PALETTE_RGB[frame[p]], p * 3);
  return out;
}

export function drawRgb(canvas, rgb) {
  toImage(canvas.getContext("2d"), (p) => [rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2]]);
}

// A perceptual "inferno"-like ramp: black -> purple -> orange -> pale yellow.
const RAMP = [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]];
function ramp(t) {
  const x = Math.min(0.9999, Math.max(0, t)) * (RAMP.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  return RAMP[i].map((v, k) => Math.round(v + (RAMP[i + 1][k] - v) * f));
}

/** Draw |a - b| as a heat map and return the PSNR (dB) between the two RGB frames. */
export function drawDiff(canvas, a, b) {
  let se = 0;
  toImage(canvas.getContext("2d"), (p) => {
    let d = 0;
    for (let c = 0; c < 3; c++) {
      const e = a[p * 3 + c] - b[p * 3 + c];
      se += e * e;
      d += Math.abs(e);
    }
    return ramp((d / 3 / 255) * 3);
  });
  const mse = se / (PIXELS * 3);
  return mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);
}
