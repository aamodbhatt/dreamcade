// onnxruntime-web session creation: fetch the split model parts, verify, pick a backend.

export const ORT_VERSION = "1.30.0";
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

function ortGlobal() {
  const ort = globalThis.ort;
  if (!ort) throw new Error("onnxruntime-web failed to load (blocked CDN or integrity mismatch)");
  return ort;
}

async function sha256Hex(bytes) {
  if (!globalThis.crypto?.subtle) return null; // non-secure contexts have no SubtleCrypto
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Download every part of a model, concatenate and check size + SHA-256 from the manifest. */
export async function fetchModel(base, entry, onProgress = () => {}) {
  const chunks = [];
  let loaded = 0;
  for (const part of entry.parts) {
    const res = await fetch(base + part);
    if (!res.ok) throw new Error(`model part ${part}: HTTP ${res.status}`);
    const buf = new Uint8Array(await res.arrayBuffer());
    chunks.push(buf);
    loaded += buf.length;
    onProgress(loaded / entry.bytes);
  }
  if (loaded !== entry.bytes) throw new Error(`model is ${loaded} bytes, manifest says ${entry.bytes}`);
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.length;
  }
  const digest = await sha256Hex(bytes);
  if (digest !== null && digest !== entry.sha256) throw new Error("model checksum mismatch: re-download the repository");
  return bytes;
}

/** Create a session, trying WebGPU first (unless disabled) and falling back to WASM. */
export async function createSession(bytes, preferred) {
  const ort = ortGlobal();
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const order = preferred === "wasm" || !navigator.gpu ? ["wasm"] : ["webgpu", "wasm"];
  let lastError = null;
  for (const backend of order) {
    try {
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: [backend],
        graphOptimizationLevel: "all",
      });
      return { session, backend };
    } catch (err) {
      lastError = err;
    }
  }
  throw new Error(`no onnxruntime backend could load the model: ${lastError}`);
}

export function tensor(type, data, dims) {
  return new (ortGlobal().Tensor)(type, data, dims);
}

/** Small seeded PRNG (mulberry32) returning uniforms in [0, 1). */
export function uniformSource(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Seeded standard normals (Box-Muller), so dreams are reproducible run to run. */
export function gaussianSource(seed) {
  const uniform = uniformSource(seed);
  return () => {
    const u = Math.max(uniform(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * uniform());
  };
}
