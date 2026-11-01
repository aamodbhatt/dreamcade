# dreamcade design

## Goals

- Two games that are *played inside a neural network*: every frame the player sees in the
  Dream view is produced by a model from past frames and key presses, with no engine in the loop.
- A real engine next to every dream, fed the same inputs, so fidelity is visible (Split, Diff)
  and measurable (PSNR, LPIPS, exact-frame rate) rather than asserted.
- Research hygiene: deterministic engines, held-out evaluation, ablations that change one
  thing, probes with a random-network control, and honest reporting of what fails.
- Runs in a browser with no build step and no server-side inference.

## Non-goals

- Learning from pixels of commercial games (both engines are written here, so ground truth is exact).
- Long-term memory: the dungeon model sees 4 frames. Remembering a room you left a minute ago
  is out of scope; the failure gallery shows what happens instead.
- Training an agent. Data comes from scripted exploration policies.

## Architecture

```
engines (numpy, deterministic) ──► shards (.npy, memory-mapped) ──► training (PyTorch, bf16, A40)
      │                                                                  │
      └── JS ports (pixel-identical) ─┐                                  ▼
                                      ├──► browser player ◄── ONNX (fp16 storage) ◄── EMA weights
                     onnxruntime-web ─┘
```

| Module | Responsibility |
| --- | --- |
| `dreamcade.game` | Breakout rules + palette renderer (integer-exact). |
| `dreamcade.dungeon` | World data, vectorised fixed-point movement, vectorised raycaster. |
| `dreamcade.data` | Exploration policies, shard I/O, generation and coverage statistics. |
| `dreamcade.models` | Shared U-Net blocks, the dungeon EDM U-Net + sampler, the Breakout classifier. |
| `dreamcade.train` | GPU-resident data windows, EMA, JSONL metrics, atomic checkpoints, both training loops. |
| `dreamcade.evaluate` | Rollout metrics, ablations, probes, galleries and figures. |
| `dreamcade.export` | ONNX export, fp16 storage, parity check, split-file manifest. |
| `web/js` | JS engine ports, ONNX sessions, Euler sampler, views, input, attract mode. |

## Invariants

1. **Engines are deterministic and integer-exact.** Breakout uses only integers. The dungeon
   keeps positions in fixed point (256 units per tile), headings as one of 64 indices and the
   trig table as integers; the renderer's floating-point work is IEEE-754 `+ - * /` written in
   the same order in Python and JS, with explicit `floor`. Golden trajectories (601 Breakout
   frames, 301 dungeon frames, FNV-1a hashes per frame) must replay identically in Node.
2. **The browser world is the Python world.** `web/assets/dungeon_world.json` is generated
   from `dreamcade.dungeon.world`; a test fails if it drifts.
3. **Breakout state is visible.** The engine resolves collisions by look-ahead reflection, so
   the displacement between two frames *is* the velocity. A two-frame context therefore
   determines the next frame, which is why per-pixel classification (not diffusion) is the
   right model there.
4. **Holdout episodes are never trained on.** The last episodes of the last shard are reserved.
5. **Exported weights are what was evaluated.** Parity is checked against a PyTorch model whose
   weights were rounded to fp16 exactly like the ONNX initialisers.

## Data model

A shard is `frames.npy` (`uint8 [E, T+1, ...]`), `actions.npy` (`uint8 [E, T]`),
`states.npy` (`int64 [E, T+1, S]`) and `meta.json`. Fixed-length episodes make a training
window a single gather. During training the whole dataset (3.5 GB of dungeon frames) lives in
GPU memory as `uint8`, so there is no data loader to starve the GPU on 8 shared CPU cores.

## The dungeon world model

- **EDM preconditioning** (`c_skip, c_out, c_in, c_noise`), log-normal training noise
  (`P_mean = -0.4`, `P_std = 1.2`), Karras schedule (`sigma_max = 5`, `rho = 7`) and a
  deterministic Euler sampler: the DIAMOND recipe, chosen because it stays sharp at 3 steps.
- **Conditioning:** the last 4 frames are concatenated on channels with the noisy next frame;
  the 4 actions (one embedding table per slot), the noise level and the context-noise level
  are summed into one vector that drives adaptive GroupNorm in every residual block.
- **Context-noise augmentation:** each training example's context gets Gaussian noise with a
  level `s ~ U(0, 0.3)` (zero 10% of the time) and the model is told `s`. At play time the
  model's own frames are slightly wrong; trained this way it treats them as mildly noisy real
  frames and pulls them back on-manifold instead of compounding the error.
- **Pixel-unshuffle stem:** the U-Net runs at 32x32 (channels 96/192/256, attention at 8x8),
  roughly 4x cheaper than a full-resolution stem. 18.5M parameters.
- **EMA** weights (0.999) are evaluated and exported.

## Failure modes (and what the system does)

| Failure | Handling |
| --- | --- |
| Model files missing or truncated | Player checks byte count and SHA-256 from the manifest and says how to fix it. |
| WebGPU unavailable | Falls back to WASM and shows the backend in the UI. |
| Dream drifts away from the real game | Visible in Split/Diff with live PSNR; **Restart** re-syncs both from one frame. |
| Corrupted memory | The "Corrupt memory" button injects it on purpose so recovery can be watched. |
| A typo in a config | Unknown keys raise with the list of valid keys. |
| A job dies mid-run | Checkpoints are written atomically; training resumes from the last one. |
| Rays escaping a broken map | The raycaster raises instead of looping forever. |

## Alternatives considered

- **Latent diffusion (as in GameNGen).** At 64x64 an autoencoder buys little compute and adds a
  second model and its artefacts; pixel-space EDM with a pixel-unshuffle stem was simpler.
- **Diffusion for Breakout.** Unnecessary: the game is deterministic given two frames, so a
  classifier is exact when right and its mistakes are discrete and easy to measure.
- **Full-resolution U-Net.** About 4x the FLOPs per denoising step for little visible gain at
  this resolution, and it would push browser inference far from interactive.
- **Separate models per context length.** Not affordable in a 75 GPU-minute budget; the context
  length is fixed at 4 and its ablation is listed in the roadmap.
- **96x72 dungeon frames.** 1.7x more pixels per step; 64x64 keeps WASM inference closer to
  interactive and matches the Atari setting of DIAMOND.
