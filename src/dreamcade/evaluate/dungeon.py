"""Dungeon world-model evaluation: autoregressive rollouts against the real raycaster.

For every scripted action sequence we start from real context frames, let the model dream
``HORIZON`` frames on its own outputs, and compare each dreamed frame to the engine's frame
for the same inputs (PSNR and LPIPS). Sampling-time ablations: number of Euler steps and the
context-noise level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from dreamcade.config import DungeonTrainConfig, config_from_dict
from dreamcade.data.dungeon_policies import Explorer
from dreamcade.dungeon import engine as dengine
from dreamcade.dungeon.engine import Action
from dreamcade.dungeon.render import render
from dreamcade.evaluate.plots import line_plot, save_strip, training_curves
from dreamcade.models.edm import Denoiser, psnr
from dreamcade.train.dungeon import build_denoiser, sample_next, to_unit

HORIZON = 48
STARTS = 32
WARMUP = 6
F, L, R, SL, SR = (
    int(a)
    for a in (Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.STRAFE_LEFT, Action.STRAFE_RIGHT)
)
SCRIPTS: dict[str, list[int]] = {
    "walk": (([F] * 10 + [L] * 4 + [F] * 10 + [R] * 4) * 2)[:HORIZON],
    "turn_around": [R] * 32 + [F] * 16,
    "spin": [L] * 48,
    "strafe": ([SL] * 6 + [SR] * 6) * 4,
}


@dataclass(frozen=True, slots=True)
class Episode:
    """Ground truth for one start: context frames/actions and the scripted future."""

    frames: npt.NDArray[np.uint8]  # [STARTS, WARMUP + HORIZON + 1, 64, 64, 3]
    actions: npt.NDArray[np.int64]  # [STARTS, WARMUP + HORIZON]


def ground_truth(script: list[int], seed: int) -> Episode:
    """Spawn ``STARTS`` agents, warm up with the exploration policy, then run ``script``."""
    rng = np.random.default_rng(seed)
    state = dengine.spawn(rng, STARTS)
    policy = Explorer(rng, STARTS)
    moved = np.ones(STARTS, dtype=np.bool_)
    frames = [render(state)]
    actions = []
    for t in range(WARMUP + len(script)):
        act = policy.act(state, moved) if t < WARMUP else np.full(STARTS, script[t - WARMUP], dtype=np.int64)
        state, moved = dengine.step(state, act)
        actions.append(act)
        frames.append(render(state))
    return Episode(np.stack(frames, axis=1), np.stack(actions, axis=1))


@torch.no_grad()
def rollout(
    model: Denoiser, ep: Episode, context: int, steps: int, ctx_sigma: float, seed: int, device: torch.device
) -> Tensor:
    """Dream ``HORIZON`` frames; returns ``[STARTS, HORIZON, 3, 64, 64]`` in ``[-1, 1]``."""
    gen = torch.Generator(device=device).manual_seed(seed)
    frames = to_unit(torch.from_numpy(ep.frames).to(device))  # [S, T, 3, H, W]
    actions = torch.from_numpy(ep.actions).to(device)
    start = WARMUP + 1 - context  # context = the last `context` real frames before the script
    ctx = [frames[:, start + i] for i in range(context)]
    out = []
    for t in range(ep.actions.shape[1] - WARMUP):
        a = actions[:, start + t : start + t + context]
        c = torch.cat(ctx, dim=1)
        c = c + ctx_sigma * torch.randn(c.shape, generator=gen, device=device)
        noise = torch.randn((c.shape[0], 3, 64, 64), generator=gen, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            nxt = sample_next(model, c, a, ctx_sigma, steps, noise).float()
        nxt = (nxt * 127.5 + 127.5).round() / 127.5 - 1  # quantise like a displayed frame
        out.append(nxt)
        ctx = [*ctx[1:], nxt]
    return torch.stack(out, dim=1)


class Scorer:
    """PSNR and LPIPS (AlexNet) per frame."""

    def __init__(self, device: torch.device) -> None:
        import lpips

        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(
        self, pred: Tensor, target: Tensor
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per-step means over starts: ``(psnr[HORIZON], lpips[HORIZON])``."""
        s, h = pred.shape[:2]
        p = pred.reshape(s * h, 3, 64, 64)
        t = target.reshape(s * h, 3, 64, 64)
        ps = psnr(p, t).reshape(s, h).mean(0)
        lp = torch.cat([self.lpips(p[i : i + 512], t[i : i + 512]).flatten() for i in range(0, s * h, 512)])
        return ps.cpu().double().numpy(), lp.reshape(s, h).mean(0).cpu().double().numpy()


def load_model(run: Path, device: torch.device) -> tuple[Denoiser, DungeonTrainConfig, int]:
    """EMA weights and config from a run directory."""
    ckpt: dict[str, Any] = torch.load(run / "ema.pt", map_location=device, weights_only=False)
    cfg = config_from_dict(DungeonTrainConfig, ckpt["config"])
    model = build_denoiser(cfg).to(device).eval()
    model.load_state_dict(ckpt["ema"])
    return model, cfg, int(ckpt["step"])


def main(run: Path, data: Path, out: Path) -> None:
    """Run every evaluation and write JSON, tables and figures to ``out``."""
    del data  # ground truth comes straight from the engine for scripted actions
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step = load_model(run, device)
    scorer = Scorer(device)
    episodes = {name: ground_truth(script, seed=100 + i) for i, (name, script) in enumerate(SCRIPTS.items())}
    targets = {
        name: to_unit(torch.from_numpy(ep.frames[:, WARMUP + 1 :]).to(device))
        for name, ep in episodes.items()
    }

    def score(steps: int, ctx_sigma: float) -> dict[str, dict[str, list[float]]]:
        res: dict[str, dict[str, list[float]]] = {}
        for name, ep in episodes.items():
            pred = rollout(model, ep, cfg.context, steps, ctx_sigma, seed=7, device=device)
            ps, lp = scorer(pred, targets[name])
            res[name] = {"psnr": ps.round(3).tolist(), "lpips": lp.round(4).tolist()}
        return res

    default = score(3, 0.05)
    step_ablation = {k: score(k, 0.05) for k in (1, 2, 4, 8)} | {3: default}
    noise_ablation = {s: score(3, s) for s in (0.0, 0.1, 0.2)} | {0.05: default}

    def summary(res: dict[str, dict[str, list[float]]]) -> dict[str, float]:
        ps = np.mean([np.mean(v["psnr"]) for v in res.values()])
        lp = np.mean([np.mean(v["lpips"]) for v in res.values()])
        first = np.mean([np.mean(v["psnr"][:8]) for v in res.values()])
        return {
            "psnr": round(float(ps), 2),
            "lpips": round(float(lp), 4),
            "psnr_first8": round(float(first), 2),
        }

    results: dict[str, Any] = {
        "trained_steps": step,
        "horizon": HORIZON,
        "starts": STARTS,
        "context": cfg.context,
        "default": {"steps": 3, "ctx_sigma": 0.05, "per_script": default, "summary": summary(default)},
        "sampling_steps": {str(k): summary(v) for k, v in sorted(step_ablation.items())},
        "context_noise": {str(s): summary(v) for s, v in sorted(noise_ablation.items())},
    }
    (out / "dungeon_eval.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    hx = list(range(1, HORIZON + 1))
    line_plot(
        out / "dungeon_psnr_horizon.png",
        {name: (hx, v["psnr"]) for name, v in default.items()},
        "dream step",
        "PSNR vs real (dB)",
        "Dungeon: fidelity over a 48-frame dream",
    )
    line_plot(
        out / "dungeon_lpips_horizon.png",
        {name: (hx, v["lpips"]) for name, v in default.items()},
        "dream step",
        "LPIPS vs real (lower is better)",
        "Dungeon: perceptual distance over the dream",
    )
    _gallery(model, episodes, cfg, device, out)
    training_curves(
        run / "metrics.jsonl",
        out / "dungeon_training.png",
        ["train/loss", "val/psnr_1step"],
        "Dungeon diffusion training",
    )
    print(json.dumps({k: results[k] for k in ("sampling_steps", "context_noise")}, indent=2))
    print(json.dumps(results["default"]["summary"]))


def _gallery(
    model: Denoiser, episodes: dict[str, Episode], cfg: DungeonTrainConfig, device: torch.device, out: Path
) -> None:
    """Real-vs-dream strips: the typical case and the worst turn-around (dream logic)."""
    picks = (0, 8, 16, 24, 32, 40, 47)
    for name in ("walk", "turn_around", "spin"):
        ep = episodes[name]
        pred = rollout(model, ep, cfg.context, 3, 0.05, seed=7, device=device)
        target = to_unit(torch.from_numpy(ep.frames[:, WARMUP + 1 :]).to(device))
        err = ((pred - target) ** 2).flatten(2).mean(-1)[:, -16:].mean(-1)  # late-horizon error per start
        order = torch.argsort(err)
        chosen = [int(order[len(order) // 2]), int(order[-1])]  # median and worst
        rows = []
        for idx in chosen:
            rows.append([target[idx, t] for t in picks])
            rows.append([pred[idx, t] for t in picks])
        save_strip(
            out / f"dungeon_gallery_{name}.png",
            rows,
            [f"t+{t + 1}" for t in picks],
            ["real", "dream", "real", "dream"],
        )
