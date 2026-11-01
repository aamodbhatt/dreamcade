"""Breakout as a study of learned physics.

1. **Rollout fidelity.** From held-out real contexts, each model plays ``HORIZON`` steps on its
   own predictions with the recorded actions; we track exact-frame agreement with the engine,
   ball-position error and the step of first divergence.
2. **Linear probes.** Ridge regression from bottleneck activations to the ball's position and
   velocity, against the same probe on a randomly initialised network of the same shape.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from dreamcade.config import BreakoutTrainConfig, config_from_dict
from dreamcade.data.generate import BREAKOUT_STATE_FIELDS
from dreamcade.evaluate.plots import bar_plot, line_plot, training_curves
from dreamcade.game.constants import Color, Mode
from dreamcade.models.breakout_net import BreakoutNet
from dreamcade.train.breakout import build_model
from dreamcade.train.common import EpisodeStore

HORIZON = 120
STARTS = 256
PROBE_TARGETS = ("ball_x", "ball_y", "vx", "vy")


def load(run: Path, device: torch.device) -> tuple[BreakoutNet, BreakoutTrainConfig]:
    """EMA model and config."""
    ckpt: dict[str, Any] = torch.load(run / "ema.pt", map_location=device, weights_only=False)
    cfg = config_from_dict(BreakoutTrainConfig, ckpt["config"])
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["ema"])
    return model, cfg


def ball_centroid(frames: Tensor) -> tuple[Tensor, Tensor]:
    """Ball centroid ``[N, 2]`` and a visibility mask from palette frames ``[N, H, W]``."""
    mask = (frames == Color.BALL).float()
    count = mask.flatten(1).sum(1)
    ys = torch.arange(frames.shape[1], device=frames.device).float()
    xs = torch.arange(frames.shape[2], device=frames.device).float()
    cy = (mask.sum(2) * ys).sum(1) / count.clamp_min(1)
    cx = (mask.sum(1) * xs).sum(1) / count.clamp_min(1)
    return torch.stack([cx, cy], 1), count > 0


@torch.no_grad()
def rollout_fidelity(
    model: BreakoutNet, store: EpisodeStore, context: int, gen: torch.Generator
) -> dict[str, Any]:
    """Autoregressive rollouts from real contexts with the recorded actions."""
    e = torch.randint(0, store.episodes, (STARTS,), generator=gen)
    t = torch.randint(0, store.steps - context - HORIZON + 1, (STARTS,), generator=gen)
    frames, actions, _ = store.window_at(e, t, context + HORIZON - 1)
    ctx = frames[:, :context].long()
    exact, pixel, ball_err, ball_lost = [], [], [], []
    diverged = torch.full((STARTS,), HORIZON, device=frames.device)
    for h in range(HORIZON):
        pred = model(ctx, actions[:, context - 1 + h]).argmax(1)
        real = frames[:, context + h].long()
        ok = (pred == real).flatten(1).all(1)
        diverged = torch.where(~ok & (diverged == HORIZON), torch.full_like(diverged, h), diverged)
        exact.append(float(ok.float().mean()))
        pixel.append(float((pred == real).float().mean()))
        (pc, pv), (rc, rv) = ball_centroid(pred), ball_centroid(real)
        both = pv & rv
        ball_err.append(float((pc - rc).norm(dim=1)[both].mean()) if both.any() else float("nan"))
        ball_lost.append(float((rv & ~pv).float().sum() / rv.float().sum().clamp_min(1)))
        ctx = torch.cat([ctx[:, 1:], pred[:, None]], dim=1)
    return {
        "exact_frame": exact,
        "pixel_acc": pixel,
        "ball_err_px": ball_err,
        "ball_lost": ball_lost,
        "median_first_divergence": float(diverged.float().median()),
        "exact_at": {str(h): round(exact[h - 1], 4) for h in (1, 10, 30, 60, 120)},
    }


def _ridge_r2(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-6
    a, b = (x_train - mu) / sd, (x_test - mu) / sd
    best = -np.inf
    for lam in (1.0, 10.0, 100.0, 1000.0):
        # Dual form: features (12k) outnumber samples, so solve the n x n system.
        k = a @ a.T
        alpha = np.linalg.solve(k + lam * np.eye(len(a)), y_train - y_train.mean())
        pred = b @ (a.T @ alpha) + y_train.mean()
        r2 = 1 - ((y_test - pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
        best = max(best, float(r2))
    return best


@torch.no_grad()
def probes(
    model: BreakoutNet, random_model: BreakoutNet, store: EpisodeStore, context: int, gen: torch.Generator
) -> dict[str, dict[str, float]]:
    """R^2 of linear probes for ball state, trained vs random network (held-out split)."""
    e = torch.randint(0, store.episodes, (6000,), generator=gen)
    t = torch.randint(0, store.steps - context + 1, (6000,), generator=gen)
    frames, actions, states = store.window_at(e, t, context)
    fields = list(BREAKOUT_STATE_FIELDS)
    last = states[:, context - 1]
    keep = (last[:, fields.index("mode")] == Mode.PLAY).nonzero().flatten()[:3000]
    ctx, act = frames[keep, :context], actions[keep, context - 1]
    targets = {name: last[keep, fields.index(name)].float().cpu().numpy() for name in PROBE_TARGETS}
    out: dict[str, dict[str, float]] = {}
    for label, net in (("trained", model), ("random init", random_model)):
        feats = torch.cat(
            [net.features(ctx[i : i + 256], act[i : i + 256]).flatten(1) for i in range(0, len(keep), 256)]
        )
        x = feats.float().cpu().numpy()
        n = int(0.7 * len(x))
        out[label] = {name: round(_ridge_r2(x[:n], y[:n], x[n:], y[n:]), 4) for name, y in targets.items()}
    return out


def main(runs: list[Path], data: Path, out: Path) -> None:
    """Evaluate every run, write ``breakout_eval.json`` and figures."""
    if not runs:
        msg = "pass at least one run directory with --runs"
        raise SystemExit(msg)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {"horizon": HORIZON, "starts": STARTS, "runs": {}}
    curves: dict[str, tuple[Sequence[float], Sequence[float]]] = {}
    for run in runs:
        model, cfg = load(run, device)
        store = EpisodeStore.load(data, device, cfg.holdout_episodes, "holdout")
        gen = torch.Generator().manual_seed(5)
        fid = rollout_fidelity(model, store, cfg.context, gen)
        label = "context noise" if cfg.ctx_flip_max > 0 else "teacher forcing"
        results["runs"][label] = {"run": run.name, "ctx_flip_max": cfg.ctx_flip_max, **fid}
        curves[label] = (list(range(1, HORIZON + 1)), fid["exact_frame"])
        torch.manual_seed(0)
        random_model = build_model(cfg).to(device).eval()
        results["runs"][label]["probes"] = probes(
            model, random_model, store, cfg.context, torch.Generator().manual_seed(6)
        )
        training_curves(
            run / "metrics.jsonl",
            out / f"breakout_training_{run.name}.png",
            ["train/loss", "val/frame_exact"],
            f"Breakout training ({label})",
        )
    (out / "breakout_eval.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    line_plot(
        out / "breakout_rollout.png",
        curves,
        "steps dreamed on its own outputs",
        "exact-frame match rate",
        "Breakout: rollout fidelity",
    )
    first = next(iter(results["runs"].values()))["probes"]
    bar_plot(
        out / "breakout_probes.png",
        {t: {k: max(0.0, first[k][t]) for k in first} for t in PROBE_TARGETS},
        "held-out R^2",
        "Linear probes on the bottleneck",
    )
    print(
        json.dumps(
            {
                k: {m: v[m] for m in ("exact_at", "median_first_divergence", "probes")}
                for k, v in results["runs"].items()
            },
            indent=2,
        )
    )
