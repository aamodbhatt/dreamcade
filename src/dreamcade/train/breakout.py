"""Train the Breakout next-frame classifier.

Two regimes are compared in the rollout-fidelity study:

* teacher forcing (``ctx_flip_max = 0``): context frames are always ground truth;
* context noise (``ctx_flip_max > 0``): each context pixel is re-coloured at random with a
  per-sample probability, the discrete analogue of Gaussian context noise in the dungeon.

The loss up-weights pixels that change between frames: otherwise "copy the last frame" is
already 99% pixel-accurate and the ball (4 of 4096 pixels) would barely matter.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from dreamcade.config import BreakoutTrainConfig, load_breakout
from dreamcade.game.constants import NUM_COLORS
from dreamcade.models.breakout_net import BreakoutNet, BreakoutSpec
from dreamcade.models.unet import count_params
from dreamcade.train.common import (
    EMA,
    EpisodeStore,
    MetricsLog,
    config_dict,
    lr_at,
    pick_device,
    save_checkpoint,
    seed_everything,
)


def build_model(cfg: BreakoutTrainConfig) -> BreakoutNet:
    """Model from a config."""
    return BreakoutNet(BreakoutSpec(context=cfg.context, channels=cfg.channels))


def flip_context(ctx: Tensor, flip_max: float) -> Tensor:
    """Re-colour each context pixel with probability ``p ~ U(0, flip_max)`` (per sample)."""
    if flip_max <= 0:
        return ctx
    p = torch.rand(ctx.shape[0], 1, 1, 1, device=ctx.device) * flip_max
    mask = torch.rand(ctx.shape, device=ctx.device) < p
    noise = torch.randint(0, NUM_COLORS, ctx.shape, device=ctx.device, dtype=ctx.dtype)
    return torch.where(mask, noise, ctx)


def weighted_ce(logits: Tensor, target: Tensor, last: Tensor, changed_weight: float) -> Tensor:
    """Cross-entropy with extra weight on pixels that differ from the last context frame."""
    ce = F.cross_entropy(logits.float(), target.long(), reduction="none")
    w = 1 + (changed_weight - 1) * (target != last).float()
    return (ce * w).sum() / w.sum()


@torch.no_grad()
def _validate(model: BreakoutNet, store: EpisodeStore, cfg: BreakoutTrainConfig) -> dict[str, float]:
    gen = torch.Generator()
    gen.manual_seed(99)
    frames, actions, _ = store.windows(256, cfg.context, gen)
    pred = model(frames[:, : cfg.context], actions[:, cfg.context - 1]).argmax(1)
    target = frames[:, cfg.context].long()
    changed = target != frames[:, cfg.context - 1].long()
    return {
        "val/pixel_acc": float((pred == target).float().mean()),
        "val/changed_acc": float((pred == target)[changed].float().mean()),
        "val/frame_exact": float((pred == target).flatten(1).all(1).float().mean()),
    }


def train(cfg: BreakoutTrainConfig, resume: bool = True) -> Path:
    """Run training until ``max_steps`` or ``max_minutes``."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    gen = seed_everything(cfg.seed)
    store_device = device if cfg.data_on_gpu else torch.device("cpu")
    train_store = EpisodeStore.load(Path(cfg.data), store_device, cfg.holdout_episodes, "train")
    val_store = EpisodeStore.load(Path(cfg.data), store_device, cfg.holdout_episodes, "holdout")
    model = build_model(cfg).to(device)
    ema = EMA(model, cfg.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step, elapsed = 0, 0.0
    ckpt_path = out / "checkpoint.pt"
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        step, elapsed = int(ckpt["step"]), float(ckpt["elapsed"])
        gen.set_state(ckpt["gen"])
        print(f"resumed from step {step}")
    print(f"params: {count_params(model) / 1e6:.2f}M, train episodes {train_store.episodes}, device {device}")
    log = MetricsLog(out / "metrics.jsonl")

    def checkpoint() -> None:
        payload: dict[str, Any] = {
            "model": model.state_dict(),
            "ema": ema.model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "elapsed": elapsed,
            "gen": gen.get_state(),
            "config": config_dict(cfg),
        }
        save_checkpoint(ckpt_path, payload)
        save_checkpoint(
            out / "ema.pt", {"ema": ema.model.state_dict(), "step": step, "config": config_dict(cfg)}
        )

    autocast = cfg.bf16 and device.type == "cuda"
    started = time.time() - elapsed
    t_last, n_last = time.time(), 0
    while step < cfg.max_steps and elapsed < cfg.max_minutes * 60:
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg.lr, cfg.warmup, cfg.max_steps)
        frames, actions, _ = train_store.windows(cfg.batch, cfg.context, gen)
        ctx = flip_context(frames[:, : cfg.context], cfg.ctx_flip_max)
        target = frames[:, cfg.context]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            logits = model(ctx, actions[:, cfg.context - 1])
        loss = weighted_ce(logits, target, frames[:, cfg.context - 1], cfg.changed_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model, step)
        step += 1
        n_last += cfg.batch
        elapsed = time.time() - started
        if step % cfg.log_every == 0:
            now = time.time()
            log.write(
                step,
                {
                    "train/loss": float(loss.detach()),
                    "lr": opt.param_groups[0]["lr"],
                    "samples_per_s": n_last / (now - t_last),
                },
            )
            t_last, n_last = now, 0
        if step % cfg.ckpt_every == 0:
            log.write(step, _validate(ema.model, val_store, cfg))
            checkpoint()
    log.write(step, _validate(ema.model, val_store, cfg))
    checkpoint()
    log.close()
    print(f"finished at step {step} after {elapsed / 60:.1f} min")
    return out


def main(config: Path, resume: bool = True) -> None:
    """CLI entry point."""
    train(load_breakout(config), resume=resume)
