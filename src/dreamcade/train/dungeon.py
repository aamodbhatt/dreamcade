"""Train the dungeon diffusion world model.

Each example is ``context`` past frames and actions plus the next frame. The context frames
receive Gaussian noise of a random level that the network is told about (context-noise
augmentation, as in GameNGen): at play time the model's own slightly-off frames look like
mildly noisy real frames, so errors are corrected instead of compounded.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dreamcade.config import DungeonTrainConfig, load_dungeon
from dreamcade.dungeon.engine import NUM_ACTIONS
from dreamcade.models.edm import Denoiser, EDMSchedule, euler_sample, karras_sigmas, psnr
from dreamcade.models.unet import DungeonUNet, UNetSpec, count_params
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


def to_unit(frames: Tensor) -> Tensor:
    """uint8 ``[..., H, W, 3]`` -> float ``[..., 3, H, W]`` in ``[-1, 1]``."""
    return frames.movedim(-1, -3).float() / 127.5 - 1


def build_denoiser(cfg: DungeonTrainConfig) -> Denoiser:
    """Model from a config."""
    spec = UNetSpec(
        context=cfg.context,
        channels=cfg.channels,
        blocks=cfg.blocks,
        attn_levels=cfg.attn_levels,
        emb_dim=cfg.emb_dim,
        num_actions=NUM_ACTIONS,
    )
    return Denoiser(DungeonUNet(spec), EDMSchedule())


def split_window(frames: Tensor, actions: Tensor, context: int) -> tuple[Tensor, Tensor, Tensor]:
    """Window -> (context frames ``[B, 3*context, H, W]``, actions ``[B, context]``, target)."""
    x = to_unit(frames)
    b, _, c, h, w = x.shape
    return x[:, :context].reshape(b, context * c, h, w), actions[:, :context], x[:, context]


def noise_context(ctx: Tensor, cfg: DungeonTrainConfig, gen: torch.Generator | None) -> tuple[Tensor, Tensor]:
    """Context-noise augmentation; returns the noisy context and the per-sample level."""
    b = ctx.shape[0]
    dev = ctx.device
    level = torch.rand(b, generator=gen, device=dev) * cfg.ctx_noise_max
    level = torch.where(torch.rand(b, generator=gen, device=dev) < cfg.ctx_noise_zero_prob, 0.0, level)
    noise = torch.randn(ctx.shape, generator=gen, device=dev)
    return ctx + level[:, None, None, None] * noise, level


@torch.no_grad()
def sample_next(
    model: Denoiser, ctx: Tensor, actions: Tensor, ctx_sigma: float, steps: int, noise: Tensor
) -> Tensor:
    """One generated frame in ``[-1, 1]``. ``ctx`` must already contain any context noise."""
    s = model.schedule
    level = torch.full((ctx.shape[0],), ctx_sigma, device=ctx.device)

    def denoise(x: Tensor, sigma: Tensor) -> Tensor:
        out: Tensor = model(x, sigma, ctx, actions, level)
        return out

    out = euler_sample(denoise, noise, karras_sigmas(steps, s.sigma_min, s.sigma_max, s.rho))
    return out.clamp(-1, 1)


def _validate(
    model: Denoiser, store: EpisodeStore, cfg: DungeonTrainConfig, device: torch.device
) -> dict[str, float]:
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(1234)
    frames, actions, _ = store.windows(64, cfg.context, cpu_gen)
    ctx, act, target = split_window(frames.to(device), actions.to(device), cfg.context)
    noise = torch.randn(target.shape, generator=gen, device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.bf16 and device.type == "cuda"):
        pred = sample_next(model, ctx, act, 0.0, 3, noise)
    return {"val/psnr_1step": float(psnr(pred, target).mean())}


def train(cfg: DungeonTrainConfig, resume: bool = True) -> Path:
    """Run training until ``max_steps`` or ``max_minutes``; returns the run directory."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    gen = seed_everything(cfg.seed)
    store_device = device if cfg.data_on_gpu else torch.device("cpu")
    train_store = EpisodeStore.load(Path(cfg.data), store_device, cfg.holdout_episodes, "train")
    val_store = EpisodeStore.load(Path(cfg.data), store_device, cfg.holdout_episodes, "holdout")

    model = build_denoiser(cfg).to(device)
    ema = EMA(model, cfg.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
    step, elapsed = 0, 0.0
    ckpt_path = out / "checkpoint.pt"
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        step, elapsed = int(ckpt["step"]), float(ckpt["elapsed"])
        gen.set_state(ckpt["gen"])
        print(f"resumed from step {step} ({elapsed / 60:.1f} min)")
    print(f"params: {count_params(model) / 1e6:.2f}M, train episodes {train_store.episodes}, device {device}")
    log = MetricsLog(out / "metrics.jsonl")
    log.write(step, {"params": float(count_params(model))})

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

    budget = cfg.max_minutes * 60
    autocast = cfg.bf16 and device.type == "cuda"
    t_last, n_last = time.time(), 0
    started = time.time() - elapsed
    while step < cfg.max_steps and elapsed < budget:
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg.lr, cfg.warmup, cfg.max_steps)
        frames, actions, _ = train_store.windows(cfg.batch, cfg.context, gen)
        ctx, act, target = split_window(frames.to(device, non_blocking=True), actions.to(device), cfg.context)
        ctx, level = noise_context(ctx, cfg, None)
        sigma = model.sample_sigma(cfg.batch, None, device)
        noise = torch.randn_like(target)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            loss = model.loss(target, ctx, act, level, noise, sigma)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    "train/grad_norm": float(grad_norm),
                    "lr": opt.param_groups[0]["lr"],
                    "samples_per_s": n_last / (now - t_last),
                    "minutes": elapsed / 60,
                },
            )
            t_last, n_last = now, 0
        if step % cfg.ckpt_every == 0:
            log.write(step, _validate(ema.model, val_store, cfg, device))
            checkpoint()
    log.write(step, _validate(ema.model, val_store, cfg, device))
    checkpoint()
    log.close()
    print(f"finished at step {step} after {elapsed / 60:.1f} min")
    return out


def main(config: Path, resume: bool = True) -> None:
    """CLI entry point."""
    train(load_dungeon(config), resume=resume)
