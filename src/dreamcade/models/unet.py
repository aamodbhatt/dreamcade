"""Action-conditioned U-Net for the dungeon diffusion world model.

Inputs are the noisy next frame plus the last ``context`` frames, concatenated on channels
(as in DIAMOND). A 2x2 pixel-unshuffle stem runs the whole network at 32x32, which cuts
compute by ~4x versus a full-resolution stem; that is the difference between an
interactive and a slideshow frame rate in the browser.

Conditioning (noise level, context-noise level and the action history) is summed into one
vector that modulates every residual block through adaptive GroupNorm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dreamcade.models.layers import FourierEmbedding, ResBlock, SelfAttention, group_norm, zero_init


@dataclass(frozen=True, slots=True)
class UNetSpec:
    """Architecture hyper-parameters (stored in checkpoints)."""

    context: int = 4
    channels: tuple[int, ...] = (96, 192, 256)
    blocks: int = 2
    attn_levels: tuple[int, ...] = (2,)
    emb_dim: int = 256
    num_actions: int = 7
    image_channels: int = 3


class Level(nn.Module):
    """A stack of residual blocks (optionally followed by attention) at one resolution."""

    def __init__(
        self, c_in: int, c_out: int, n: int, emb_dim: int, attn: bool, skip_ch: tuple[int, ...] = ()
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        for i in range(n):
            extra = skip_ch[i] if skip_ch else 0
            self.blocks.append(ResBlock((c_in if i == 0 else c_out) + extra, c_out, emb_dim))
            self.attns.append(SelfAttention(c_out) if attn else nn.Identity())


class DungeonUNet(nn.Module):
    """``F_theta`` of the EDM denoiser: maps preconditioned inputs to the network output."""

    def __init__(self, spec: UNetSpec) -> None:
        super().__init__()
        self.spec = spec
        ch = spec.channels
        in_ch = spec.image_channels * (spec.context + 1) * 4
        self.stem = nn.Conv2d(in_ch, ch[0], 3, padding=1)
        self.noise_emb = FourierEmbedding(spec.emb_dim)
        self.ctx_noise_emb = FourierEmbedding(spec.emb_dim)
        self.action_emb = nn.Embedding(spec.num_actions * spec.context, spec.emb_dim)
        self.slot_offsets: Tensor
        self.register_buffer("slot_offsets", torch.arange(spec.context) * spec.num_actions, persistent=False)

        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        skips: list[int] = []
        c_prev = ch[0]
        for i, c in enumerate(ch):
            level = Level(c_prev, c, spec.blocks, spec.emb_dim, i in spec.attn_levels)
            self.down.append(level)
            skips.extend([c] * spec.blocks)
            c_prev = c
            if i < len(ch) - 1:
                self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
        self.mid1 = ResBlock(c_prev, c_prev, spec.emb_dim)
        self.mid_attn = SelfAttention(c_prev)
        self.mid2 = ResBlock(c_prev, c_prev, spec.emb_dim)

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in reversed(range(len(ch))):
            c = ch[i]
            skip_ch = tuple(skips.pop() for _ in range(spec.blocks))
            self.up.append(Level(c_prev, c, spec.blocks, spec.emb_dim, i in spec.attn_levels, skip_ch))
            c_prev = c
            if i > 0:
                self.upsample.append(nn.Conv2d(c, ch[i - 1], 3, padding=1))
                c_prev = ch[i - 1]
        self.out_norm = group_norm(ch[0])
        self.out = nn.Conv2d(ch[0], spec.image_channels * 4, 3, padding=1)
        zero_init(self.out)

    def embed(self, c_noise: Tensor, ctx_sigma: Tensor, actions: Tensor) -> Tensor:
        """Conditioning vector ``[B, emb_dim]``."""
        act = self.action_emb(actions.long() + self.slot_offsets[None, :]).sum(dim=1)
        emb: Tensor = self.noise_emb(c_noise) + self.ctx_noise_emb(ctx_sigma) + act
        return emb

    def forward(self, x: Tensor, c_noise: Tensor, ctx: Tensor, actions: Tensor, ctx_sigma: Tensor) -> Tensor:
        """``x`` [B,3,H,W] (already scaled by c_in), ``ctx`` [B,3*context,H,W], ``actions`` [B,context]."""
        emb = self.embed(c_noise, ctx_sigma, actions)
        h = self.stem(F.pixel_unshuffle(torch.cat([x, ctx], dim=1), 2))
        hs: list[Tensor] = []
        for i, level in enumerate(self.down):
            assert isinstance(level, Level)
            for block, attn in zip(level.blocks, level.attns, strict=True):
                h = attn(block(h, emb))
                hs.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for i, level in enumerate(self.up):
            assert isinstance(level, Level)
            for block, attn in zip(level.blocks, level.attns, strict=True):
                h = attn(block(torch.cat([h, hs.pop()], dim=1), emb))
            if i < len(self.upsample):
                h = self.upsample[i](F.interpolate(h, scale_factor=2.0, mode="nearest"))
        out: Tensor = F.pixel_shuffle(self.out(F.silu(self.out_norm(h))), 2)
        return out


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
