"""Action-conditioned next-frame model for Breakout: per-pixel palette classification.

Breakout is deterministic given two frames and an action (see ``dreamcade.game.engine``),
so a single forward pass that classifies every pixel into one of 8 palette colours is the
right model: no sampling, no blur, and a crisp frame every step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dreamcade.game.constants import NUM_ACTIONS, NUM_COLORS
from dreamcade.models.layers import ResBlock, SelfAttention, group_norm


@dataclass(frozen=True, slots=True)
class BreakoutSpec:
    """Architecture hyper-parameters (stored in checkpoints)."""

    context: int = 4
    channels: tuple[int, ...] = (48, 96, 192)
    emb_dim: int = 128


class BreakoutNet(nn.Module):
    """U-Net from ``context`` palette frames + action to next-frame logits ``[B, 8, 64, 64]``."""

    def __init__(self, spec: BreakoutSpec) -> None:
        super().__init__()
        self.spec = spec
        ch = spec.channels
        self.palette: Tensor
        self.register_buffer("palette", torch.arange(NUM_COLORS, dtype=torch.float32), persistent=False)
        self.action_emb = nn.Sequential(
            nn.Embedding(NUM_ACTIONS, spec.emb_dim), nn.SiLU(), nn.Linear(spec.emb_dim, spec.emb_dim)
        )
        self.stem = nn.Conv2d(spec.context * NUM_COLORS, ch[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        c_prev = ch[0]
        for c in ch:
            self.down.append(ResBlock(c_prev, c, spec.emb_dim))
            self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
            c_prev = c
        self.mid1 = ResBlock(c_prev, c_prev, spec.emb_dim)
        self.mid_attn = SelfAttention(c_prev)
        self.mid2 = ResBlock(c_prev, c_prev, spec.emb_dim)
        self.up = nn.ModuleList()
        for c in reversed(ch):
            self.up.append(ResBlock(c_prev + c, c, spec.emb_dim))
            c_prev = c
        self.out_norm = group_norm(c_prev)
        self.out = nn.Conv2d(c_prev, NUM_COLORS, 3, padding=1)

    def one_hot(self, frames: Tensor) -> Tensor:
        """Palette indices ``[B, K, H, W]`` (any dtype) -> one-hot ``[B, K*8, H, W]``."""
        b, k, h, w = frames.shape
        hot = (frames.float()[:, :, None] == self.palette[None, None, :, None, None]).float()
        return hot.reshape(b, k * NUM_COLORS, h, w)

    def _encode(self, frames: Tensor, action: Tensor) -> tuple[Tensor, list[Tensor], Tensor]:
        emb = self.action_emb(action.long())
        h = self.stem(self.one_hot(frames))
        skips: list[Tensor] = []
        for block, down in zip(self.down, self.downsample, strict=True):
            h = block(h, emb)
            skips.append(h)
            h = down(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        return h, skips, emb

    def features(self, frames: Tensor, action: Tensor) -> Tensor:
        """Bottleneck activations ``[B, C, 8, 8]`` (used by the linear probes)."""
        return self._encode(frames, action)[0]

    def forward(self, frames: Tensor, action: Tensor) -> Tensor:
        """Next-frame logits."""
        h, skips, emb = self._encode(frames, action)
        for block in self.up:
            h = F.interpolate(h, scale_factor=2.0, mode="nearest")
            h = block(torch.cat([h, skips.pop()], dim=1), emb)
        logits: Tensor = self.out(F.silu(self.out_norm(h)))
        return logits
