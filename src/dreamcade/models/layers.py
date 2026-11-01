"""U-Net building blocks shared by both world models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def zero_init(conv: nn.Conv2d) -> None:
    """Zero a layer so its residual branch starts as the identity (stable early training)."""
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with as many groups as divide ``channels`` (up to 32)."""
    groups = math.gcd(32, channels)
    return nn.GroupNorm(groups, channels, affine=False)


class FourierEmbedding(nn.Module):
    """Sinusoidal features of a scalar (noise level, context-noise level) followed by an MLP."""

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.freqs: Tensor
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, value: Tensor) -> Tensor:
        """``value``: ``[B]`` -> ``[B, dim]``."""
        args = value.float()[:, None] * self.freqs[None, :] * 1000.0
        out: Tensor = self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=1))
        return out


class ResBlock(nn.Module):
    """Pre-norm residual block whose norms are modulated by the conditioning vector (AdaGN)."""

    def __init__(self, c_in: int, c_out: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = group_norm(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.ada1 = nn.Linear(emb_dim, 2 * c_in)
        self.norm2 = group_norm(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.ada2 = nn.Linear(emb_dim, 2 * c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        zero_init(self.conv2)

    @staticmethod
    def _modulate(x: Tensor, ada: Tensor) -> Tensor:
        scale, shift = ada[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Apply the block; ``emb`` is ``[B, emb_dim]``."""
        h = self.conv1(F.silu(self._modulate(self.norm1(x), self.ada1(emb))))
        h = self.conv2(F.silu(self._modulate(self.norm2(h), self.ada2(emb))))
        out: Tensor = self.skip(x) + h
        return out


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions.

    Written with explicit matmuls instead of ``scaled_dot_product_attention`` so the ONNX
    graph uses only ops every onnxruntime-web backend implements.
    """

    def __init__(self, channels: int, head_dim: int = 64) -> None:
        super().__init__()
        self.heads = max(1, channels // head_dim)
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        zero_init(self.proj)

    def forward(self, x: Tensor) -> Tensor:
        """Residual attention."""
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, heads, d, N]
        scale = 1.0 / math.sqrt(c // self.heads)
        attn = torch.softmax(torch.matmul(q.transpose(-1, -2), k) * scale, dim=-1)  # [B, heads, N, N]
        out = torch.matmul(v, attn.transpose(-1, -2)).reshape(b, c, h, w)
        result: Tensor = x + self.proj(out)
        return result
