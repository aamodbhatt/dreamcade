"""EDM (Karras et al., 2022) preconditioning, training loss and few-step Euler sampling.

The denoiser ``D(x; sigma) = c_skip x + c_out F(c_in x, c_noise)`` keeps the network's
input and target at unit variance for every noise level. That is what makes 3-step sampling
usable: at the last step the model still predicts a clean frame rather than a noise residual.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dreamcade.models.unet import DungeonUNet


@dataclass(frozen=True, slots=True)
class EDMSchedule:
    """Noise-level distribution for training and the sampling schedule (DIAMOND defaults)."""

    sigma_data: float = 0.5
    p_mean: float = -0.4
    p_std: float = 1.2
    train_sigma_min: float = 2e-3
    train_sigma_max: float = 20.0
    sigma_min: float = 2e-3
    sigma_max: float = 5.0
    rho: float = 7.0


def precondition(sigma: Tensor, sigma_data: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``c_skip, c_out, c_in, c_noise`` for each noise level (shape ``[B]``)."""
    s2 = sigma * sigma
    d2 = sigma_data * sigma_data
    c_skip = d2 / (s2 + d2)
    c_out = sigma * sigma_data / torch.sqrt(s2 + d2)
    c_in = 1 / torch.sqrt(s2 + d2)
    c_noise = torch.log(sigma) / 4
    return c_skip, c_out, c_in, c_noise


class Denoiser(nn.Module):
    """Wraps the U-Net with EDM preconditioning: returns the denoised next frame."""

    def __init__(self, net: DungeonUNet, schedule: EDMSchedule) -> None:
        super().__init__()
        self.net = net
        self.schedule = schedule

    def forward(self, x: Tensor, sigma: Tensor, ctx: Tensor, actions: Tensor, ctx_sigma: Tensor) -> Tensor:
        """``D(x; sigma | ctx, actions)`` with every per-sample scalar of shape ``[B]``."""
        c_skip, c_out, c_in, c_noise = precondition(sigma, self.schedule.sigma_data)
        f: Tensor = self.net(x * c_in[:, None, None, None], c_noise, ctx, actions, ctx_sigma)
        out: Tensor = c_skip[:, None, None, None] * x + c_out[:, None, None, None] * f
        return out

    def loss(
        self, x0: Tensor, ctx: Tensor, actions: Tensor, ctx_sigma: Tensor, noise: Tensor, sigma: Tensor
    ) -> Tensor:
        """EDM loss in the network's output space (equivalent to the lambda(sigma)-weighted MSE)."""
        c_skip, c_out, c_in, c_noise = precondition(sigma, self.schedule.sigma_data)
        x = x0 + sigma[:, None, None, None] * noise
        f: Tensor = self.net(x * c_in[:, None, None, None], c_noise, ctx, actions, ctx_sigma)
        target = (x0 - c_skip[:, None, None, None] * x) / c_out[:, None, None, None]
        return ((f.float() - target) ** 2).mean()

    def sample_sigma(self, n: int, generator: torch.Generator | None, device: torch.device) -> Tensor:
        """Training noise levels: log-normal, clipped."""
        s = self.schedule
        z = torch.randn(n, generator=generator, device=device)
        return (s.p_mean + s.p_std * z).exp().clamp(s.train_sigma_min, s.train_sigma_max)


def karras_sigmas(steps: int, sigma_min: float, sigma_max: float, rho: float) -> list[float]:
    """``steps`` decreasing noise levels plus a final 0 (Karras et al., eq. 5)."""
    if steps < 1:
        msg = f"need at least one sampling step, got {steps}"
        raise ValueError(msg)
    if steps == 1:
        return [sigma_max, 0.0]
    lo, hi = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    sig = [(hi + i / (steps - 1) * (lo - hi)) ** rho for i in range(steps)]
    return [*sig, 0.0]


DenoiseFn = Callable[[Tensor, Tensor], Tensor]


def euler_sample(denoise: DenoiseFn, noise: Tensor, sigmas: list[float]) -> Tensor:
    """Deterministic Euler integration of the probability-flow ODE.

    ``denoise(x, sigma[B])`` returns the denoised estimate; ``noise`` is unit Gaussian. The
    browser player implements the identical loop in JavaScript.
    """
    x = noise * sigmas[0]
    for sigma, sigma_next in itertools.pairwise(sigmas):
        s = torch.full((x.shape[0],), sigma, device=x.device, dtype=x.dtype)
        denoised = denoise(x, s)
        # The last step lands exactly on the denoised estimate (sigma_next = 0).
        x = denoised if sigma_next == 0.0 else x + (x - denoised) / sigma * (sigma_next - sigma)
    return x


def heun_sample(denoise: DenoiseFn, noise: Tensor, sigmas: list[float]) -> Tensor:
    """Second-order Heun sampler (used in tests as an accuracy reference)."""
    x = noise * sigmas[0]
    for sigma, sigma_next in itertools.pairwise(sigmas):
        s = torch.full((x.shape[0],), sigma, device=x.device, dtype=x.dtype)
        d = (x - denoise(x, s)) / sigma
        x_next = x + d * (sigma_next - sigma)
        if sigma_next != 0.0:
            s2 = torch.full_like(s, sigma_next)
            d2 = (x_next - denoise(x_next, s2)) / sigma_next
            x_next = x + (d + d2) / 2 * (sigma_next - sigma)
        x = x_next
    return x


def psnr(a: Tensor, b: Tensor) -> Tensor:
    """Per-sample PSNR in dB for images in ``[-1, 1]``."""
    mse = ((a.float() - b.float()) ** 2).flatten(1).mean(dim=1).clamp_min(1e-10)
    return 10 * torch.log10(4.0 / mse)
