"""Model shapes, EDM preconditioning and the sampler on a toy distribution."""

from __future__ import annotations

import itertools
import math

import torch

from dreamcade.models.breakout_net import BreakoutNet, BreakoutSpec
from dreamcade.models.edm import Denoiser, EDMSchedule, euler_sample, heun_sample, karras_sigmas, precondition
from dreamcade.models.unet import DungeonUNet, UNetSpec

TINY = UNetSpec(context=2, channels=(32, 64), attn_levels=(1,), emb_dim=64)


def test_unet_forward_shape() -> None:
    net = DungeonUNet(TINY)
    x = torch.randn(2, 3, 64, 64)
    out = net(x, torch.zeros(2), torch.randn(2, 6, 64, 64), torch.randint(0, 7, (2, 2)), torch.zeros(2))
    assert out.shape == (2, 3, 64, 64)


def test_denoiser_is_identity_at_tiny_sigma() -> None:
    """With a zero-initialised output layer, D(x; sigma -> 0) -> x (c_skip -> 1)."""
    model = Denoiser(DungeonUNet(TINY), EDMSchedule())
    x = torch.randn(1, 3, 64, 64)
    out = model(
        x,
        torch.tensor([1e-4]),
        torch.zeros(1, 6, 64, 64),
        torch.zeros(1, 2, dtype=torch.long),
        torch.zeros(1),
    )
    assert torch.allclose(out, x, atol=1e-3)


def test_preconditioning_unit_variance() -> None:
    sigma = torch.tensor([0.01, 0.5, 5.0])
    c_skip, c_out, c_in, _ = precondition(sigma, 0.5)
    # The target (x0 - c_skip x) / c_out has unit variance when x0 ~ N(0, sd^2):
    # c_skip^2 s^2 + (1 - c_skip)^2 sd^2 = c_out^2.
    lhs = c_skip**2 * sigma**2 + (1 - c_skip) ** 2 * 0.25
    assert torch.allclose(lhs, c_out**2, rtol=1e-5)
    assert torch.allclose(c_in**2 * (sigma**2 + 0.25), torch.ones(3))


def test_breakout_net_shapes() -> None:
    net = BreakoutNet(BreakoutSpec(context=4, channels=(16, 32)))
    frames = torch.randint(0, 8, (3, 4, 64, 64))
    assert net(frames, torch.tensor([0, 1, 3])).shape == (3, 8, 64, 64)
    assert net.features(frames, torch.tensor([0, 1, 3])).shape == (3, 32, 16, 16)


def test_karras_sigmas() -> None:
    s = karras_sigmas(4, 0.002, 5.0, 7.0)
    assert len(s) == 5 and s[-1] == 0.0
    assert math.isclose(s[0], 5.0) and math.isclose(s[3], 0.002, rel_tol=1e-6)
    assert all(a > b for a, b in itertools.pairwise(s))


# --- sampler correctness on a 2-D Gaussian mixture with an exact denoiser ---------------

MEANS = torch.tensor([[-2.0, 0.0], [2.0, 1.0], [0.0, -2.5]])
WEIGHTS = torch.tensor([0.5, 0.3, 0.2])
STD = 0.3


def exact_denoiser(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """E[x0 | x] for the mixture: posterior responsibilities times per-component Wiener filters."""
    s2 = sigma[:, None] ** 2
    var = STD**2 + s2  # [B, 1]
    d2 = ((x[:, None, :] - MEANS[None]) ** 2).sum(-1)  # [B, K]
    logp = (
        torch.log(WEIGHTS)[None] - d2 / (2 * var) - torch.log(var)
    )  # 2-D Gaussian log-density (up to const)
    resp = torch.softmax(logp, dim=1)
    shrink = (STD**2 / var)[:, :, None]  # [B, 1, 1]
    comp = MEANS[None] + shrink * (x[:, None, :] - MEANS[None])
    return (resp[:, :, None] * comp).sum(1)


def _check_samples(samples: torch.Tensor) -> None:
    assign = ((samples[:, None, :] - MEANS[None]) ** 2).sum(-1).argmin(1)
    freq = torch.bincount(assign, minlength=3).float() / len(samples)
    assert torch.allclose(freq, WEIGHTS, atol=0.03), freq
    for k in range(3):
        pts = samples[assign == k]
        assert torch.allclose(pts.mean(0), MEANS[k], atol=0.06)
        assert abs(float(pts.std(0).mean()) - STD) < 0.05


def test_euler_sampler_recovers_mixture() -> None:
    torch.manual_seed(0)
    noise = torch.randn(8000, 2)
    sigmas = karras_sigmas(80, 0.002, 40.0, 7.0)
    _check_samples(euler_sample(exact_denoiser, noise, sigmas))


def test_heun_sampler_recovers_mixture() -> None:
    torch.manual_seed(1)
    noise = torch.randn(8000, 2)
    _check_samples(heun_sample(exact_denoiser, noise, karras_sigmas(30, 0.002, 40.0, 7.0)))
