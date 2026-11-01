"""Figures for docs/results (matplotlib, headless)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from PIL import Image
from torch import Tensor

COLORS = ("#7c3aed", "#0891b2", "#e11d48", "#ea580c", "#16a34a", "#64748b")


def _style(ax: Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def line_plot(
    path: Path,
    series: dict[str, tuple[Sequence[float], Sequence[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
    logy: bool = False,
) -> None:
    """One line per series, direct-labelled legend."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    for (name, (x, y)), color in zip(series.items(), COLORS, strict=False):
        ax.plot(x, y, label=name, color=color, linewidth=1.8)
    if logy:
        ax.set_yscale("log")
    _style(ax, xlabel, ylabel, title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def bar_plot(path: Path, groups: dict[str, dict[str, float]], ylabel: str, title: str) -> None:
    """Grouped bars: ``groups[label][series] = value``."""
    labels = list(groups)
    series = list(next(iter(groups.values())))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    for i, (name, color) in enumerate(zip(series, COLORS, strict=False)):
        xs = np.arange(len(labels)) + (i - (len(series) - 1) / 2) * width
        ax.bar(xs, [groups[g][name] for g in labels], width, label=name, color=color)
    ax.set_xticks(np.arange(len(labels)), labels)
    _style(ax, "", ylabel, title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _to_uint8(img: Tensor) -> np.ndarray:
    arr = ((img.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    return arr.permute(1, 2, 0).numpy()


def save_strip(
    path: Path, rows: list[list[Tensor]], col_labels: list[str], row_labels: list[str], scale: int = 3
) -> None:
    """A grid of 64x64 frames (``[3, H, W]`` in ``[-1, 1]``) with labels."""
    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(1.25 * len(rows[0]), 1.3 * len(rows)), dpi=110)
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r, c]
            ax.imshow(
                Image.fromarray(_to_uint8(img)).resize((64 * scale, 64 * scale), Image.Resampling.NEAREST)
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_labels[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def training_curves(metrics: Path, path: Path, keys: Sequence[str], title: str) -> None:
    """Plot JSONL metrics (one panel per key, smoothed with a running mean)."""
    records = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.0), dpi=150)
    for ax, key, color in zip(np.atleast_1d(axes), keys, COLORS, strict=False):
        pts = [(r["step"], r[key]) for r in records if key in r]
        if not pts:
            msg = f"{metrics} has no '{key}' records"
            raise ValueError(msg)
        x, y = map(np.array, zip(*pts, strict=True))
        if len(y) > 20:
            k = max(1, len(y) // 40)
            y = np.convolve(y, np.ones(k) / k, mode="same")
        ax.plot(x, y, color=color, linewidth=1.6)
        _style(ax, "step", key, "")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
