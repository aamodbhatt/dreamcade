"""Training infrastructure shared by both games: data windows, EMA, metrics, checkpoints."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from torch import Tensor, nn

from dreamcade.data.shards import list_shards, read_shard


def pick_device(name: str) -> torch.device:
    """``auto`` -> CUDA when available, else CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> torch.Generator:
    """Seed numpy/torch and return a dedicated generator for data sampling."""
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


@dataclass(slots=True)
class EpisodeStore:
    """All episodes as one tensor, optionally resident in GPU memory as uint8.

    Keeping the frames on the GPU as uint8 removes the data loader entirely: a training
    batch is a single gather, so the GPU never waits on 8 shared CPU cores.
    """

    frames: Tensor  # uint8 [E, T+1, ...]
    actions: Tensor  # int64 [E, T]
    states: Tensor  # int64 [E, T+1, S]

    @classmethod
    def load(cls, root: Path, device: torch.device, holdout: int, split: str) -> EpisodeStore:
        """Load every shard; ``split`` is ``train`` (all but the last ``holdout`` episodes) or ``holdout``."""
        frames, actions, states = [], [], []
        for path in list_shards(root):
            shard = read_shard(path)
            # Copy shard by shard so host memory never holds more than one shard at a time.
            frames.append(torch.from_numpy(np.array(shard.frames)).to(device))
            actions.append(torch.from_numpy(np.array(shard.actions, dtype=np.int64)).to(device))
            states.append(torch.from_numpy(np.array(shard.states, dtype=np.int64)).to(device))
        f, a, s = torch.cat(frames), torch.cat(actions), torch.cat(states)
        if holdout >= f.shape[0]:
            msg = f"holdout ({holdout}) must be smaller than the number of episodes ({f.shape[0]})"
            raise ValueError(msg)
        part = slice(None, f.shape[0] - holdout) if split == "train" else slice(f.shape[0] - holdout, None)
        return cls(f[part].contiguous(), a[part].contiguous(), s[part].contiguous())

    @property
    def episodes(self) -> int:
        """Number of episodes."""
        return int(self.frames.shape[0])

    @property
    def steps(self) -> int:
        """Actions per episode."""
        return int(self.actions.shape[1])

    def windows(self, n: int, length: int, generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
        """``n`` random windows: frames ``[n, length+1, ...]``, actions ``[n, length]``, states."""
        e = torch.randint(0, self.episodes, (n,), generator=generator)
        t = torch.randint(0, self.steps - length + 1, (n,), generator=generator)
        return self.window_at(e, t, length)

    def window_at(self, e: Tensor, t: Tensor, length: int) -> tuple[Tensor, Tensor, Tensor]:
        """Windows starting at explicit episode/time indices."""
        dev = self.frames.device
        e, t = e.to(dev), t.to(dev)
        idx = t[:, None] + torch.arange(length + 1, device=dev)[None, :]
        rows = e[:, None]
        return self.frames[rows, idx], self.actions[rows, idx[:, :length]], self.states[rows, idx]


class EMA[M: nn.Module]:
    """Exponential moving average of weights with the usual warm-up of the decay."""

    def __init__(self, model: M, decay: float) -> None:
        self.decay = decay
        self.model: M = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Blend the live weights into the average."""
        d = min(self.decay, (1 + step) / (10 + step))
        for ema_p, p in zip(self.model.parameters(), model.parameters(), strict=True):
            ema_p.lerp_(p.detach(), 1 - d)


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    """Linear warm-up then cosine decay to 10% of the base rate."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


class MetricsLog:
    """Append-only JSONL metrics (``{"step": ..., "train/loss": ..., "_time": ...}``)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = path.open("a", encoding="utf-8")

    def write(self, step: int, values: dict[str, float]) -> None:
        """Write one record and flush (so a crash loses at most one line)."""
        record: dict[str, Any] = {
            "step": step,
            **{k: round(v, 6) for k, v in values.items()},
            "_time": round(time.time(), 3),
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the file."""
        self._fh.close()


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write atomically: a killed job never leaves a truncated checkpoint behind."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def config_dict(cfg: object) -> dict[str, Any]:
    """Dataclass -> plain dict for checkpoints."""
    return asdict(cfg)  # type: ignore[call-overload,no-any-return]
