"""Export trained world models to ONNX for the browser, with a parity check against PyTorch.

Weights are stored as float16 and cast to float32 inside the graph ("fp16 storage, fp32
compute"): the download halves, every onnxruntime-web backend (including WASM, which has no
fp16 kernels) can run it, and the casts are constant-folded when the session is created.
GitHub rejects files over 100 MB and this repo keeps every file under 20 MB, so the model
bytes are split into parts that the player concatenates before creating the session.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import Tensor, nn

from dreamcade.config import BreakoutTrainConfig, DungeonTrainConfig, config_from_dict
from dreamcade.models.edm import EDMSchedule

PART_BYTES = 15 * 1024 * 1024
MIN_FP16_ELEMENTS = 256


class DungeonGraph(nn.Module):
    """``D(x; sigma | ctx, actions, ctx_sigma)`` with batch size 1 tensors."""

    def __init__(self, denoiser: nn.Module) -> None:
        super().__init__()
        self.denoiser = denoiser

    def forward(self, x: Tensor, sigma: Tensor, ctx: Tensor, actions: Tensor, ctx_sigma: Tensor) -> Tensor:
        """Denoised frame in ``[-1, 1]`` space (not clamped; the sampler clamps the last step)."""
        out: Tensor = self.denoiser(x, sigma, ctx, actions, ctx_sigma)
        return out


def _load(run: Path) -> dict[str, Any]:
    path = run / "ema.pt"
    if not path.exists():
        msg = f"{path} not found; train the model first (dreamcade train ...)"
        raise FileNotFoundError(msg)
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def _round_weights_fp16(model: nn.Module) -> None:
    """Make the PyTorch reference use exactly the weights the ONNX file stores."""
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() >= MIN_FP16_ELEMENTS:
                p.copy_(p.half().float())


def fp16_storage(model: onnx.ModelProto) -> onnx.ModelProto:
    """Store large float initializers as float16 and insert Cast nodes back to float32."""
    graph = model.graph
    casts = []
    keep = []
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and arr.size >= MIN_FP16_ELEMENTS:
            half = numpy_helper.from_array(arr.astype(np.float16), name=f"{init.name}__fp16")
            keep.append(half)
            casts.append(
                helper.make_node(
                    "Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=f"{init.name}__cast"
                )
            )
        else:
            keep.append(init)
    del graph.initializer[:]
    graph.initializer.extend(keep)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    return model


def _export_onnx(module: nn.Module, args: tuple[Tensor, ...], names: list[str], path: Path) -> None:
    module.eval()
    torch.onnx.export(
        module,
        args,
        str(path),
        input_names=names,
        output_names=["out"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    model = fp16_storage(onnx.load(str(path)))
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _split(path: Path, out: Path, name: str) -> dict[str, Any]:
    data = path.read_bytes()
    parts = []
    for i in range(0, len(data), PART_BYTES):
        part = out / f"{name}.onnx.part{i // PART_BYTES}"
        part.write_bytes(data[i : i + PART_BYTES])
        parts.append(part.name)
    return {"parts": parts, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _parity(
    path: Path, module: nn.Module, feeds: dict[str, np.ndarray], args: tuple[Tensor, ...]
) -> dict[str, float]:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)[0]
    with torch.no_grad():
        want = module(*args).numpy()
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(None, feeds)
    ms = (time.perf_counter() - t0) / 5 * 1000
    if np.issubdtype(got.dtype, np.integer):
        return {"pixel_mismatch": float((got != want).mean()), "ort_cpu_ms": round(ms, 2)}
    return {
        "max_abs_diff": float(np.abs(got - want).max()),
        "mean_abs_diff": float(np.abs(got - want).mean()),
        "ort_cpu_ms": round(ms, 2),
    }


def export_dungeon(run: Path, out: Path) -> dict[str, Any]:
    """Export the dungeon denoiser; returns its manifest entry."""
    from dreamcade.train.dungeon import build_denoiser

    ckpt = _load(run)
    cfg = config_from_dict(DungeonTrainConfig, ckpt["config"])
    denoiser = build_denoiser(cfg)
    denoiser.load_state_dict(ckpt["ema"])
    _round_weights_fp16(denoiser)
    graph = DungeonGraph(denoiser).eval()
    g = torch.Generator().manual_seed(0)
    n = cfg.context
    args = (
        torch.randn(1, 3, 64, 64, generator=g),
        torch.tensor([1.3]),
        torch.rand(1, 3 * n, 64, 64, generator=g) * 2 - 1,
        torch.randint(0, 7, (1, n), generator=g),
        torch.tensor([0.05]),
    )
    names = ["x", "sigma", "ctx", "actions", "ctx_sigma"]
    tmp = out / "dungeon.onnx"
    _export_onnx(graph, args, names, tmp)
    parity = _parity(tmp, graph, {k: v.numpy() for k, v in zip(names, args, strict=True)}, args)
    entry = _split(tmp, out, "dungeon")
    tmp.unlink()
    s = EDMSchedule()
    entry |= {
        "context": n,
        "sigmaMin": s.sigma_min,
        "sigmaMax": s.sigma_max,
        "rho": s.rho,
        "defaultSteps": 3,
        "defaultCtxSigma": 0.05,
        "trainedSteps": int(ckpt["step"]),
        "parity": parity,
    }
    return entry


class BreakoutGraph(nn.Module):
    """Palette frames ``[1, K, 64, 64]`` (float) + action ``[1]`` -> next-frame indices ``[1, 64, 64]``."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, frames: Tensor, action: Tensor) -> Tensor:
        """Arg-max is done in the graph so the browser receives palette indices directly."""
        logits: Tensor = self.net(frames, action)
        return logits.argmax(dim=1).to(torch.int32)


def export_breakout(run: Path, out: Path) -> dict[str, Any]:
    """Export the Breakout classifier; returns its manifest entry."""
    from dreamcade.train.breakout import build_model

    ckpt = _load(run)
    cfg = config_from_dict(BreakoutTrainConfig, ckpt["config"])
    net = build_model(cfg)
    net.load_state_dict(ckpt["ema"])
    _round_weights_fp16(net)
    graph = BreakoutGraph(net).eval()
    g = torch.Generator().manual_seed(0)
    args = (torch.randint(0, 8, (1, cfg.context, 64, 64), generator=g).float(), torch.tensor([2]))
    names = ["frames", "action"]
    tmp = out / "breakout.onnx"
    _export_onnx(graph, args, names, tmp)
    parity = _parity(tmp, graph, {k: v.numpy() for k, v in zip(names, args, strict=True)}, args)
    entry = _split(tmp, out, "breakout")
    tmp.unlink()
    return entry | {"context": cfg.context, "trainedSteps": int(ckpt["step"]), "parity": parity}


def export(game: str, run: Path, out: Path) -> None:
    """Export one game and update ``manifest.json`` next to the parts."""
    out.mkdir(parents=True, exist_ok=True)
    entry = export_dungeon(run, out) if game == "dungeon" else export_breakout(run, out)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[game] = entry
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({game: entry["parity"]}, indent=2))
