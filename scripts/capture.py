"""Record the real browser player with Playwright: README GIFs and an FPS measurement.

The page runs in attract mode (scripted inputs fed to both the real game and the dream). We
screenshot the stage after every completed tick, so the GIF plays at the game's tick rate
no matter how fast inference ran during capture; the measured FPS is reported separately.

    uv run --group capture python scripts/capture.py --game dungeon --frames 150 \
        --gif docs/assets/dungeon_split.gif --stats docs/results/browser_dungeon.json
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import Browser, Page, sync_playwright

TICK_MS = {"dungeon": 1000 / 15, "breakout": 1000 / 30}
# Opt-in: on a headless server these flags usually yield a *software* (SwiftShader) WebGPU
# adapter, which is far slower than WASM, so captures default to the browser's own choice.
WEBGPU_FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan"]
# Collected after the speed run so the stats file records what actually executed the model.
PAGE_INFO_JS = """async () => {
    const describe = (a) => (a ? `${a.info?.vendor} ${a.info?.architecture} ${a.info?.description}` : null);
    return {
        backend: window.dreamcade.backend,
        userAgent: navigator.userAgent,
        webgpuAdapter: navigator.gpu
            ? await navigator.gpu.requestAdapter().then(describe).catch(() => null)
            : null,
        crossOriginIsolated: crossOriginIsolated,
        hardwareConcurrency: navigator.hardwareConcurrency,
    };
}"""


def _open(browser: Browser, args: argparse.Namespace, query: str) -> Page:
    page = browser.new_page(viewport={"width": args.width, "height": 760}, device_scale_factor=1)
    page.goto(args.url + query)
    page.wait_for_function(
        "window.dreamcade && (window.dreamcade.ready || window.dreamcade.error)", timeout=300_000
    )
    error = page.evaluate("window.dreamcade.error")
    if error:
        msg = f"player failed to start: {error}"
        raise SystemExit(msg)
    return page


def record(page: Page, n: int, scale: float) -> list[Image.Image]:
    """Frame-step mode: advance one tick, wait for it to render, screenshot the stage."""
    stage = page.locator(".stage")
    frames: list[Image.Image] = []
    last = page.evaluate("window.dreamcade.frame")
    while len(frames) < n:
        page.evaluate("window.dreamcade.next()")
        page.wait_for_function(f"window.dreamcade.frame > {last}", timeout=120_000)
        last = page.evaluate("window.dreamcade.frame")
        img = Image.open(io.BytesIO(stage.screenshot(animations="disabled"))).convert("RGB")
        if scale != 1.0:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
        frames.append(img)
    return frames


def gif_durations(n: int, tick_ms: float) -> list[int]:
    """Per-frame GIF delays (ms) that keep the animation on the game's tick schedule.

    GIF stores delays in whole centiseconds and Pillow truncates, so a constant 67 ms delay
    would play every frame at 60 ms (16.7 Hz instead of 15 Hz). Ending each frame on the
    centisecond nearest its true end time (70, 60, 70, 70, 60, ... ms) keeps the mean exact.
    """
    ends = [round((i + 1) * tick_ms / 10) * 10 for i in range(n)]
    return [end - start for start, end in zip([0, *ends[:-1]], ends, strict=True)]


def measure(page: Page, ticks: int) -> dict[str, float]:
    """Free-running mode: let the player run and read its own FPS and per-frame model time."""
    page.wait_for_function(f"window.dreamcade.frame >= {ticks}", timeout=600_000)
    status = page.evaluate("window.dreamcade")
    return {"dream_fps": round(status["fps"], 2), "model_ms_per_frame": round(status["modelMs"], 1)}


def capture(args: argparse.Namespace) -> None:
    """Record the GIF in frame-step mode, then measure real-time speed free-running."""
    query = f"?game={args.game}&view={args.view}&attract=1&steps={args.steps}"
    if args.backend:
        query += f"&backend={args.backend}"
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=WEBGPU_FLAGS if args.webgpu else [])
        frames = (
            record(_open(browser, args, query + "&step=manual"), args.frames, args.scale) if args.gif else []
        )
        page = _open(browser, args, query)
        speed = measure(page, args.measure_ticks)
        info = page.evaluate(PAGE_INFO_JS)
        browser.close()
    if args.gif:
        quantized = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
        quantized[0].save(
            args.gif,
            save_all=True,
            append_images=quantized[1:],
            duration=gif_durations(len(quantized), TICK_MS[args.game]),
            loop=0,
            optimize=True,
        )
        print(f"wrote {args.gif} ({Path(args.gif).stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")
    stats = {
        "game": args.game,
        "diffusion_steps": args.steps if args.game == "dungeon" else None,
        "gif_frames": len(frames),
        "measured_over_ticks": args.measure_ticks,
        **speed,
        **info,
    }
    if args.stats:
        Path(args.stats).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main() -> None:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8095/")
    p.add_argument("--game", choices=["dungeon", "breakout"], default="dungeon")
    p.add_argument("--view", default="split")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--backend", choices=["webgpu", "wasm"], default=None)
    p.add_argument("--webgpu", action="store_true", help="launch Chromium with WebGPU flags")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--width", type=int, default=1000)
    p.add_argument("--scale", type=float, default=0.75)
    p.add_argument("--gif", default=None)
    p.add_argument("--stats", default=None)
    p.add_argument("--measure-ticks", type=int, default=40)
    capture(p.parse_args())


if __name__ == "__main__":
    main()
