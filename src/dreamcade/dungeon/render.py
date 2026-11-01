"""Vectorised Wolfenstein-style raycaster: textured walls, a fogged checker floor, orbs.

The JavaScript port (``web/js/dungeon/render.js``) performs the same IEEE-754 double
operations in the same order, so frames match pixel-for-pixel. That is why expressions here
are written out step by step (``a * b * c`` is always ``(a * b) * c`` in both languages) and
every float-to-int conversion is an explicit ``floor``.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from dreamcade.dungeon.engine import DungeonState
from dreamcade.dungeon.world import (
    CEILING_RGB,
    FIXED,
    ORB_RADIUS,
    SIZE,
    TEX,
    TRIG_ONE,
    WORLD,
    World,
)

WIDTH: Final = 64
HEIGHT: Final = 64
PLANE: Final = 0.66  # camera plane half-width: about 66 degrees horizontal field of view
FOCAL: Final = WIDTH / (2 * PLANE)  # pixels per world unit at distance 1 (square pixels)
FOG: Final = 0.02
SIDE_SHADE: Final = 0.8
CHECKER_DARK: Final = 0.75
ORB_HEIGHT: Final = 0.45  # orb centre above the floor; the eye is at 0.5
ORB_CORE: Final = 0.6
NO_DIR: Final = 1e30  # stands in for 1/0 so both languages avoid infinities

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
Frames = npt.NDArray[np.uint8]


def _camera(state: DungeonState, world: World) -> tuple[F64, F64, F64, F64]:
    pos_x = state.x / FIXED
    pos_y = state.y / FIXED
    dir_x = world.cos[state.a] / TRIG_ONE
    dir_y = world.sin[state.a] / TRIG_ONE
    return pos_x, pos_y, dir_x, dir_y


def _cast(world: World, pos_x: F64, pos_y: F64, ray_x: F64, ray_y: F64) -> tuple[F64, I64, I64, I64]:
    """DDA over the grid for ``[B, W]`` rays. Returns perp distance, side, map x, map y."""
    map_x = np.floor(pos_x).astype(np.int64)
    map_y = np.floor(pos_y).astype(np.int64)
    delta_x = np.where(ray_x == 0, NO_DIR, np.abs(1 / np.where(ray_x == 0, 1.0, ray_x)))
    delta_y = np.where(ray_y == 0, NO_DIR, np.abs(1 / np.where(ray_y == 0, 1.0, ray_y)))
    step_x = np.where(ray_x < 0, -1, 1)
    step_y = np.where(ray_y < 0, -1, 1)
    side_x = np.where(ray_x < 0, (pos_x - map_x) * delta_x, (map_x + 1.0 - pos_x) * delta_x)
    side_y = np.where(ray_y < 0, (pos_y - map_y) * delta_y, (map_y + 1.0 - pos_y) * delta_y)
    side = np.zeros(ray_x.shape, dtype=np.int64)
    active = np.ones(ray_x.shape, dtype=np.bool_)
    for _ in range(4 * SIZE):
        go_x = active & (side_x < side_y)
        go_y = active & ~(side_x < side_y)
        side_x = np.where(go_x, side_x + delta_x, side_x)
        map_x = np.where(go_x, map_x + step_x, map_x)
        side_y = np.where(go_y, side_y + delta_y, side_y)
        map_y = np.where(go_y, map_y + step_y, map_y)
        side = np.where(go_x, 0, np.where(go_y, 1, side))
        active &= world.walls[map_y, map_x] == 0
        if not active.any():
            break
    else:
        msg = "ray escaped the map: the outer wall ring is broken"
        raise RuntimeError(msg)
    perp = np.where(side == 0, side_x - delta_x, side_y - delta_y)
    return perp, side, map_x, map_y


def render(state: DungeonState, world: World = WORLD) -> Frames:
    """Render a batch of states to ``uint8[B, HEIGHT, WIDTH, 3]`` RGB frames."""
    pos_x, pos_y, dir_x, dir_y = _camera(state, world)
    batch = pos_x.shape[0]
    plane_x = -dir_y * PLANE
    plane_y = dir_x * PLANE

    cam = (2.0 * np.arange(WIDTH) + 1.0) / WIDTH - 1.0  # [W]
    px, py = pos_x[:, None], pos_y[:, None]
    ray_x = dir_x[:, None] + plane_x[:, None] * cam[None, :]  # [B, W]
    ray_y = dir_y[:, None] + plane_y[:, None] * cam[None, :]
    perp, side, map_x, map_y = _cast(world, px, py, ray_x, ray_y)

    # Wall texture column.
    tex_id = world.walls[map_y, map_x].astype(np.int64) - 1
    wall_x = np.where(side == 0, py + perp * ray_y, px + perp * ray_x)
    wall_x = wall_x - np.floor(wall_x)
    tex_x = np.floor(wall_x * TEX).astype(np.int64)
    flip = ((side == 0) & (ray_x > 0)) | ((side == 1) & (ray_y < 0))
    tex_x = np.where(flip, TEX - 1 - tex_x, tex_x)
    line_h = FOCAL / perp
    top = HEIGHT / 2 - line_h / 2
    wall_shade = np.where(side == 1, SIDE_SHADE, 1.0) / (1 + perp * perp * FOG)

    rows = np.arange(HEIGHT, dtype=np.float64) + 0.5  # pixel centres [H]
    v = (rows[None, :, None] - top[:, None, :]) / line_h[:, None, :]  # [B, H, W]
    is_wall = (v >= 0) & (v < 1)
    tex_y = np.minimum(np.floor(np.where(is_wall, v, 0.0) * TEX).astype(np.int64), TEX - 1)
    texel = world.textures[tex_id[:, None, :], tex_y, tex_x[:, None, :]]  # [B, H, W, 3]
    wall_rgb = np.floor(texel * wall_shade[:, None, :, None])

    # Floor and ceiling: distance to the floor point seen by each row.
    horizon = rows - HEIGHT / 2  # negative above the horizon
    row_dist = (FOCAL * 0.5) / np.abs(horizon)  # [H]
    row_fog = 1 / (1 + row_dist * row_dist * FOG)
    fx = px[:, None, :] + row_dist[None, :, None] * ray_x[:, None, :]  # [B, H, W]
    fy = py[:, None, :] + row_dist[None, :, None] * ray_y[:, None, :]
    cell_x = np.clip(np.floor(fx).astype(np.int64), 0, SIZE - 1)
    cell_y = np.clip(np.floor(fy).astype(np.int64), 0, SIZE - 1)
    checker = (np.floor(fx * 2).astype(np.int64) + np.floor(fy * 2).astype(np.int64)) & 1
    tint = world.floor_tint[world.rooms[cell_y, cell_x]]  # [B, H, W, 3]
    floor_rgb = np.floor(
        tint * np.where(checker == 1, CHECKER_DARK, 1.0)[..., None] * row_fog[None, :, None, None]
    )
    ceil_rgb = np.floor(np.array(CEILING_RGB, dtype=np.float64) * row_fog[:, None])  # [H, 3]
    below = (horizon > 0)[None, :, None, None]
    flat_rgb = np.where(below, floor_rgb, np.broadcast_to(ceil_rgb[None, :, None, :], floor_rgb.shape))
    out = np.where(is_wall[..., None], wall_rgb, flat_rgb)

    # Orbs: depth-tested spheres. The z-buffer starts at each column's wall distance.
    zbuf = np.broadcast_to(perp[:, None, :], (batch, HEIGHT, WIDTH)).copy()
    cols = np.arange(WIDTH, dtype=np.float64) + 0.5
    for (ox, oy), rgb in zip(world.orbs_xy, world.orbs_rgb, strict=True):
        sx = ox / FIXED - pos_x
        sy = oy / FIXED - pos_y
        depth = dir_x * sx + dir_y * sy
        lateral = dir_x * sy - dir_y * sx
        safe = np.where(depth > 0.05, depth, 1.0)
        cx = WIDTH / 2 * (1 + lateral / (PLANE * safe))
        cy = HEIGHT / 2 + (0.5 - ORB_HEIGHT) * FOCAL / safe
        radius = ORB_RADIUS * FOCAL / safe
        r2 = radius * radius
        ddx = cols[None, None, :] - cx[:, None, None]
        ddy = rows[None, :, None] - cy[:, None, None]
        d2 = ddx * ddx + ddy * ddy
        hit = (depth[:, None, None] > 0.05) & (d2 < r2[:, None, None]) & (depth[:, None, None] < zbuf)
        k = (1 - ORB_CORE * d2 / r2[:, None, None]) / (1 + depth * depth * FOG)[:, None, None]
        orb_rgb = np.floor(rgb[None, None, None, :] * k[..., None])
        out = np.where(hit[..., None], orb_rgb, out)
        zbuf = np.where(hit, depth[:, None, None], zbuf)
    return out.astype(np.uint8)
