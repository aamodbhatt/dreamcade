"""The dungeon: a fixed tile map, procedural wall textures, floating orbs and a trig table.

Everything the renderer needs is plain integers so the Python engine and the JavaScript port
(``web/js/dungeon/``) agree pixel-for-pixel. The browser loads the exact same data from
``web/assets/dungeon_world.json``, written by :func:`world_to_json`; a test fails if that
file ever drifts from this module.

Layout (20x20 tiles): four themed rooms in the corners, a hub in the middle and corridors
that form loops, so exploration policies can come back to a room from a different side.
Every room has its own wall texture, floor tint, coloured door and orb: these landmarks
are what lets a world model recover *where* it is from a single frame.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

SIZE: Final = 20  # map is SIZE x SIZE tiles
TEX: Final = 16  # texture edge length in texels
FIXED: Final = 256  # fixed-point units per tile for positions
TRIG_ONE: Final = 16384  # fixed-point scale of the trig table
NUM_ANGLES: Final = 64  # heading resolution: 5.625 degrees per turn step

#: Inclusive interior rectangles (x0, y0, x1, y1) of the rooms; id 0 is "corridor".
ROOMS: Final[dict[int, tuple[int, int, int, int]]] = {
    1: (1, 1, 6, 6),  # red brick, north-west
    2: (13, 1, 18, 6),  # blue stone, north-east
    3: (1, 13, 6, 18),  # mossy green, south-west
    4: (13, 13, 18, 18),  # wood, south-east
    5: (8, 8, 11, 11),  # hub
}
ROOM_NAMES: Final[dict[int, str]] = {
    0: "corridor",
    1: "red hall",
    2: "blue vault",
    3: "moss cave",
    4: "wood cabin",
    5: "hub",
}
#: Corridors carved as inclusive floor rectangles; they connect rooms in two loops.
CORRIDORS: Final[tuple[tuple[int, int, int, int], ...]] = (
    (7, 3, 12, 3),  # red hall <-> blue vault
    (7, 16, 12, 16),  # moss cave <-> wood cabin
    (3, 7, 3, 12),  # red hall <-> moss cave
    (16, 7, 16, 12),  # blue vault <-> wood cabin
    (9, 4, 9, 7),  # north corridor <-> hub
    (12, 10, 15, 10),  # hub <-> east corridor
    (4, 9, 7, 9),  # west corridor <-> hub
)


class Tex:
    """Wall texture ids (map cell value - 1). Cell value 0 means floor."""

    STONE = 0  # corridor walls
    BRICK = 1  # red hall
    BLUE = 2  # blue vault
    MOSS = 3  # moss cave
    WOOD = 4  # wood cabin
    HUB = 5  # hub panels
    TORCH = 6
    DOOR_RED = 7
    DOOR_BLUE = 8
    DOOR_GREEN = 9
    DOOR_GOLD = 10


NUM_TEXTURES: Final = 11
ROOM_WALL_TEX: Final[dict[int, int]] = {
    1: Tex.BRICK,
    2: Tex.BLUE,
    3: Tex.MOSS,
    4: Tex.WOOD,
    5: Tex.HUB,
}
#: Decorative (closed) doors on each room's outer wall, and torches along corridors.
DOORS: Final[tuple[tuple[int, int, int], ...]] = (
    (3, 0, Tex.DOOR_RED),
    (16, 0, Tex.DOOR_BLUE),
    (3, 19, Tex.DOOR_GREEN),
    (16, 19, Tex.DOOR_GOLD),
)
TORCHES: Final[tuple[tuple[int, int], ...]] = (
    (8, 2),
    (11, 4),
    (8, 15),
    (11, 17),
    (2, 8),
    (4, 11),
    (15, 8),
    (17, 11),
    (8, 7),
    (12, 11),
)
#: Floor RGB tint per room id (corridor = 0). The checkerboard alternates tint and a darker copy.
FLOOR_TINT: Final[tuple[tuple[int, int, int], ...]] = (
    (92, 88, 84),
    (150, 80, 70),
    (70, 100, 160),
    (80, 140, 80),
    (150, 115, 70),
    (120, 110, 150),
)
CEILING_RGB: Final = (34, 32, 44)
#: Floating orbs: (x, y) in fixed point, RGB. One per room, placed off the main walking lines.
ORBS: Final[tuple[tuple[int, int, tuple[int, int, int]], ...]] = (
    (2 * FIXED + 128, 5 * FIXED + 128, (255, 90, 70)),
    (17 * FIXED + 128, 5 * FIXED + 128, (90, 170, 255)),
    (5 * FIXED + 128, 14 * FIXED + 128, (120, 255, 120)),
    (14 * FIXED + 128, 17 * FIXED + 128, (255, 210, 80)),
    (10 * FIXED, 10 * FIXED, (230, 120, 255)),
)
ORB_RADIUS: Final = 0.22  # tiles, for rendering
ORB_COLLIDE: Final = 96  # fixed-point collision radius (player radius + orb radius)


@dataclass(frozen=True, slots=True)
class World:
    """Immutable world data shared by the engine, renderer and exporters."""

    walls: npt.NDArray[np.int16]  # [SIZE, SIZE], 0 = floor, k = texture k-1
    rooms: npt.NDArray[np.int16]  # [SIZE, SIZE], room id of floor cells (0 = corridor)
    textures: npt.NDArray[np.uint8]  # [NUM_TEXTURES, TEX, TEX, 3]
    cos: npt.NDArray[np.int64]  # [NUM_ANGLES] fixed-point cos
    sin: npt.NDArray[np.int64]  # [NUM_ANGLES] fixed-point sin
    floor_tint: npt.NDArray[np.int64]  # [num_rooms, 3]
    orbs_xy: npt.NDArray[np.int64]  # [num_orbs, 2] fixed point
    orbs_rgb: npt.NDArray[np.int64]  # [num_orbs, 3]


def _carve() -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
    floor = np.zeros((SIZE, SIZE), dtype=np.bool_)
    rooms = np.zeros((SIZE, SIZE), dtype=np.int16)
    for room_id, (x0, y0, x1, y1) in ROOMS.items():
        floor[y0 : y1 + 1, x0 : x1 + 1] = True
        rooms[y0 : y1 + 1, x0 : x1 + 1] = room_id
    for x0, y0, x1, y1 in CORRIDORS:
        floor[y0 : y1 + 1, x0 : x1 + 1] = True
    walls = np.zeros((SIZE, SIZE), dtype=np.int16)
    for y in range(SIZE):
        for x in range(SIZE):
            if floor[y, x]:
                continue
            tex = Tex.STONE
            # A wall that faces a room takes that room's texture (rooms win over corridors).
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < SIZE and 0 <= nx < SIZE and rooms[ny, nx] > 0:
                    tex = ROOM_WALL_TEX[int(rooms[ny, nx])]
            walls[y, x] = tex + 1
    for x, y, tex in DOORS:
        walls[y, x] = tex + 1
    for x, y in TORCHES:
        if walls[y, x] == 0:
            msg = f"torch at ({x}, {y}) is not on a wall"
            raise ValueError(msg)
        walls[y, x] = Tex.TORCH + 1
    return walls, rooms


def _noise(rng: np.random.Generator, lo: int, hi: int) -> npt.NDArray[np.int64]:
    return rng.integers(lo, hi + 1, size=(TEX, TEX, 1))


def _textures() -> npt.NDArray[np.uint8]:
    """Procedural 16x16 textures. Seeded, so the exported JSON is reproducible."""
    rng = np.random.default_rng(1993)
    ys, xs = np.mgrid[0:TEX, 0:TEX]
    out = np.zeros((NUM_TEXTURES, TEX, TEX, 3), dtype=np.int64)

    def brick_pattern(h: int, w: int) -> npt.NDArray[np.bool_]:
        row = ys // h
        offset = (row % 2) * (w // 2)
        mask: npt.NDArray[np.bool_] = ((ys % h) == 0) | (((xs + offset) % w) == 0)
        return mask

    def fill(index: int, base: tuple[int, int, int], jitter: int) -> None:
        out[index] = np.array(base) + _noise(rng, -jitter, jitter)

    # Corridor stone: large grey blocks.
    fill(Tex.STONE, (110, 108, 104), 10)
    out[Tex.STONE][brick_pattern(8, 8)] = (60, 58, 56)
    # Red brick.
    fill(Tex.BRICK, (168, 58, 48), 14)
    out[Tex.BRICK][brick_pattern(4, 8)] = (70, 40, 36)
    # Blue stone: square tiles with a highlight edge.
    fill(Tex.BLUE, (58, 92, 170), 10)
    out[Tex.BLUE][(ys % 8 == 0) | (xs % 8 == 0)] = (24, 36, 80)
    out[Tex.BLUE][(ys % 8 == 1) & (xs % 8 != 0)] = (110, 150, 220)
    # Moss: green stone blotches.
    fill(Tex.MOSS, (70, 110, 60), 18)
    blotch = rng.random((TEX, TEX)) < 0.25
    out[Tex.MOSS][blotch] = (40, 70, 36)
    out[Tex.MOSS][brick_pattern(8, 16)] = (30, 50, 28)
    # Wood: vertical planks with grain.
    fill(Tex.WOOD, (150, 104, 58), 8)
    out[Tex.WOOD][xs % 4 == 0] = (90, 58, 30)
    grain = (rng.random((TEX, TEX)) < 0.12) & (xs % 4 != 0)
    out[Tex.WOOD][grain] = (120, 80, 44)
    # Hub: purple metal panels with rivets.
    fill(Tex.HUB, (96, 84, 128), 6)
    out[Tex.HUB][(xs % 8 == 0) | (ys % 8 == 0)] = (50, 44, 70)
    out[Tex.HUB][(xs % 8 == 2) & (ys % 8 == 2)] = (200, 190, 230)
    # Torch: stone with a bracket and a bright flame.
    out[Tex.TORCH] = out[Tex.STONE]
    out[Tex.TORCH][(ys >= 9) & (ys <= 13) & (xs >= 7) & (xs <= 8)] = (70, 50, 30)
    flame = ((xs - 7.5) ** 2 / 4.0 + (ys - 6.0) ** 2 / 9.0) <= 1.0
    core = ((xs - 7.5) ** 2 / 1.5 + (ys - 6.5) ** 2 / 4.0) <= 1.0
    out[Tex.TORCH][flame] = (255, 140, 30)
    out[Tex.TORCH][core] = (255, 240, 150)
    # Doors: framed, coloured, with a handle.
    for index, rgb in (
        (Tex.DOOR_RED, (220, 40, 40)),
        (Tex.DOOR_BLUE, (40, 110, 255)),
        (Tex.DOOR_GREEN, (40, 200, 70)),
        (Tex.DOOR_GOLD, (240, 190, 40)),
    ):
        out[index] = np.array(rgb) + _noise(rng, -8, 8)
        frame = (xs <= 1) | (xs >= TEX - 2) | (ys <= 1)
        out[index][frame] = (40, 34, 30)
        out[index][(xs == 8) & (ys >= 2)] = np.array(rgb) // 2
        out[index][(xs >= 11) & (xs <= 12) & (ys >= 8) & (ys <= 9)] = (250, 250, 230)
    return np.clip(out, 0, 255).astype(np.uint8)


def _trig() -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    angles = [2.0 * math.pi * a / NUM_ANGLES for a in range(NUM_ANGLES)]
    cos = np.array([round(TRIG_ONE * math.cos(t)) for t in angles], dtype=np.int64)
    sin = np.array([round(TRIG_ONE * math.sin(t)) for t in angles], dtype=np.int64)
    return cos, sin


def build_world() -> World:
    """Build the (deterministic) world. Cheap enough to call at import time."""
    walls, rooms = _carve()
    cos, sin = _trig()
    return World(
        walls=walls,
        rooms=rooms,
        textures=_textures(),
        cos=cos,
        sin=sin,
        floor_tint=np.array(FLOOR_TINT, dtype=np.int64),
        orbs_xy=np.array([(x, y) for x, y, _ in ORBS], dtype=np.int64),
        orbs_rgb=np.array([rgb for _, _, rgb in ORBS], dtype=np.int64),
    )


WORLD: Final = build_world()


def free_cells(world: World = WORLD) -> list[tuple[int, int]]:
    """All walkable (x, y) tiles."""
    ys, xs = np.nonzero(world.walls == 0)
    return [(int(x), int(y)) for x, y in zip(xs, ys, strict=True)]


def world_to_json(world: World = WORLD) -> str:
    """Serialise the world for the browser port (stable key order, compact)."""
    payload = {
        "size": SIZE,
        "tex": TEX,
        "fixed": FIXED,
        "trigOne": TRIG_ONE,
        "numAngles": NUM_ANGLES,
        "walls": world.walls.reshape(-1).tolist(),
        "rooms": world.rooms.reshape(-1).tolist(),
        "textures": world.textures.reshape(-1).tolist(),
        "cos": world.cos.tolist(),
        "sin": world.sin.tolist(),
        "floorTint": world.floor_tint.tolist(),
        "ceiling": list(CEILING_RGB),
        "orbs": [
            [*xy, *rgb] for xy, rgb in zip(world.orbs_xy.tolist(), world.orbs_rgb.tolist(), strict=True)
        ],
        "orbRadius": ORB_RADIUS,
        "orbCollide": ORB_COLLIDE,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
