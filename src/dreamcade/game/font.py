"""A 3x5 pixel font for the score digits.

Several glyph pairs (0/8, 8/9, 3/9, 5/6, ...) differ by a single pixel, which is exactly
why the evaluation decodes digits by *exact* match and treats anything else as a
score mismatch rather than snapping to the nearest glyph.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

_GLYPH_ROWS: Final[tuple[tuple[str, ...], ...]] = (
    ("111", "101", "101", "101", "111"),  # 0
    ("010", "110", "010", "010", "111"),  # 1
    ("111", "001", "111", "100", "111"),  # 2
    ("111", "001", "111", "001", "111"),  # 3
    ("101", "101", "111", "001", "001"),  # 4
    ("111", "100", "111", "001", "111"),  # 5
    ("111", "100", "111", "101", "111"),  # 6
    ("111", "001", "010", "010", "010"),  # 7
    ("111", "101", "111", "101", "111"),  # 8
    ("111", "101", "111", "001", "111"),  # 9
)

#: ``GLYPHS[d]`` is a ``bool[5, 3]`` bitmap for digit ``d``.
GLYPHS: Final[npt.NDArray[np.bool_]] = np.array(
    [[[c == "1" for c in row] for row in glyph] for glyph in _GLYPH_ROWS], dtype=np.bool_
)
GLYPHS.flags.writeable = False


def glyph_rows_as_bits() -> list[list[int]]:
    """Return each glyph as five 3-bit row masks (MSB = leftmost pixel), for the JS port."""
    return [[int(row, 2) for row in glyph] for glyph in _GLYPH_ROWS]
