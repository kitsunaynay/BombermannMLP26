"""Shared test fixtures and arena builders."""

import numpy as np

import settings as s
from shared.kit import geometry as G


def build_arena(seed: int = 0, crate_density: float = 0.3) -> np.ndarray:
    """A structurally valid arena: border walls plus the odd-odd wall lattice.

    Mirrors ``BombeRLeWorld.build_arena`` (environment.py:348) minus the coin
    and agent placement, so geometry tests run against realistic boards.
    """
    rng = np.random.default_rng(seed)
    field = np.zeros((s.COLS, s.ROWS), dtype=int)
    field[rng.random((s.COLS, s.ROWS)) < crate_density] = G.CRATE
    field[:1, :] = field[-1:, :] = field[:, :1] = field[:, -1:] = G.WALL
    for x in range(s.COLS):
        for y in range(s.ROWS):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = G.WALL
    return field


def clear_start_corners(field: np.ndarray) -> np.ndarray:
    """Free up the four spawn corners, as the framework does."""
    field = field.copy()
    corners = [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]
    for x, y in corners:
        for xx, yy in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if field[xx, yy] == G.CRATE:
                field[xx, yy] = G.FREE
    return field
