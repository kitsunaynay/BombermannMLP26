"""Geometry tests, cross-checked against the framework's own bomb logic."""

import numpy as np
import pytest

import settings as s
from helpers import build_arena
from items import Bomb
from shared.kit import geometry as G


@pytest.mark.parametrize("seed", range(5))
def test_blast_coords_matches_framework(seed):
    """Our blast must be byte-identical to items.Bomb.get_blast_coords.

    This is the ground-truth check for the whole danger model: if the blast
    shape is wrong, every survival decision downstream is wrong too.
    """
    field = build_arena(seed)
    for x in range(1, s.COLS - 1):
        for y in range(1, s.ROWS - 1):
            if field[x, y] == G.WALL:
                continue
            reference = Bomb((x, y), None, s.BOMB_TIMER, s.BOMB_POWER, None)
            assert set(reference.get_blast_coords(field)) == set(G.blast_coords(field, x, y))


def test_blast_stops_at_walls_and_passes_crates():
    field = np.full((9, 9), G.WALL)
    field[1:8, 4] = G.FREE
    field[3, 4] = G.CRATE  # a crate must NOT stop the blast

    coords = set(G.blast_coords(field, 4, 4))

    assert (3, 4) in coords, "blast should pass through (and destroy) crates"
    assert (1, 4) in coords, "power 3 reaches three tiles left"
    assert (4, 3) not in coords, "blast must not enter a wall"


def test_blast_does_not_turn_corners():
    field = np.full((9, 9), G.WALL)
    field[1:8, 4] = G.FREE
    field[6, 1:8] = G.FREE

    coords = set(G.blast_coords(field, 4, 4))

    assert (6, 4) in coords, "reaches the junction along the corridor"
    assert (6, 3) not in coords, "must not propagate around the corner"


def test_danger_map_encodes_timers_and_takes_the_minimum():
    field = np.full((9, 9), G.WALL)
    field[1:8, 4] = G.FREE

    danger = G.danger_map(field, [((2, 4), 3), ((6, 4), 1)])

    assert danger[2, 4] == 3
    assert danger[6, 4] == 1
    # (4, 4) sits in both blasts; the sooner one governs.
    assert danger[4, 4] == 1
    assert danger[1, 4] == 3
    assert danger[4, 3] == G.SAFE, "walls are never threatened"


def test_danger_map_folds_in_active_explosions():
    """explosion_map >= 1 means lethal this step (environment.py:411-416)."""
    field = np.zeros((9, 9), dtype=int)
    explosion_map = np.zeros((9, 9))
    explosion_map[5, 5] = 1
    explosion_map[6, 6] = 0  # smoke: reported but harmless

    danger = G.danger_map(field, [], explosion_map)

    assert danger[5, 5] == 0
    assert danger[6, 6] == G.SAFE


def test_lethal_at_covers_the_full_blast_duration():
    """A blast keeps killing for EXPLOSION_TIMER consecutive steps."""
    danger = np.full((3, 3), G.SAFE, dtype=np.int16)
    danger[1, 1] = 2

    assert not G.lethal_at(danger, 1, 1, when=1)
    assert G.lethal_at(danger, 1, 1, when=2)
    assert G.lethal_at(danger, 1, 1, when=3) == (G.BLAST_DURATION > 1)
    assert not G.lethal_at(danger, 1, 1, when=2 + G.BLAST_DURATION)


def test_free_mask_matches_tile_is_free_semantics():
    """Bombs and agents block movement; coins do not (environment.py:121)."""
    field = np.zeros((9, 9), dtype=int)
    field[3, 3] = G.CRATE

    mask = G.free_mask(field, bombs=[((4, 4), 2)], others=[(5, 5)])

    assert not mask[3, 3], "crates block"
    assert not mask[4, 4], "bombs block"
    assert not mask[5, 5], "other agents block"
    assert mask[6, 6]


def test_adjacent_crates_and_dead_end():
    field = np.full((9, 9), G.WALL)
    field[4, 4] = G.FREE
    field[4, 5] = G.FREE  # single free neighbour -> dead end
    field[5, 4] = G.CRATE

    assert G.adjacent_crates(field, 4, 4) == 1
    assert G.is_dead_end(field, 4, 4)

    field[3, 4] = G.FREE
    assert not G.is_dead_end(field, 4, 4)
