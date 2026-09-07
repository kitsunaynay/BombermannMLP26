"""Blast geometry and danger accounting.

When a tile kills you follows from the step resolution order
(environment.py:158)::

    poll_and_run_agents()   # every agent moves
    collect_coins()
    update_explosions()     # explosions age; timer 1 -> harmless smoke
    update_bombs()          # bombs with timer <= 0 detonate NOW
    evaluate_explosions()   # agents standing in a live blast die

The agent moves first and the world resolves after, so a bomb reported as
``t = 0`` in ``game_state['bombs']`` detonates at the end of the step being
decided now.

This module encodes that as a single ``danger`` array with the semantics:

    ``danger[x, y] == k``  ->  the tile is lethal at the end of move ``k``
                              (``k = 0`` means "lethal at the end of this step")
    ``danger[x, y] == SAFE`` -> no known bomb or explosion threatens the tile

Because an explosion lingers for one extra dangerous step
(``s.EXPLOSION_TIMER == 2``), a tile with ``danger == k`` is actually deadly for
occupancy at the end of moves ``k`` and ``k + 1``; use :func:`lethal_at` rather
than comparing against ``danger`` by hand.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

import settings as s

WALL = -1
FREE = 0
CRATE = 1

#: Sentinel for "this tile is not threatened". Large enough to be bigger than
#: any real timer, small enough to keep the array in a compact int dtype.
SAFE = 99

#: Number of steps a blast keeps killing once it goes off (2 with stock rules).
BLAST_DURATION = s.EXPLOSION_TIMER

Coord = Tuple[int, int]


def blast_coords(field: np.ndarray, x: int, y: int, power: int | None = None) -> List[Coord]:
    """Tiles covered by a bomb detonating at ``(x, y)``.

    Mirrors :meth:`items.Bomb.get_blast_coords` exactly: the blast stops at the
    first stone wall in each direction, passes *through* crates, and does not
    turn corners. Bounds checking is unnecessary because the arena border is
    solid wall, which is the same assumption the framework makes.
    """
    if power is None:
        power = s.BOMB_POWER

    coords: List[Coord] = [(x, y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == WALL:
                break
            coords.append((nx, ny))
    return coords


def danger_map(
    field: np.ndarray,
    bombs: Sequence[Tuple[Coord, int]],
    explosion_map: np.ndarray | None = None,
    power: int | None = None,
) -> np.ndarray:
    """Steps-until-lethal for every tile. See module docstring for semantics.

    ``bombs`` is ``game_state['bombs']`` and ``explosion_map`` is
    ``game_state['explosion_map']``. The framework only reports *dangerous*
    explosions there, encoded as ``timer - 1`` (environment.py:411-416), so a
    value ``>= 1`` means "this tile kills at the end of the current step" --
    which is precisely the test the provided rule_based_agent uses.
    """
    danger = np.full(field.shape, SAFE, dtype=np.int16)

    for (bx, by), timer in bombs:
        for cx, cy in blast_coords(field, bx, by, power):
            if timer < danger[cx, cy]:
                danger[cx, cy] = timer

    if explosion_map is not None:
        danger[np.asarray(explosion_map) >= 1] = 0

    return danger


def lethal_at(danger: np.ndarray, x: int, y: int, when: int = 0) -> bool:
    """Would standing on ``(x, y)`` at the end of move ``when`` be fatal?

    A blast kills for :data:`BLAST_DURATION` consecutive steps starting at the
    step its bomb detonates, hence the interval test.
    """
    d = int(danger[x, y])
    return d <= when < d + BLAST_DURATION


def is_lethal_now(danger: np.ndarray, x: int, y: int) -> bool:
    """Shorthand for ``lethal_at(danger, x, y, when=0)``."""
    return lethal_at(danger, x, y, 0)


def in_bounds(field: np.ndarray, x: int, y: int) -> bool:
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def occupied_tiles(
    bombs: Sequence[Tuple[Coord, int]] = (),
    others: Sequence[Coord] = (),
) -> set:
    """Tiles blocked by bombs or other agents.

    ``GenericWorld.tile_is_free`` (environment.py:121) treats both as
    obstacles, so movement planning has to as well.
    """
    blocked = {xy for xy, _ in bombs}
    blocked.update(others)
    return blocked


def free_mask(
    field: np.ndarray,
    bombs: Sequence[Tuple[Coord, int]] = (),
    others: Sequence[Coord] = (),
) -> np.ndarray:
    """Boolean array of tiles an agent may legally step onto.

    Matches ``GenericWorld.tile_is_free``: the tile must be empty floor and
    must not hold a bomb or another agent. Note that coins do *not* block.
    """
    mask = field == FREE
    for bx, by in occupied_tiles(bombs, others):
        if in_bounds(field, bx, by):
            mask[bx, by] = False
    return mask


def tile_free(
    field: np.ndarray,
    x: int,
    y: int,
    bombs: Sequence[Tuple[Coord, int]] = (),
    others: Sequence[Coord] = (),
) -> bool:
    """Single-tile version of :func:`free_mask`."""
    if not in_bounds(field, x, y):
        return False
    if field[x, y] != FREE:
        return False
    return (x, y) not in occupied_tiles(bombs, others)


def adjacent_crates(field: np.ndarray, x: int, y: int) -> int:
    """Number of crates orthogonally adjacent to ``(x, y)``."""
    count = 0
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx, ny = x + dx, y + dy
        if in_bounds(field, nx, ny) and field[nx, ny] == CRATE:
            count += 1
    return count


def is_dead_end(field: np.ndarray, x: int, y: int) -> bool:
    """True when exactly one orthogonal neighbour is free floor.

    Dead ends are good bombing spots against opponents and are used as targets
    by the provided rule_based_agent.
    """
    if field[x, y] != FREE:
        return False
    free_neighbours = 0
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx, ny = x + dx, y + dy
        if in_bounds(field, nx, ny) and field[nx, ny] == FREE:
            free_neighbours += 1
    return free_neighbours == 1


def crate_positions(field: np.ndarray) -> List[Coord]:
    xs, ys = np.nonzero(field == CRATE)
    return list(zip(xs.tolist(), ys.tolist()))
