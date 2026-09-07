# --------------------------------------------------------------------------
# GENERATED FILE -- DO NOT EDIT.
# Vendored from shared/kit/pathfind.py by tools/sync_kit.py.
# Edit the original, then re-run:  python tools/sync_kit.py
# --------------------------------------------------------------------------
"""Breadth-first navigation and survival analysis.

Two families of function live here.

*Navigation* (:func:`bfs`, :func:`direction_to_nearest`) answers "which way do I
step to approach the nearest coin/crate". This replaces ``look_for_targets``
from the provided rule_based_agent, which shuffles its neighbour order and so
returns different answers for the same state. That is unusable as a feature.

*Survival* (:func:`survives_after`, :func:`safe_actions`) answers "if I take
this action, does a sequence of follow-up moves exist that keeps me alive". A
plain distance check is not enough: escaping a blast is a timing problem, so
this does a time-indexed search over ``(x, y, moves_taken)`` where a tile is
enterable only if it is not lethal at the moment you would be standing on it.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

import settings as s

from .actions import (
    ACTION_DELTAS,
    BOMB,
    DIR_DELTAS,
    MOVE_ACTIONS,
    N_ACTIONS,
    WAIT,
    action_to_dir,
)
from .geometry import (
    BLAST_DURATION,
    SAFE,
    blast_coords,
    free_mask,
    in_bounds,
    lethal_at,
)

Coord = Tuple[int, int]

#: Horizon for survival search: long enough to outlast a freshly dropped bomb
#: (``BOMB_TIMER`` steps) plus the lingering blast.
SURVIVAL_HORIZON = int(s.BOMB_TIMER + BLAST_DURATION)


def bfs(
    field: np.ndarray,
    start: Coord,
    passable: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Breadth-first search from ``start`` over ``passable`` tiles.

    Neighbours are expanded in a fixed UP/RIGHT/DOWN/LEFT order, so the same
    game state always produces the same features.

    Returns ``(dist, first_step)`` where ``dist[x, y]`` is the number of moves
    to reach the tile (``-1`` if unreachable) and ``first_step[x, y]`` is the
    direction code (1..4) of the opening move of a shortest path to it.
    """
    dist = np.full(field.shape, -1, dtype=np.int16)
    first_step = np.zeros(field.shape, dtype=np.int8)

    sx, sy = start
    # The start tile is always reachable, even when it is not "passable" --
    # an agent standing on its own freshly dropped bomb is the common case.
    dist[sx, sy] = 0

    queue = deque([(sx, sy)])
    while queue:
        x, y = queue.popleft()
        for direction in range(1, len(DIR_DELTAS)):
            dx, dy = DIR_DELTAS[direction]
            nx, ny = x + dx, y + dy
            if not in_bounds(field, nx, ny):
                continue
            if dist[nx, ny] != -1:
                continue
            if not passable[nx, ny]:
                continue
            dist[nx, ny] = dist[x, y] + 1
            # Inherit the opening move, or start one if we are leaving the origin.
            first_step[nx, ny] = direction if (x, y) == (sx, sy) else first_step[x, y]
            queue.append((nx, ny))

    return dist, first_step


def nearest_target(
    dist: np.ndarray,
    targets: Iterable[Coord],
) -> Tuple[Optional[Coord], int]:
    """Closest reachable target and its distance, or ``(None, -1)``."""
    best: Optional[Coord] = None
    best_dist = -1
    for tx, ty in targets:
        if not (0 <= tx < dist.shape[0] and 0 <= ty < dist.shape[1]):
            continue
        d = int(dist[tx, ty])
        if d < 0:
            continue
        if best is None or d < best_dist:
            best, best_dist = (tx, ty), d
    return best, best_dist


def direction_to_nearest(
    dist: np.ndarray,
    first_step: np.ndarray,
    targets: Iterable[Coord],
) -> int:
    """Direction code of the opening move toward the nearest reachable target.

    Returns ``0`` when nothing is reachable, or when we are already standing on
    the closest target.
    """
    target, d = nearest_target(dist, targets)
    if target is None or d <= 0:
        return 0
    return int(first_step[target])


def distance_to_nearest(dist: np.ndarray, targets: Iterable[Coord]) -> int:
    """Move count to the nearest reachable target, ``-1`` if none is reachable."""
    return nearest_target(dist, targets)[1]


def _augment_danger_with_bomb(
    field: np.ndarray,
    danger: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    """Danger map as it would be after dropping a bomb at ``(x, y)`` right now.

    In the "moves from now" frame, move 0 is the current step. A bomb dropped
    on move 0 detonates at the end of move ``s.BOMB_TIMER``: the framework sets
    ``timer = BOMB_TIMER`` and decrements it once in the same step, so the agent
    sees ``BOMB_TIMER - 1`` on the next step and gets exactly ``BOMB_TIMER``
    further moves before it goes off.
    """
    augmented = danger.copy()
    detonation = int(s.BOMB_TIMER)
    for cx, cy in blast_coords(field, x, y):
        if detonation < augmented[cx, cy]:
            augmented[cx, cy] = detonation
    return augmented


def survives_after(
    field: np.ndarray,
    start: Coord,
    action: int,
    danger: np.ndarray,
    passable: np.ndarray,
    horizon: int = SURVIVAL_HORIZON,
) -> bool:
    """Does a follow-up plan exist that keeps us alive after taking ``action``?

    Time-indexed breadth-first search over ``(x, y, moves_taken)``. A state is
    admissible only if the tile is not lethal at that exact moment, using the
    interval test in :func:`kit.geometry.lethal_at`.

    Success is surviving to ``horizon``, or reaching a tile no known bomb
    threatens, from where waiting is safe.

    Only currently-visible bombs are modelled; opponents dropping new bombs
    later is out of scope.
    """
    sx, sy = start

    if action == BOMB:
        working_danger = _augment_danger_with_bomb(field, danger, sx, sy)
        first_x, first_y = sx, sy
        # The bomb now occupies our tile; we may step off it but never back on.
        sealed: Optional[Coord] = (sx, sy)
    else:
        working_danger = danger
        dx, dy = ACTION_DELTAS[action]
        first_x, first_y = sx + dx, sy + dy
        sealed = None
        if action in MOVE_ACTIONS:
            if not in_bounds(field, first_x, first_y) or not passable[first_x, first_y]:
                return False  # illegal move: the framework would score it INVALID_ACTION

    if lethal_at(working_danger, first_x, first_y, 0):
        return False
    if working_danger[first_x, first_y] == SAFE:
        return True

    visited: Set[Tuple[int, int, int]] = {(first_x, first_y, 0)}
    queue = deque([(first_x, first_y, 0)])

    while queue:
        x, y, t = queue.popleft()
        if t >= horizon:
            return True

        nt = t + 1
        # Candidate follow-ups: hold position, or step to an adjacent free tile.
        candidates: List[Coord] = [(x, y)]
        for direction in range(1, len(DIR_DELTAS)):
            dx, dy = DIR_DELTAS[direction]
            nx, ny = x + dx, y + dy
            if not in_bounds(field, nx, ny):
                continue
            if sealed is not None and (nx, ny) == sealed:
                continue
            if not passable[nx, ny]:
                continue
            candidates.append((nx, ny))

        for nx, ny in candidates:
            if lethal_at(working_danger, nx, ny, nt):
                continue
            if working_danger[nx, ny] == SAFE:
                return True
            key = (nx, ny, nt)
            if key in visited:
                continue
            visited.add(key)
            queue.append(key)

    return False


def safe_actions(
    field: np.ndarray,
    start: Coord,
    danger: np.ndarray,
    passable: np.ndarray,
    bomb_available: bool = True,
    horizon: int = SURVIVAL_HORIZON,
) -> np.ndarray:
    """Boolean mask over :data:`kit.actions.ACTIONS` of non-suicidal actions.

    Fast path: with no bombs and no explosions in play nothing can kill us, so
    every legal action is safe and the search is skipped entirely. This matters
    because the mask is evaluated on every step of every training episode.
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    sx, sy = start
    no_threat = bool(np.all(danger == SAFE))

    for action in range(N_ACTIONS):
        if action == BOMB and not bomb_available:
            continue
        if action in MOVE_ACTIONS:
            dx, dy = ACTION_DELTAS[action]
            nx, ny = sx + dx, sy + dy
            if not in_bounds(field, nx, ny) or not passable[nx, ny]:
                continue  # would be an INVALID_ACTION
        if no_threat and action != BOMB:
            mask[action] = True
            continue
        mask[action] = survives_after(field, start, action, danger, passable, horizon)

    return mask


def has_escape_after_bomb(
    field: np.ndarray,
    start: Coord,
    danger: np.ndarray,
    passable: np.ndarray,
    horizon: int = SURVIVAL_HORIZON,
) -> bool:
    """Can we drop a bomb here and still get away? The anti-suicide primitive."""
    return survives_after(field, start, BOMB, danger, passable, horizon)


def escape_direction(
    field: np.ndarray,
    start: Coord,
    danger: np.ndarray,
    passable: np.ndarray,
) -> int:
    """Direction code pointing at the nearest tile no bomb threatens.

    Returns ``0`` when we are already safe or when no safe tile is reachable.
    """
    sx, sy = start
    if danger[sx, sy] == SAFE:
        return 0

    dist, first_step = bfs(field, start, passable)
    safe_x, safe_y = np.nonzero((danger == SAFE) & (dist >= 0))
    if len(safe_x) == 0:
        return 0
    return direction_to_nearest(dist, first_step, zip(safe_x.tolist(), safe_y.tolist()))


def game_state_context(game_state: dict):
    """Unpack the pieces of ``game_state`` every feature extractor needs.

    Returns ``(field, (x, y), bomb_available, bombs, others, coins, danger,
    passable)``. Centralised so the QL features and the PPO tensorizer cannot
    drift in how they read the state dictionary.
    """
    from .geometry import danger_map  # local import keeps the module import graph flat

    field = game_state["field"]
    _, _, bomb_available, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    others = [xy for (_, _, _, xy) in game_state["others"]]
    coins = game_state["coins"]

    danger = danger_map(field, bombs, game_state["explosion_map"])
    passable = free_mask(field, bombs, others)

    return field, (x, y), bool(bomb_available), bombs, others, coins, danger, passable
