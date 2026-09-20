# --------------------------------------------------------------------------
# GENERATED FILE -- DO NOT EDIT.
# Vendored from shared/kit/pathfind.py by tools/sync_kit.py.
# Edit the original, then re-run:  python tools/sync_kit.py
# --------------------------------------------------------------------------
"""Breadth-first navigation and survival analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


def augment_danger_with_threats(
    field: np.ndarray,
    danger: np.ndarray,
    threats: Iterable[Coord],
) -> np.ndarray:
    """Danger map assuming every tile in ``threats`` bombs on this step.

    The escape search otherwise certifies a plan against the bombs visible when
    it commits, and an opponent that drops a bomb one step later can cover the
    escape tile. Measured on `classic` with the stage-3 table: opponents that
    block but never bomb cause 0.00 suicides, opponents that bomb cause 0.38.

    Timing matches :func:`_augment_danger_with_bomb`: a bomb dropped on move 0
    detonates at the end of move ``s.BOMB_TIMER``.
    """
    detonation = int(s.BOMB_TIMER)
    augmented = None
    for tx, ty in threats:
        if not in_bounds(field, tx, ty):
            continue
        for cx, cy in blast_coords(field, tx, ty):
            if detonation < danger[cx, cy]:
                if augmented is None:
                    augmented = danger.copy()
                if detonation < augmented[cx, cy]:
                    augmented[cx, cy] = detonation
    return danger if augmented is None else augmented


def survives_after(
    field: np.ndarray,
    start: Coord,
    action: int,
    danger: np.ndarray,
    passable: np.ndarray,
    horizon: int = SURVIVAL_HORIZON,
    threats: Sequence[Coord] = (),
) -> bool:
    """Does a follow-up plan exist that keeps us alive after taking ``action``?

    Time-indexed breadth-first search over ``(x, y, moves_taken)``. A state is
    admissible only if the tile is not lethal at that exact moment, using the
    interval test in :func:`kit.geometry.lethal_at`.

    Success is surviving to ``horizon``, or reaching a tile no known bomb
    threatens, from where waiting is safe.

    Only currently-visible bombs are modelled unless ``threats`` is given, in
    which case each listed tile is treated as bombing on this step.
    """
    sx, sy = start

    if threats:
        danger = augment_danger_with_threats(field, danger, threats)

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
    threats: Sequence[Coord] = (),
) -> np.ndarray:
    """Boolean mask over :data:`kit.actions.ACTIONS` of non-suicidal actions.

    Fast path: with no bombs and no explosions in play nothing can kill us, so
    every legal action is safe and the search is skipped entirely. This matters
    because the mask is evaluated on every step of every training episode.
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    sx, sy = start
    if threats:
        danger = augment_danger_with_threats(field, danger, threats)
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
    threats: Sequence[Coord] = (),
) -> bool:
    """Can we drop a bomb here and still get away? The anti-suicide primitive."""
    return survives_after(field, start, BOMB, danger, passable, horizon, threats)


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


def armed_opponents(game_state: dict) -> List[Coord]:
    """Positions of opponents that could drop a bomb on this step.

    Reads the third field of each ``others`` entry, which the framework sets to
    the same ``bombs_left`` flag it reports for us. An opponent with a bomb
    already ticking cannot drop another, so excluding those keeps the
    pessimistic danger map from being needlessly wide.
    """
    return [xy for (_, _, bomb_available, xy) in game_state["others"] if bomb_available]


def opponent_reachability(
    field: np.ndarray,
    others: Sequence[Coord],
    horizon: int = SURVIVAL_HORIZON,
    origin: Optional[Coord] = None,
) -> List[np.ndarray]:
    """Over-approximate which tiles any opponent could occupy at each future step.

    Floor-connectivity only: an opponent may stay or step to an orthogonally
    free tile, ignoring bombs and other agents as blockers. That's a
    deliberate over-approximation, since it feeds the ``contested`` signal
    the policy learns to weigh, rather than a hard mask.

    Because "stay in place" is always an option, the reachable set only grows
    with ``t`` and typically floods most of an open board within a few steps.
    A Python set-of-tuples BFS recomputing that growing frontier every game
    step measured as **the dominant cost of the whole tensorizer** (5.2x
    rollout slowdown, profiled to this function). Vectorised numpy shifts do
    the same flood as bulk boolean-array ops instead, the same way
    :func:`kit.geometry.danger_map` avoids a per-tile Python loop.

    ``origin``, when given, additionally drops any opponent more than
    ``horizon`` moves away (Manhattan distance) up front: a move changes that
    distance by at most 1, so such an opponent provably cannot reach
    anywhere our own ``horizon``-limited search would consider.

    Returns a list indexed by time (index 0 = current positions) of a
    boolean mask, shaped like ``field``, of tiles reachable by *some*
    opponent by that step.
    """
    if origin is not None:
        ox, oy = origin
        others = [
            (x, y) for (x, y) in others if abs(x - ox) + abs(y - oy) <= horizon
        ]

    free = field == 0
    current = np.zeros(field.shape, dtype=bool)
    for x, y in others:
        if in_bounds(field, x, y):
            current[x, y] = True

    reachable = [current.copy()]
    for _ in range(horizon):
        expanded = current.copy()
        expanded[1:, :] |= current[:-1, :]
        expanded[:-1, :] |= current[1:, :]
        expanded[:, 1:] |= current[:, :-1]
        expanded[:, :-1] |= current[:, 1:]
        expanded &= free
        current = expanded
        reachable.append(current.copy())
    return reachable


@dataclass(frozen=True)
class SurvivalProfile:
    """Continuous counterpart to :func:`survives_after`'s boolean verdict.

    ``duration``: last time step a follow-up plan was found for (capped at
    ``horizon``). ``breadth``: how many alternative follow-ups existed along
    the way, normalised to ``[0, 1]`` (wide == robust to a wrong guess about
    what the opponent does next). ``min_margin``: the tightest slack, in
    steps, between being on a tile and that tile turning lethal, across the
    best-surviving path. ``contested``: fraction of the surviving frontier
    that overlaps tiles an opponent could also reach at the same time,
    averaged over the plan's duration -- 0 when no ``opponent_reachable`` is
    supplied.
    """

    safe: bool
    duration: int
    breadth: float
    min_margin: float
    contested: float
    #: First time step at which the plan stands on a tile with no danger at
    #: all (0 = the very first tile). ``-1`` when no such tile is reached.
    exit_step: int = -1
    #: Size of the surviving frontier when the search stopped: how many
    #: distinct tiles the agent could be on at the end of the plan.
    terminal_width: int = 0


def _tile_margin(danger: np.ndarray, x: int, y: int, t: int, cap: float) -> float:
    d = int(danger[x, y])
    if d == SAFE:
        return cap
    return max(0.0, min(cap, float(d - t)))


def survival_profile(
    field: np.ndarray,
    start: Coord,
    action: int,
    danger: np.ndarray,
    passable: np.ndarray,
    horizon: int = SURVIVAL_HORIZON,
    threats: Sequence[Coord] = (),
    opponent_reachable: Optional[Sequence[np.ndarray]] = None,
) -> SurvivalProfile:
    """Richer sibling of :func:`survives_after`: how robust is the escape, not
    just whether one exists.

    Shares :func:`survives_after`'s admissibility rules (same ``danger``,
    ``passable``, bomb-sealing and ``threats`` semantics) but keeps the whole
    frontier at each time step instead of stopping at the first admissible
    state, so it can report the plan's width and margin alongside its
    existence. This is meant to feed the PPO tensorizer's continuous safety
    channels (``docs/ROUTES.md`` route A), not to replace the cheaper boolean
    search the mask uses on every candidate action.
    """
    sx, sy = start
    if threats:
        danger = augment_danger_with_threats(field, danger, threats)

    if action == BOMB:
        working_danger = _augment_danger_with_bomb(field, danger, sx, sy)
        first_x, first_y = sx, sy
        sealed: Optional[Coord] = (sx, sy)
    else:
        working_danger = danger
        dx, dy = ACTION_DELTAS[action]
        first_x, first_y = sx + dx, sy + dy
        sealed = None
        if action in MOVE_ACTIONS:
            if not in_bounds(field, first_x, first_y) or not passable[first_x, first_y]:
                return SurvivalProfile(False, 0, 0.0, 0.0, 1.0, -1, 0)

    margin_cap = float(horizon)

    if lethal_at(working_danger, first_x, first_y, 0):
        return SurvivalProfile(False, 0, 0.0, 0.0, 1.0, -1, 0)

    def contested_at(tiles: Set[Coord], t: int) -> float:
        if not opponent_reachable or not tiles:
            return 0.0
        opp = opponent_reachable[min(t, len(opponent_reachable) - 1)]
        return sum(1 for tile in tiles if opp[tile]) / len(tiles)

    if working_danger[first_x, first_y] == SAFE:
        return SurvivalProfile(
            True, horizon, 1.0, margin_cap, contested_at({(first_x, first_y)}, 0), 0, 1
        )

    # pos -> best margin seen along the best path reaching it
    frontier: Dict[Coord, float] = {(first_x, first_y): _tile_margin(working_danger, first_x, first_y, 0, margin_cap)}
    widths = [1]
    contested_steps = [contested_at(set(frontier), 0)]
    duration = 0

    for t in range(horizon):
        candidates: Dict[Coord, float] = {}
        for (x, y), margin in frontier.items():
            options = [(x, y)]
            for direction in range(1, len(DIR_DELTAS)):
                dx, dy = DIR_DELTAS[direction]
                nx, ny = x + dx, y + dy
                if not in_bounds(field, nx, ny):
                    continue
                if sealed is not None and (nx, ny) == sealed:
                    continue
                if not passable[nx, ny]:
                    continue
                options.append((nx, ny))

            nt = t + 1
            for nx, ny in options:
                if lethal_at(working_danger, nx, ny, nt):
                    continue
                new_margin = min(margin, _tile_margin(working_danger, nx, ny, nt, margin_cap))
                existing = candidates.get((nx, ny))
                if existing is None or new_margin > existing:
                    candidates[(nx, ny)] = new_margin

        if not candidates:
            break

        frontier = candidates
        duration = t + 1
        widths.append(len(frontier))
        contested_steps.append(contested_at(set(frontier), duration))

        if any(working_danger[xy] == SAFE for xy in frontier):
            breadth = float(np.mean([min(1.0, w / 8.0) for w in widths]))
            best_margin = max(
                m for xy, m in frontier.items() if working_danger[xy] == SAFE
            )
            return SurvivalProfile(
                True, horizon, breadth, best_margin, float(np.mean(contested_steps)),
                duration, len(frontier),
            )

    breadth = float(np.mean([min(1.0, w / 8.0) for w in widths]))
    best_margin = max(frontier.values()) if frontier else 0.0
    safe = duration >= horizon
    return SurvivalProfile(
        safe, duration, breadth, best_margin, float(np.mean(contested_steps)), -1, len(frontier)
    )


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
