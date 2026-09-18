from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .config import QLConfig
from .kit import geometry as G
from .kit import pathfind as P
from .kit import symmetry
from .kit.actions import DIR_DELTAS

# state is tuple of small ints keying dict directly; ~10^4-10^5 reachable of 1.8M possible
NB_FREE, NB_BLOCKED, NB_LETHAL = 0, 1, 2

# danger values capped at 4 (no known threat)
DANGER_CAP = 4


def _neighbour_code(
    field: np.ndarray,
    passable: np.ndarray,
    danger: np.ndarray,
    x: int,
    y: int,
) -> int:
    if not G.in_bounds(field, x, y):
        return NB_BLOCKED
    if not passable[x, y]:
        return NB_BLOCKED
    if G.lethal_at(danger, x, y, when=0):
        return NB_LETHAL
    return NB_FREE


def _opponent_proximity(position: Tuple[int, int], others: Sequence[Tuple[int, int]]) -> int:
    if not others:
        return 0
    x, y = position
    closest = min(abs(ox - x) + abs(oy - y) for ox, oy in others)
    if closest <= 1:
        return 2
    if closest <= 3:
        return 1
    return 0


def _dead_end_exit(field: np.ndarray, x: int, y: int) -> Optional[Tuple[int, int]]:
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx, ny = x + dx, y + dy
        if G.in_bounds(field, nx, ny) and field[nx, ny] == G.FREE:
            return nx, ny
    return None


def _trap_features(
    field: np.ndarray,
    position: Tuple[int, int],
    others: Sequence[Tuple[int, int]],
) -> Dict[str, int]:
    # rule_based_agent bombs dead ends; trappable if we block exit
    x, y = position
    in_dead_end = int(G.is_dead_end(field, x, y))

    opponent_in_dead_end = 0
    can_seal_opponent = 0
    for ox, oy in others:
        if abs(ox - x) + abs(oy - y) > 3:
            continue
        if not G.is_dead_end(field, ox, oy):
            continue
        opponent_in_dead_end = 1
        exit_tile = _dead_end_exit(field, ox, oy)
        if exit_tile is None:
            continue
        ex, ey = exit_tile
        if (x, y) == (ex, ey) or abs(ex - x) + abs(ey - y) == 1:
            can_seal_opponent = 1

    return {
        "in_dead_end": in_dead_end,
        "opponent_in_dead_end": opponent_in_dead_end,
        "can_seal_opponent": can_seal_opponent,
    }


def compute_all_features(game_state: dict) -> Dict[str, int]:
    # compute all features; variant selects which go in the key
    (
        field,
        position,
        bomb_available,
        bombs,
        others,
        coins,
        danger,
        passable,
    ) = P.game_state_context(game_state)

    x, y = position

    # navigation
    dist, first_step = P.bfs(field, position, passable)
    coin_dir = P.direction_to_nearest(dist, first_step, coins)

    # crates are blocked; path to adjacent tile instead
    crate_dir = 0
    crates = G.crate_positions(field)
    if crates:
        approach = set()
        for cx, cy in crates:
            for dx, dy in DIR_DELTAS[1:]:
                nx, ny = cx + dx, cy + dy
                if G.in_bounds(field, nx, ny) and dist[nx, ny] >= 0:
                    approach.add((nx, ny))
        crate_dir = P.direction_to_nearest(dist, first_step, sorted(approach))

    # danger state
    danger_here = int(min(danger[x, y], DANGER_CAP))
    escape_dir = P.escape_direction(field, position, danger, passable) if danger_here < DANGER_CAP else 0

    # adjacent tiles state
    neighbours = {}
    for name, direction in (
        ("neighbour_up", 1),
        ("neighbour_right", 2),
        ("neighbour_down", 3),
        ("neighbour_left", 4),
    ):
        dx, dy = DIR_DELTAS[direction]
        neighbours[name] = _neighbour_code(field, passable, danger, x + dx, y + dy)

    # bomb safety
    escape_if_bomb = 0
    if bomb_available:
        escape_if_bomb = int(P.has_escape_after_bomb(field, position, danger, passable))

    return {
        "coin_dir": coin_dir,
        "crate_dir": crate_dir,
        "escape_dir": escape_dir,
        "danger_here": danger_here,
        **neighbours,
        "bomb_available": int(bool(bomb_available)),
        "crates_adjacent": min(G.adjacent_crates(field, x, y), 2),
        "opponent_near": _opponent_proximity(position, others),
        "escape_if_bomb": escape_if_bomb,
        **_trap_features(field, position, others),
    }


# canonicalize to d4 orbit representative; shrinks table 8x; share corners
# direction features rotate cleanly only with unique nearest target
DIRECTION_FEATURES = ("coin_dir", "crate_dir", "escape_dir")

NEIGHBOUR_BY_DIR = {
    1: "neighbour_up",
    2: "neighbour_right",
    3: "neighbour_down",
    4: "neighbour_left",
}


def _transform_values(values: Dict[str, int], transform: str) -> Dict[str, int]:
    # rotate feature values to canonical frame
    out = dict(values)

    for name in DIRECTION_FEATURES:
        if name in values:
            out[name] = symmetry.transform_direction(values[name], transform)

    # neighbour direction rotates; read from values to avoid clobbering
    for direction, name in NEIGHBOUR_BY_DIR.items():
        moved = symmetry.transform_direction(direction, transform)
        out[NEIGHBOUR_BY_DIR[moved]] = values[name]

    return out


class FeatureView(NamedTuple):
    # transform maps the real board into the canonical frame; an action
    # chosen against `key` must go through symmetry.inverse_transform_action
    # before it's actually played.
    key: Tuple[int, ...]
    transform: str


def extract(
    game_state: Optional[dict],
    config: Optional[QLConfig] = None,
) -> Optional[FeatureView]:
    # extract features and canonicalize if symmetry enabled
    if game_state is None:
        return None

    config = config or QLConfig.load()
    names = config.feature_names
    values = compute_all_features(game_state)

    if not config.use_symmetry:
        return FeatureView(tuple(values[name] for name in names), "identity")

    # find lexicographically smallest key among d4 transforms
    best_key: Optional[Tuple[int, ...]] = None
    best_transform = "identity"
    for transform in symmetry.TRANSFORMS:
        candidate = _transform_values(values, transform)
        key = tuple(candidate[name] for name in names)
        if best_key is None or key < best_key:
            best_key, best_transform = key, transform

    return FeatureView(best_key, best_transform)


def state_to_features(
    game_state: Optional[dict],
    config: Optional[QLConfig] = None,
) -> Optional[Tuple[int, ...]]:
    view = extract(game_state, config)
    return None if view is None else view.key


def feature_cardinalities(config: QLConfig) -> Tuple[int, ...]:
    sizes = {
        "coin_dir": 5,
        "crate_dir": 5,
        "escape_dir": 5,
        "danger_here": DANGER_CAP + 1,
        "neighbour_up": 3,
        "neighbour_right": 3,
        "neighbour_down": 3,
        "neighbour_left": 3,
        "bomb_available": 2,
        "crates_adjacent": 3,
        "opponent_near": 3,
        "escape_if_bomb": 2,
        "in_dead_end": 2,
        "opponent_in_dead_end": 2,
        "can_seal_opponent": 2,
    }
    return tuple(sizes[name] for name in config.feature_names)


def state_space_size(config: QLConfig) -> int:
    total = 1
    for size in feature_cardinalities(config):
        total *= size
    return total
