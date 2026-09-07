"""Discrete feature extraction for the tabular Q-learning agent.

Three kinds of feature: situational awareness (``neighbour_*``), pathfinding
(``coin_dir``, ``crate_dir``) and life-saving (``danger_here``, ``escape_dir``,
``escape_if_bomb``).

Every feature is a small non-negative integer, so a state is a tuple of ints and
can key a dictionary directly. Cardinalities::

    coin_dir         5   direction code 0..4
    crate_dir        5
    escape_dir       5
    danger_here      5   0 = lethal now ... 4 = safe
    neighbour_*      3   each: 0 free, 1 blocked, 2 lethal
    bomb_available   2
    crates_adjacent  3   0, 1, >=2
    opponent_near    3   none / within 3 / adjacent
    escape_if_bomb   2

The full variant spans about 1.8M states but only reachable ones get allocated,
in practice 10^4..10^5. The compact variant is what Tasks 1-2 train on.

``state_to_features`` is pure and deterministic: the BFS in :mod:`kit.pathfind`
expands neighbours in a fixed order.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .config import QLConfig
from .kit import geometry as G
from .kit import pathfind as P
from .kit import symmetry
from .kit.actions import DIR_DELTAS

#: Neighbour tile encoding.
NB_FREE, NB_BLOCKED, NB_LETHAL = 0, 1, 2

#: ``danger_here`` saturates here; 4 means "no known threat".
DANGER_CAP = 4


def _neighbour_code(
    field: np.ndarray,
    passable: np.ndarray,
    danger: np.ndarray,
    x: int,
    y: int,
) -> int:
    """Classify the tile in one direction as free, blocked, or lethal."""
    if not G.in_bounds(field, x, y):
        return NB_BLOCKED
    if not passable[x, y]:
        return NB_BLOCKED
    if G.lethal_at(danger, x, y, when=0):
        return NB_LETHAL
    return NB_FREE


def _opponent_proximity(position: Tuple[int, int], others: Sequence[Tuple[int, int]]) -> int:
    """0 = none nearby, 1 = within three tiles, 2 = orthogonally adjacent."""
    if not others:
        return 0
    x, y = position
    closest = min(abs(ox - x) + abs(oy - y) for ox, oy in others)
    if closest <= 1:
        return 2
    if closest <= 3:
        return 1
    return 0


def compute_all_features(game_state: dict) -> Dict[str, int]:
    """Every feature, keyed by name, before the variant selects a subset.

    Kept separate from :func:`state_to_features` so diagnostics and tests can
    inspect individual features, and so ablations cost nothing at runtime.
    """
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

    # Navigation. Bombs and opponents block movement, so BFS runs on `passable`.
    dist, first_step = P.bfs(field, position, passable)
    coin_dir = P.direction_to_nearest(dist, first_step, coins)

    # Crates are not walkable, so we path toward the free tiles beside them.
    crate_dir = 0
    crates = G.crate_positions(field)
    if crates:
        approach = set()
        for cx, cy in crates:
            for dx, dy in DIR_DELTAS[1:]:
                nx, ny = cx + dx, cy + dy
                if G.in_bounds(field, nx, ny) and dist[nx, ny] >= 0:
                    approach.add((nx, ny))
        # Sorted, not raw set order: iteration order over a set of tuples is a
        # hash artefact, which would make tie-breaking between equidistant
        # crates arbitrary and irreproducible.
        crate_dir = P.direction_to_nearest(dist, first_step, sorted(approach))

    danger_here = int(min(danger[x, y], DANGER_CAP))
    escape_dir = P.escape_direction(field, position, danger, passable) if danger_here < DANGER_CAP else 0

    neighbours = {}
    for name, direction in (
        ("neighbour_up", 1),
        ("neighbour_right", 2),
        ("neighbour_down", 3),
        ("neighbour_left", 4),
    ):
        dx, dy = DIR_DELTAS[direction]
        neighbours[name] = _neighbour_code(field, passable, danger, x + dx, y + dy)

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
    }


# ---------------------------------------------------------------------------
# D4 canonicalisation
# ---------------------------------------------------------------------------
# Map each state to a fixed representative of its symmetry orbit. Shrinks the
# table by up to 8x and shares experience between the four corners. The action
# comes back in the canonical frame and needs the inverse transform applied.
#
# Canonicalisation acts on the feature tuple, and the direction features are
# exactly equivariant only when the nearest target is unique. Equidistant
# targets need a tie-break, and no deterministic rule breaks a symmetric tie
# symmetrically, so mirrored boards can land on different keys. That costs some
# sharing but never correctness: key and transform are computed together.
# tests/test_ql_agent.py covers both cases.

#: Direction-valued features, which rotate with the board.
DIRECTION_FEATURES = ("coin_dir", "crate_dir", "escape_dir")

#: Neighbour feature for each direction code.
NEIGHBOUR_BY_DIR = {
    1: "neighbour_up",
    2: "neighbour_right",
    3: "neighbour_down",
    4: "neighbour_left",
}


def _transform_values(values: Dict[str, int], transform: str) -> Dict[str, int]:
    """Re-express features as seen from a D4-transformed board."""
    out = dict(values)

    for name in DIRECTION_FEATURES:
        if name in values:
            out[name] = symmetry.transform_direction(values[name], transform)

    # A neighbour observed in direction d is observed in direction T(d) after
    # the transform. Reads come from `values` so the four writes cannot alias.
    for direction, name in NEIGHBOUR_BY_DIR.items():
        moved = symmetry.transform_direction(direction, transform)
        out[NEIGHBOUR_BY_DIR[moved]] = values[name]

    return out


class FeatureView(NamedTuple):
    """A canonicalised state key plus the transform used to reach it.

    ``transform`` is what maps the *real* board into the canonical frame, so an
    action chosen against ``key`` must be pushed through
    :func:`kit.symmetry.inverse_transform_action` before being played.
    """

    key: Tuple[int, ...]
    transform: str


def extract(
    game_state: Optional[dict],
    config: Optional[QLConfig] = None,
) -> Optional[FeatureView]:
    """Feature key for a game state, canonicalised when the config asks for it."""
    if game_state is None:
        return None

    config = config or QLConfig.load()
    names = config.feature_names
    values = compute_all_features(game_state)

    if not config.use_symmetry:
        return FeatureView(tuple(values[name] for name in names), "identity")

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
    """Convert a game state into a hashable discrete feature tuple.

    Returns ``None`` for a missing state. The framework hands out ``None`` for
    dead agents (environment.py:397) and both training callbacks can receive it,
    so every caller must tolerate it.
    """
    view = extract(game_state, config)
    return None if view is None else view.key


def feature_cardinalities(config: QLConfig) -> Tuple[int, ...]:
    """Per-feature cardinality, for reporting the theoretical state-space size."""
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
    }
    return tuple(sizes[name] for name in config.feature_names)


def state_space_size(config: QLConfig) -> int:
    """Theoretical upper bound on distinct states for a variant."""
    total = 1
    for size in feature_cardinalities(config):
        total *= size
    return total
