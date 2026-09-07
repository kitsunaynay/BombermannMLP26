"""Dihedral (D4) symmetries of the arena.

The Bomberman board is symmetric under the eight-element dihedral group: four
rotations and four reflections. Two uses:

* **Data augmentation** for PPO: one rollout step becomes up to eight.
* **State canonicalisation** for the Q-table: mapping each state to a fixed
  representative of its orbit shrinks the table by up to 8x and shares
  experience between the four corners.

Grids are indexed ``[x, y]`` to match ``game_state['field']``; stacked tensors
are ``(C, X, Y)``. The action permutation for each transform is *derived* at
import time by probing the transform with a one-hot marker rather than being
written out by hand, so it cannot disagree with the grid operation.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from .actions import (
    ACTION_DELTAS,
    DELTA_TO_DIR,
    MOVE_ACTIONS,
    N_ACTIONS,
    dir_to_action,
)

#: Names of the eight group elements, in a fixed order.
TRANSFORMS: Tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_x",
    "flip_y",
    "transpose",
    "anti_transpose",
)

N_TRANSFORMS = len(TRANSFORMS)


def _grid_op(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return the array operation for a transform, acting on the last two axes."""
    if name == "identity":
        return lambda a: a
    if name == "rot90":
        return lambda a: np.rot90(a, k=1, axes=(-2, -1))
    if name == "rot180":
        return lambda a: np.rot90(a, k=2, axes=(-2, -1))
    if name == "rot270":
        return lambda a: np.rot90(a, k=3, axes=(-2, -1))
    if name == "flip_x":
        return lambda a: np.flip(a, axis=-2)
    if name == "flip_y":
        return lambda a: np.flip(a, axis=-1)
    if name == "transpose":
        return lambda a: np.swapaxes(a, -2, -1)
    if name == "anti_transpose":
        return lambda a: np.rot90(np.swapaxes(a, -2, -1), k=2, axes=(-2, -1))
    raise ValueError(f"Unknown transform {name!r}")


def transform_grid(grid: np.ndarray, transform: str) -> np.ndarray:
    """Apply a D4 transform to a ``(..., X, Y)`` array. Returns a contiguous copy."""
    return np.ascontiguousarray(_grid_op(transform)(grid))


def _derive_action_permutation(transform: str) -> Tuple[int, ...]:
    """Work out how a transform permutes actions, by probing it.

    A 3x3 grid with a single marker one step away from the centre is pushed
    through the grid operation; wherever the marker lands is the transformed
    direction. Deriving this empirically means a sign error in ``_grid_op``
    surfaces as a failing symmetry test instead of a silently wrong mapping.
    """
    op = _grid_op(transform)
    permutation: List[int] = list(range(N_ACTIONS))

    for action in MOVE_ACTIONS:
        dx, dy = ACTION_DELTAS[action]
        probe = np.zeros((3, 3), dtype=np.int8)
        probe[1 + dx, 1 + dy] = 1

        moved = op(probe)
        (nx,), (ny,) = np.nonzero(moved)
        new_delta = (int(nx) - 1, int(ny) - 1)
        permutation[action] = dir_to_action(DELTA_TO_DIR[new_delta])

    # WAIT and BOMB are rotation-invariant and keep their indices.
    return tuple(permutation)


#: ``ACTION_PERMUTATION[name][a]`` is the index action ``a`` becomes under ``name``.
ACTION_PERMUTATION: Dict[str, Tuple[int, ...]] = {
    name: _derive_action_permutation(name) for name in TRANSFORMS
}


def _invert(permutation: Tuple[int, ...]) -> Tuple[int, ...]:
    inverse = [0] * len(permutation)
    for source, destination in enumerate(permutation):
        inverse[destination] = source
    return tuple(inverse)


#: Undoes :data:`ACTION_PERMUTATION`. Needed when a policy is queried in a
#: canonicalised frame and its choice has to be replayed on the real board.
INVERSE_ACTION_PERMUTATION: Dict[str, Tuple[int, ...]] = {
    name: _invert(permutation) for name, permutation in ACTION_PERMUTATION.items()
}


def transform_action(action: int, transform: str) -> int:
    """Map an action index through a D4 transform."""
    return ACTION_PERMUTATION[transform][action]


def inverse_transform_action(action: int, transform: str) -> int:
    """Map an action index back from a transformed frame to the real board."""
    return INVERSE_ACTION_PERMUTATION[transform][action]


def transform_direction(direction: int, transform: str) -> int:
    """Map a direction code (0..4) through a D4 transform. ``0`` is fixed."""
    if direction == 0:
        return 0
    from .actions import action_to_dir, dir_to_action

    return action_to_dir(transform_action(dir_to_action(direction), transform))


def transform_action_array(values: np.ndarray, transform: str) -> np.ndarray:
    """Permute a per-action vector (logits, Q-values, masks) through a transform.

    ``out[transform_action(a)] == values[a]``, i.e. each entry moves to the slot
    its action maps to.
    """
    permutation = ACTION_PERMUTATION[transform]
    out = np.empty_like(values)
    for action in range(N_ACTIONS):
        out[permutation[action]] = values[action]
    return out


def augment(obs: np.ndarray, action: int) -> List[Tuple[np.ndarray, int]]:
    """All eight symmetric views of an ``(C, X, Y)`` observation/action pair."""
    return [
        (transform_grid(obs, name), transform_action(action, name)) for name in TRANSFORMS
    ]


def canonical_form(obs: np.ndarray) -> Tuple[np.ndarray, str]:
    """Pick a deterministic representative of an observation's D4 orbit.

    The representative is the transform whose flattened byte string sorts first,
    which is stable and independent of how the state was reached.
    """
    best_name = TRANSFORMS[0]
    best_grid = transform_grid(obs, best_name)
    best_key = best_grid.tobytes()

    for name in TRANSFORMS[1:]:
        candidate = transform_grid(obs, name)
        key = candidate.tobytes()
        if key < best_key:
            best_name, best_grid, best_key = name, candidate, key

    return best_grid, best_name
