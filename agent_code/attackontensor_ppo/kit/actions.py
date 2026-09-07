# --------------------------------------------------------------------------
# GENERATED FILE -- DO NOT EDIT.
# Vendored from shared/kit/actions.py by tools/sync_kit.py.
# Edit the original, then re-run:  python tools/sync_kit.py
# --------------------------------------------------------------------------
"""Canonical action vocabulary.

Everything that indexes actions (Q-table columns, policy logits, action masks,
symmetry permutations) takes the ordering from here, so a Q-table column and a
policy logit cannot disagree about what index 3 means.

The direction encoding used across the kit is::

    0 = none / stay      1 = UP      2 = RIGHT      3 = DOWN      4 = LEFT

which is ``ACTIONS.index(name) + 1`` for the four movement actions, so
converting between the two is a +/-1.
"""

from __future__ import annotations

from typing import Dict, Tuple

# Order matches tpl_agent/callbacks.py so shared artifacts stay compatible.
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
N_ACTIONS = len(ACTIONS)

ACTION_INDEX: Dict[str, int] = {name: i for i, name in enumerate(ACTIONS)}

UP, RIGHT, DOWN, LEFT, WAIT, BOMB = range(N_ACTIONS)

MOVE_ACTIONS = (UP, RIGHT, DOWN, LEFT)

# The board uses image coordinates (x, y) with y growing downwards, exactly as
# environment.py indexes ``field[x, y]``. UP therefore decrements y.
ACTION_DELTAS: Tuple[Tuple[int, int], ...] = (
    (0, -1),  # UP
    (1, 0),  # RIGHT
    (0, 1),  # DOWN
    (-1, 0),  # LEFT
    (0, 0),  # WAIT
    (0, 0),  # BOMB
)

# Direction codes 1..4 -> (dx, dy). Index 0 is "no direction".
DIR_DELTAS: Tuple[Tuple[int, int], ...] = (
    (0, 0),  # 0 = none
    (0, -1),  # 1 = UP
    (1, 0),  # 2 = RIGHT
    (0, 1),  # 3 = DOWN
    (-1, 0),  # 4 = LEFT
)

N_DIRECTIONS = len(DIR_DELTAS)  # 5, including "none"

DELTA_TO_DIR: Dict[Tuple[int, int], int] = {
    delta: code for code, delta in enumerate(DIR_DELTAS) if code != 0
}


def dir_to_action(direction: int) -> int:
    """Map a direction code (1..4) to its action index. ``0`` maps to WAIT."""
    if direction == 0:
        return WAIT
    return direction - 1


def action_to_dir(action: int) -> int:
    """Map an action index to its direction code. Non-moves map to ``0``."""
    if action in MOVE_ACTIONS:
        return action + 1
    return 0


def step_from(x: int, y: int, action: int) -> Tuple[int, int]:
    """Tile the agent occupies after taking ``action`` from ``(x, y)``.

    Assumes the move is legal; legality is decided by :mod:`kit.geometry`.
    """
    dx, dy = ACTION_DELTAS[action]
    return x + dx, y + dy
