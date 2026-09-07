"""Action masking / safety filtering.

This module never chooses an action. It only removes ones that are provably
fatal, leaving the policy to pick among the survivors, so the agent stays a
learner rather than a rule-based player behind a network.

Three modes, so the filter can be ablated rather than assumed to help:

``none``
    No filtering. The agent may walk into walls and into blasts, and learns not
    to from the ``INVALID_ACTION`` penalty and from dying. Baseline for the
    ablation.
``soft`` (default)
    Removes only *certain, immediate* death: illegal moves, and moves onto a
    tile that is lethal at the end of this step. No search.
``hard``
    Adds a full time-indexed survival search, so it also removes actions from
    which no escape plan exists, including bomb drops with no way out. Given
    ``threats``, the search also assumes those tiles bomb on this step.

If every action is filtered out, death is unavoidable; the mask then falls back
to the legal moves so the caller still has something to pick from.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .actions import ACTION_DELTAS, BOMB, MOVE_ACTIONS, N_ACTIONS
from .geometry import in_bounds, lethal_at
from .pathfind import SURVIVAL_HORIZON, safe_actions

SAFETY_MODES: Tuple[str, ...] = ("none", "soft", "hard")

Coord = Tuple[int, int]


def legal_mask(
    field: np.ndarray,
    position: Coord,
    passable: np.ndarray,
    bomb_available: bool,
) -> np.ndarray:
    """Actions the framework will actually execute rather than score INVALID.

    Mirrors ``GenericWorld.perform_agent_action`` (environment.py:128): a move
    needs a free destination tile, ``BOMB`` needs the agent's bomb to be
    available, and ``WAIT`` is always accepted.
    """
    mask = np.ones(N_ACTIONS, dtype=bool)
    x, y = position

    for action in MOVE_ACTIONS:
        dx, dy = ACTION_DELTAS[action]
        nx, ny = x + dx, y + dy
        mask[action] = in_bounds(field, nx, ny) and bool(passable[nx, ny])

    mask[BOMB] = bool(bomb_available)
    return mask


def certain_death_mask(
    field: np.ndarray,
    position: Coord,
    danger: np.ndarray,
    passable: np.ndarray,
    bomb_available: bool,
) -> np.ndarray:
    """Legal actions that do not put us on a tile that detonates this step.

    Dropping a bomb or waiting leaves us where we are, so those are judged on
    the current tile. Note this looks exactly one step ahead: it will happily
    let the agent walk into a corridor it cannot escape later. That is what
    ``hard`` mode is for.
    """
    mask = legal_mask(field, position, passable, bomb_available)
    x, y = position

    for action in range(N_ACTIONS):
        if not mask[action]:
            continue
        dx, dy = ACTION_DELTAS[action]  # (0, 0) for WAIT and BOMB
        if lethal_at(danger, x + dx, y + dy, when=0):
            mask[action] = False

    return mask


def action_mask(
    field: np.ndarray,
    position: Coord,
    danger: np.ndarray,
    passable: np.ndarray,
    bomb_available: bool,
    mode: str = "soft",
    horizon: int = SURVIVAL_HORIZON,
    threats: Sequence[Coord] = (),
) -> np.ndarray:
    """Boolean mask over :data:`kit.actions.ACTIONS` for the requested mode.

    ``threats`` lists tiles to treat as bombing on this step, which only
    ``hard`` can act on because it is the only mode that plans ahead. It is
    supplied by the caller rather than derived here so the pessimism is a
    measurable switch rather than a property of the mode.
    """
    if mode not in SAFETY_MODES:
        raise ValueError(f"Unknown safety mode {mode!r}; choose from {SAFETY_MODES}")

    if mode == "none":
        return np.ones(N_ACTIONS, dtype=bool)

    legal = legal_mask(field, position, passable, bomb_available)

    if mode == "soft":
        mask = certain_death_mask(field, position, danger, passable, bomb_available)
    else:  # hard
        mask = safe_actions(
            field, position, danger, passable, bomb_available, horizon, threats
        )
        if threats and not mask.any():
            # Assuming every armed opponent bombs at once can leave nothing
            # admissible. Fall back to the optimistic search rather than
            # straight to `legal`: a plan that survives the visible bombs is
            # still better than no plan at all.
            mask = safe_actions(
                field, position, danger, passable, bomb_available, horizon
            )

    if mask.any():
        return mask
    # Doomed either way: hand back the legal moves rather than an empty choice.
    return legal if legal.any() else np.ones(N_ACTIONS, dtype=bool)


def masked_argmax(values: np.ndarray, mask: np.ndarray) -> int:
    """Argmax restricted to permitted actions, ignoring an all-False mask."""
    if mask is not None and mask.any():
        values = np.where(mask, values, -np.inf)
    return int(np.argmax(values))
