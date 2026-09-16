"""Action masking / safety filtering."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .actions import ACTION_DELTAS, BOMB, MOVE_ACTIONS, N_ACTIONS
from .geometry import in_bounds, lethal_at
from .pathfind import SURVIVAL_HORIZON, safe_actions, survival_profile

SAFETY_MODES: Tuple[str, ...] = ("none", "soft", "hard")

#: How ``hard`` mode judges a bomb drop.
#:
#: ``escape``  any certified escape plan will do (one surviving tile suffices).
#: ``robust``  the plan must also be *redundant*: out of the bomb's own blast
#:             within ``ROBUST_EXIT_STEP`` moves, at least
#:             ``ROBUST_TERMINAL_WIDTH`` distinct end tiles, and breadth of at
#:             least ``ROBUST_BREADTH`` along the way. A single-tile escape
#:             route is exactly what a second bomb, or a body in the corridor,
#:             closes one step later -- Phase 12 measured that closing as the
#:             cause of 73% of the agent's own deaths. The thresholds follow
#:             the shield of ``survival_linear_ppo_v4``, whose 1v1 suicide
#:             rate was lower than ours with an otherwise similar search.
BOMB_GATES: Tuple[str, ...] = ("escape", "robust")
ROBUST_EXIT_STEP = 3
ROBUST_TERMINAL_WIDTH = 2
ROBUST_BREADTH = 0.2

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
    bomb_gate: str = "escape",
) -> np.ndarray:
    """Boolean mask over :data:`kit.actions.ACTIONS` for the requested mode.

    ``threats`` lists tiles to treat as bombing on this step, which only
    ``hard`` can act on because it is the only mode that plans ahead. It is
    supplied by the caller rather than derived here so the pessimism is a
    measurable switch rather than a property of the mode. ``bomb_gate`` (see
    :data:`BOMB_GATES`) likewise only applies to ``hard``.
    """
    if mode not in SAFETY_MODES:
        raise ValueError(f"Unknown safety mode {mode!r}; choose from {SAFETY_MODES}")
    if bomb_gate not in BOMB_GATES:
        raise ValueError(f"Unknown bomb gate {bomb_gate!r}; choose from {BOMB_GATES}")

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
        if bomb_gate == "robust" and mask[BOMB]:
            mask[BOMB] = robust_bomb_ok(
                field, position, danger, passable, horizon, threats
            )

    if mask.any():
        return mask
    # Doomed either way: hand back the legal moves rather than an empty choice.
    return legal if legal.any() else np.ones(N_ACTIONS, dtype=bool)


def robust_bomb_ok(
    field: np.ndarray,
    position: Coord,
    danger: np.ndarray,
    passable: np.ndarray,
    horizon: int = SURVIVAL_HORIZON,
    threats: Sequence[Coord] = (),
) -> bool:
    """Whether dropping a bomb here leaves a *redundant* escape, not just one."""
    profile = survival_profile(
        field, position, BOMB, danger, passable, horizon=horizon, threats=threats
    )
    if not profile.safe:
        return False
    return (
        0 <= profile.exit_step <= ROBUST_EXIT_STEP
        and profile.terminal_width >= ROBUST_TERMINAL_WIDTH
        and profile.breadth >= ROBUST_BREADTH
    )


def masked_argmax(values: np.ndarray, mask: np.ndarray) -> int:
    """Argmax restricted to permitted actions, ignoring an all-False mask."""
    if mask is not None and mask.any():
        values = np.where(mask, values, -np.inf)
    return int(np.argmax(values))
