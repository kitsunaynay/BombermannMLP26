"""Reward shaping primitives."""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

import events as e

from . import geometry as G
from . import pathfind as P

# --- custom event names ----------------------------------------------------
MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
ESCAPED_DANGER = "ESCAPED_DANGER"
ENTERED_DANGER = "ENTERED_DANGER"
WAITED_IN_DANGER = "WAITED_IN_DANGER"
USEFUL_BOMB = "USEFUL_BOMB"
USELESS_BOMB = "USELESS_BOMB"
SUICIDAL_BOMB = "SUICIDAL_BOMB"
SURVIVED_STEP = "SURVIVED_STEP"
TRAPPED_OPPONENT = "TRAPPED_OPPONENT"

CUSTOM_EVENTS = (
    MOVED_TOWARD_COIN,
    MOVED_AWAY_FROM_COIN,
    ESCAPED_DANGER,
    ENTERED_DANGER,
    WAITED_IN_DANGER,
    USEFUL_BOMB,
    USELESS_BOMB,
    SUICIDAL_BOMB,
    SURVIVED_STEP,
    TRAPPED_OPPONENT,
)

#: How close an opponent has to be for a body-block to count as ours.
TRAP_RADIUS = 2

#: Distance substituted when no target is reachable. Keeps the potential finite
#: and bounded instead of lurching as targets appear and disappear.
UNREACHABLE_DISTANCE = 30.0


def nearest_coin_distance(game_state: dict) -> float:
    """BFS distance to the closest collectable coin, or a bounded default."""
    field, position, _, _, _, coins, _, passable = P.game_state_context(game_state)
    if not coins:
        return UNREACHABLE_DISTANCE
    dist, _ = P.bfs(field, position, passable)
    found = P.distance_to_nearest(dist, coins)
    return UNREACHABLE_DISTANCE if found < 0 else float(found)


def potential(
    game_state: Optional[dict],
    coin_coefficient: float,
    crate_coefficient: float,
    danger_coefficient: float,
) -> float:
    """:math:`\\Phi(s)` -- a function of the state alone, never of the action.

    .. math::
       \\Phi(s) = -c_{\\text{coin}} d_{\\text{coin}}(s)
                  - c_{\\text{crate}} d_{\\text{crate}}(s)
                  - c_{\\text{danger}} \\mathbb{1}[\\text{in blast}]
    """
    if game_state is None:
        return 0.0

    field, position, _, _, _, coins, danger, passable = P.game_state_context(game_state)
    x, y = position
    dist, _ = P.bfs(field, position, passable)

    if coins:
        found = P.distance_to_nearest(dist, coins)
        coin_distance = UNREACHABLE_DISTANCE if found < 0 else float(found)
    else:
        coin_distance = UNREACHABLE_DISTANCE

    # Crates are only worth chasing once no coin is on the board, so the two
    # attraction terms never pull in opposite directions.
    crate_distance = UNREACHABLE_DISTANCE
    if not coins:
        approach = set()
        for cx, cy in G.crate_positions(field):
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = cx + dx, cy + dy
                if G.in_bounds(field, nx, ny) and dist[nx, ny] >= 0:
                    approach.add((nx, ny))
        if approach:
            found = P.distance_to_nearest(dist, sorted(approach))
            crate_distance = UNREACHABLE_DISTANCE if found < 0 else float(found)

    in_danger = 1.0 if danger[x, y] != G.SAFE else 0.0

    return -(
        coin_coefficient * coin_distance
        + crate_coefficient * crate_distance
        + danger_coefficient * in_danger
    )


def shaping_term(
    old_game_state: Optional[dict],
    new_game_state: Optional[dict],
    gamma: float,
    coin_coefficient: float,
    crate_coefficient: float,
    danger_coefficient: float,
) -> float:
    """:math:`\\gamma \\Phi(s') - \\Phi(s)`."""
    after = potential(new_game_state, coin_coefficient, crate_coefficient, danger_coefficient)
    before = potential(old_game_state, coin_coefficient, crate_coefficient, danger_coefficient)
    return gamma * after - before


def detect_custom_events(
    old_game_state: Optional[dict],
    self_action: Optional[str],
    new_game_state: Optional[dict],
    events: Sequence[str],
) -> List[str]:
    """Derive auxiliary events from a state transition.

    Returns a *new* list rather than mutating ``events``. The framework hands
    the same list object to ``game_events_occurred`` and then again to
    ``end_of_round`` (environment.py:174 and 490), so appending in place would
    make custom events pile up across both calls.
    """
    detected: List[str] = []
    if old_game_state is None:
        return detected

    field, old_position, _, _, others, old_coins, old_danger, old_passable = (
        P.game_state_context(old_game_state)
    )
    ox, oy = old_position
    was_in_danger = old_danger[ox, oy] != G.SAFE

    # --- bomb quality ------------------------------------------------------
    if self_action == "BOMB" and e.BOMB_DROPPED in events:
        blast = G.blast_coords(field, ox, oy)
        hits_crate = any(field[bx, by] == G.CRATE for bx, by in blast)
        hits_opponent = any(other in blast for other in others)

        if not P.has_escape_after_bomb(field, old_position, old_danger, old_passable):
            detected.append(SUICIDAL_BOMB)
        detected.append(USEFUL_BOMB if (hits_crate or hits_opponent) else USELESS_BOMB)

    # --- dithering while a bomb ticks -------------------------------------
    if was_in_danger and self_action in ("WAIT", None):
        detected.append(WAITED_IN_DANGER)

    if new_game_state is None:
        # The agent died this step, so there is no post-state to compare with.
        return detected

    new_field, new_position, _, _, new_others, _, new_danger, new_passable = (
        P.game_state_context(new_game_state)
    )
    nx, ny = new_position
    is_in_danger = new_danger[nx, ny] != G.SAFE

    # --- body-blocking an opponent inside a blast ---------------------------
    if new_others and trapped_opponents(new_field, new_position, new_others, new_danger, new_passable):
        detected.append(TRAPPED_OPPONENT)

    if was_in_danger and not is_in_danger:
        detected.append(ESCAPED_DANGER)
    elif not was_in_danger and is_in_danger:
        detected.append(ENTERED_DANGER)

    # --- coin approach, emitted symmetrically -----------------------------
    if old_coins and e.COIN_COLLECTED not in events:
        before = nearest_coin_distance(old_game_state)
        after = nearest_coin_distance(new_game_state)
        if after < before:
            detected.append(MOVED_TOWARD_COIN)
        elif after > before:
            detected.append(MOVED_AWAY_FROM_COIN)

    if e.GOT_KILLED not in events and e.KILLED_SELF not in events:
        detected.append(SURVIVED_STEP)

    return detected


def trapped_opponents(
    field,
    position,
    others: Sequence,
    danger,
    passable,
) -> List:
    """Opponents whose only way out of a blast is the tile we stand on.

    An opponent counts as trapped when it is within ``TRAP_RADIUS`` (Manhattan),
    its own tile is inside a live danger zone, and exactly one of its four
    neighbours is steppable -- and that neighbour is ``position``. ``passable``
    is ``free_mask(field, bombs, others)``, so bombs and other agents already
    block, while our own tile still reads as free.

    The danger condition matters: without it, standing next to a pocket would
    pay every step of a round, and the agent would learn to loiter instead of
    to bomb. Tied to a ticking bomb, the reward is bounded by the fuse.
    """
    px, py = position
    trapped = []
    for ox, oy in others:
        if abs(ox - px) + abs(oy - py) > TRAP_RADIUS:
            continue
        if not G.in_bounds(field, ox, oy) or danger[ox, oy] == G.SAFE:
            continue
        exits = [
            (ox + dx, oy + dy)
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
            if G.in_bounds(field, ox + dx, oy + dy) and passable[ox + dx, oy + dy]
        ]
        if len(exits) == 1 and exits[0] == (px, py):
            trapped.append((ox, oy))
    return trapped


def reward_from_events(events: Sequence[str], weights: Mapping[str, float]) -> float:
    """Sum the configured per-event rewards. Unknown events contribute zero."""
    return float(sum(weights.get(event, 0.0) for event in events))
