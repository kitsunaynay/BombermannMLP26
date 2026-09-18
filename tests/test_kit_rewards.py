"""kit.rewards: the body-block event."""

import numpy as np
import pytest

from agent_code.attackontensor_ppo.kit import geometry as G
from agent_code.attackontensor_ppo.kit import rewards as R
from agent_code.attackontensor_ppo.kit.pathfind import game_state_context


def pocket_field():
    """A 7x7 walled box with a one-tile dead end at (1, 1) opening onto (2, 1).

        # # # # # # #
        # o . . . . #     o = pocket tile (1, 1), exit (2, 1)
        # # # . # . #
        # . . . . . #
        # # # . # . #
        # . . . . . #
        # # # # # # #
    """
    f = np.full((7, 7), G.WALL, dtype=int)
    for x in range(1, 6):
        f[x, 1] = G.FREE
        f[x, 3] = G.FREE
        f[x, 5] = G.FREE
    f[3, 2] = f[5, 2] = f[3, 4] = f[5, 4] = G.FREE
    f[1, 2] = G.WALL  # the pocket has exactly one neighbour, (2, 1)
    return f


def state(field, me, others=(), bombs=()):
    return {
        "round": 1, "step": 5,
        "field": field,
        "self": ("me", 0, True, me),
        "others": [("o%d" % i, 0, True, xy) for i, xy in enumerate(others)],
        "bombs": list(bombs),
        "coins": [],
        "explosion_map": np.zeros_like(field),
        "user_input": None,
    }


def trapped(field, me, others, bombs):
    f, pos, _, _, oth, _, danger, passable = game_state_context(state(field, me, others, bombs))
    return R.trapped_opponents(f, pos, oth, danger, passable)


def test_sealing_a_pocket_with_a_bomb_ticking_is_a_trap():
    field = pocket_field()
    # Our bomb at (3, 1) covers the corridor including the pocket; we stand on
    # its single exit.
    assert trapped(field, me=(2, 1), others=[(1, 1)], bombs=[((3, 1), 3)]) == [(1, 1)]


def test_no_danger_means_no_trap():
    """Standing beside a pocket without a bomb is loitering, not a kill setup."""
    field = pocket_field()
    assert trapped(field, me=(2, 1), others=[(1, 1)], bombs=[]) == []


def test_a_second_exit_means_no_trap():
    field = pocket_field()
    field[1, 2] = G.FREE  # open a second way out of the pocket
    assert trapped(field, me=(2, 1), others=[(1, 1)], bombs=[((3, 1), 3)]) == []


def test_only_counts_when_we_are_the_blocker():
    field = pocket_field()
    # Same geometry and bomb, but we stand two tiles away: the exit is open.
    assert trapped(field, me=(3, 3), others=[(1, 1)], bombs=[((3, 1), 3)]) == []


def test_far_opponents_are_ignored():
    field = pocket_field()
    field[1, 2] = G.WALL
    # A distant pocket sealed by a bomb, not by us.
    assert trapped(field, me=(5, 5), others=[(1, 1)], bombs=[((2, 1), 3)]) == []


def test_event_flows_through_detect_custom_events():
    field = pocket_field()
    old = state(field, me=(3, 1), others=[(1, 1)], bombs=[])
    new = state(field, me=(2, 1), others=[(1, 1)], bombs=[((3, 1), 3)])
    events = R.detect_custom_events(old, "LEFT", new, ["MOVED_LEFT"])
    assert R.TRAPPED_OPPONENT in events
    assert R.TRAPPED_OPPONENT in R.CUSTOM_EVENTS


@pytest.mark.parametrize("weights", [{}, {"TRAPPED_OPPONENT": 0.0}])
def test_zero_or_missing_weight_pays_nothing(weights):
    assert R.reward_from_events([R.TRAPPED_OPPONENT], weights) == 0.0
    assert R.reward_from_events([R.TRAPPED_OPPONENT], {"TRAPPED_OPPONENT": 1.5}) == 1.5
