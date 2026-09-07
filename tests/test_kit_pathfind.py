"""Navigation and survival tests.

The survival cases are hand-built corridors where the correct answer is
provable by counting moves, so they pin down the *timing* arithmetic -- the
part of the agent that decides whether it lives or dies.
"""

from collections import deque

import numpy as np
import pytest

import settings as s
from helpers import build_arena
from shared.kit import geometry as G
from shared.kit import pathfind as P
from shared.kit.actions import BOMB, DOWN, LEFT, RIGHT, UP, WAIT


def corridor(length: int, width: int = 12) -> np.ndarray:
    """Walled box with a horizontal corridor at y=1 spanning x=1..length."""
    field = np.full((width, 6), G.WALL, dtype=int)
    field[1 : length + 1, 1] = G.FREE
    return field


def brute_force_distances(field, start, passable):
    """Independent BFS reference implementation for cross-checking."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if (nx, ny) in dist or not passable[nx, ny]:
                continue
            dist[(nx, ny)] = dist[(x, y)] + 1
            queue.append((nx, ny))
    return dist


@pytest.mark.parametrize("seed", range(5))
def test_bfs_matches_brute_force(seed):
    field = build_arena(seed)
    passable = G.free_mask(field)
    start = (1, 1)

    dist, _ = P.bfs(field, start, passable)
    reference = brute_force_distances(field, start, passable)

    for (x, y), expected in reference.items():
        assert dist[x, y] == expected
    reachable = int((dist >= 0).sum())
    assert reachable == len(reference)


def test_bfs_first_step_is_deterministic_and_on_a_shortest_path():
    field = corridor(6)
    passable = G.free_mask(field)

    dist, first_step = P.bfs(field, (1, 1), passable)

    assert dist[5, 1] == 4
    assert first_step[5, 1] == 2, "must open with RIGHT (direction code 2)"

    # Repeated calls must agree; rule_based_agent's look_for_targets does not.
    for _ in range(5):
        assert np.array_equal(P.bfs(field, (1, 1), passable)[1], first_step)


def test_direction_to_nearest_prefers_the_closer_target():
    field = corridor(8)
    passable = G.free_mask(field)
    dist, first_step = P.bfs(field, (4, 1), passable)

    assert P.direction_to_nearest(dist, first_step, [(7, 1), (3, 1)]) == 4  # LEFT
    assert P.direction_to_nearest(dist, first_step, [(6, 1), (8, 1)]) == 2  # RIGHT
    assert P.direction_to_nearest(dist, first_step, []) == 0
    assert P.direction_to_nearest(dist, first_step, [(4, 1)]) == 0, "already there"


def test_distance_to_nearest_reports_unreachable():
    field = corridor(4)
    passable = G.free_mask(field)
    dist, _ = P.bfs(field, (1, 1), passable)

    assert P.distance_to_nearest(dist, [(3, 1)]) == 2
    assert P.distance_to_nearest(dist, [(9, 1)]) == -1


# --------------------------------------------------------------------------
# Survival: the timing-critical half.
# --------------------------------------------------------------------------


def test_bomb_escape_possible_with_exactly_enough_room():
    """Blast covers x=1..4; the agent needs BOMB_TIMER moves to reach x=5.

    The framework gives exactly BOMB_TIMER follow-up moves after the drop step,
    so this is the tightest survivable case and must come out True.
    """
    field = corridor(5)
    passable = G.free_mask(field)
    danger = G.danger_map(field, [])

    assert s.BOMB_TIMER == 4, "test is calibrated to the stock bomb timer"
    assert P.has_escape_after_bomb(field, (1, 1), danger, passable)


def test_bomb_escape_impossible_when_the_blast_fills_the_corridor():
    field = corridor(4)  # every reachable tile lies inside the blast
    passable = G.free_mask(field)
    danger = G.danger_map(field, [])

    assert not P.has_escape_after_bomb(field, (1, 1), danger, passable)


def test_survival_requires_running_the_right_way():
    """Bomb at x=1 with timer 1; the agent at x=3 must flee right, not left.

    Fleeing left keeps it inside the blast when the bomb goes off, and waiting
    is fatal too -- a plain "is my tile dangerous" check cannot tell these apart.
    """
    field = corridor(7)
    bombs = [((1, 1), 1)]
    passable = G.free_mask(field, bombs=bombs)
    danger = G.danger_map(field, bombs)
    start = (3, 1)

    assert P.survives_after(field, start, RIGHT, danger, passable)
    assert not P.survives_after(field, start, LEFT, danger, passable)
    assert not P.survives_after(field, start, WAIT, danger, passable)


def test_safe_actions_excludes_illegal_moves():
    field = corridor(5)
    passable = G.free_mask(field)
    danger = G.danger_map(field, [])

    mask = P.safe_actions(field, (1, 1), danger, passable)

    assert not mask[UP] and not mask[DOWN] and not mask[LEFT], "walls on three sides"
    assert mask[RIGHT] and mask[WAIT]


def test_safe_actions_respects_bomb_availability():
    field = corridor(5)
    passable = G.free_mask(field)
    danger = G.danger_map(field, [])

    assert P.safe_actions(field, (1, 1), danger, passable, bomb_available=True)[BOMB]
    assert not P.safe_actions(field, (1, 1), danger, passable, bomb_available=False)[BOMB]


def test_safe_actions_is_empty_when_death_is_certain():
    """A sealed pocket already inside a blast: nothing can save the agent.

    The mask must be allowed to go all-False -- callers fall back to the raw
    policy rather than assuming at least one action survives.
    """
    field = np.full((6, 6), G.WALL, dtype=int)
    field[1:3, 1] = G.FREE
    bombs = [((2, 1), 0)]
    passable = G.free_mask(field, bombs=bombs)
    danger = G.danger_map(field, bombs)

    assert not P.safe_actions(field, (1, 1), danger, passable).any()


def test_escape_direction_points_away_from_danger():
    field = corridor(7)
    bombs = [((1, 1), 2)]
    passable = G.free_mask(field, bombs=bombs)
    danger = G.danger_map(field, bombs)

    assert P.escape_direction(field, (3, 1), danger, passable) == 2  # RIGHT
    # Standing somewhere no bomb reaches, there is nothing to escape from.
    assert P.escape_direction(field, (6, 1), danger, passable) == 0


def test_survival_fast_path_when_no_bombs_exist():
    field = corridor(7)
    passable = G.free_mask(field)
    danger = G.danger_map(field, [])

    mask = P.safe_actions(field, (3, 1), danger, passable)

    assert mask[RIGHT] and mask[LEFT] and mask[WAIT]
    assert mask[BOMB], "corridor is long enough to run out of our own blast"


# ---------------------------------------------------------------------------
# Opponent-bomb lookahead (M2)
# ---------------------------------------------------------------------------


def test_threats_are_ignored_when_the_list_is_empty():
    """The default path must be byte-identical to the pre-M2 behaviour."""
    field = corridor(9)
    danger = np.full(field.shape, G.SAFE, dtype=int)
    passable = P.free_mask(field, [], [])

    without = P.safe_actions(field, (1, 1), danger, passable, True)
    with_empty = P.safe_actions(field, (1, 1), danger, passable, True, threats=())

    assert np.array_equal(without, with_empty)


def test_threat_blast_reaches_the_same_tiles_as_a_real_bomb():
    field = corridor(9)
    danger = np.full(field.shape, G.SAFE, dtype=int)

    threatened = P.augment_danger_with_threats(field, danger, [(5, 1)])
    own = P._augment_danger_with_bomb(field, danger, 5, 1)

    assert np.array_equal(threatened, own)


def test_threats_do_not_mutate_the_caller_s_danger_map():
    field = corridor(9)
    danger = np.full(field.shape, G.SAFE, dtype=int)
    original = danger.copy()

    P.augment_danger_with_threats(field, danger, [(5, 1)])

    assert np.array_equal(danger, original)


def test_an_out_of_range_threat_changes_nothing():
    field = corridor(9)
    danger = np.full(field.shape, G.SAFE, dtype=int)

    # Far enough that its blast cannot reach the corridor we occupy.
    augmented = P.augment_danger_with_threats(field, danger, [(9, 1)])

    assert augmented[1, 1] == G.SAFE


def test_a_threat_can_make_a_bomb_drop_unsurvivable():
    """A dead-end short enough that our bomb alone is escapable, but not with
    an opponent bombing the mouth of it at the same moment."""
    field = corridor(s.BOMB_POWER + 3)
    danger = np.full(field.shape, G.SAFE, dtype=int)
    passable = P.free_mask(field, [], [])
    start = (1, 1)

    assert P.has_escape_after_bomb(field, start, danger, passable)

    mouth = (s.BOMB_POWER + 3, 1)
    assert not P.has_escape_after_bomb(
        field, start, danger, passable, threats=[mouth]
    )


def test_armed_opponents_excludes_those_with_a_bomb_out():
    state = {
        "others": [
            ("a", 0, True, (3, 3)),
            ("b", 0, False, (5, 5)),
            ("c", 0, True, (7, 7)),
        ]
    }

    assert P.armed_opponents(state) == [(3, 3), (7, 7)]


def test_hard_mask_falls_back_to_the_optimistic_search_when_threats_seal_everything():
    """Assuming every armed opponent bombs at once can admit nothing. The mask
    must then return the plan that survives the *visible* bombs, not collapse
    all the way to the legal moves."""
    from shared.kit import safety

    field = corridor(s.BOMB_POWER + 3)
    danger = np.full(field.shape, G.SAFE, dtype=int)
    passable = P.free_mask(field, [], [])
    start = (1, 1)

    # Threats from both ends leave no admissible tile in the corridor.
    threats = [(2, 1), (s.BOMB_POWER + 3, 1)]
    pessimistic = P.safe_actions(field, start, danger, passable, True, threats=threats)
    assert not pessimistic.any()

    optimistic = P.safe_actions(field, start, danger, passable, True)
    masked = safety.action_mask(
        field, start, danger, passable, True, mode="hard", threats=threats
    )

    assert np.array_equal(masked, optimistic)
    assert masked.any()
