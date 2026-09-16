"""Navigation and survival tests."""

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


# ---------------------------------------------------------------------------
# Continuous survival features (route A: v4 escape-shield merge)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "length, bombs, action, start, expected_safe",
    [
        (5, [], RIGHT, (1, 1), True),  # exactly enough room, see BOMB_TIMER test above
        (5, [((1, 1), 1)], RIGHT, (3, 1), True),
        (5, [((1, 1), 1)], LEFT, (3, 1), False),
        (5, [((1, 1), 1)], WAIT, (3, 1), False),
    ],
)
def test_survival_profile_safety_verdict_matches_survives_after(
    length, bombs, action, start, expected_safe
):
    field = corridor(length + 2)
    passable = G.free_mask(field, bombs=bombs)
    danger = G.danger_map(field, bombs)

    profile = P.survival_profile(field, start, action, danger, passable)

    assert profile.safe == expected_safe
    assert profile.safe == P.survives_after(field, start, action, danger, passable)


def test_survival_profile_is_maximally_confident_with_no_danger():
    field = corridor(7)
    passable = G.free_mask(field)
    danger = G.danger_map(field, [])

    profile = P.survival_profile(field, (1, 1), RIGHT, danger, passable)

    assert profile.safe
    assert profile.duration == P.SURVIVAL_HORIZON
    assert profile.breadth == pytest.approx(1.0)
    assert profile.min_margin == pytest.approx(float(P.SURVIVAL_HORIZON))
    assert profile.contested == pytest.approx(0.0)


def test_survival_profile_margin_shrinks_near_a_ticking_bomb():
    """Standing right next to a soon-to-detonate bomb should certify a
    tighter margin than fleeing straight past its blast radius."""
    field = corridor(9)
    bombs = [((1, 1), 3)]
    passable = G.free_mask(field, bombs=bombs)
    danger = G.danger_map(field, bombs)

    near = P.survival_profile(field, (2, 1), RIGHT, danger, passable)
    far = P.survival_profile(field, (7, 1), RIGHT, danger, passable)

    assert near.safe and far.safe
    assert near.min_margin < far.min_margin


def test_survival_profile_rejects_an_illegal_move():
    field = corridor(5)
    passable = G.free_mask(field)
    danger = G.danger_map(field, [])

    profile = P.survival_profile(field, (1, 1), UP, danger, passable)  # wall above

    assert not profile.safe
    assert profile.duration == 0


def tiles_of(mask: np.ndarray) -> set:
    """Boolean mask -> set of ``(x, y)`` tuples, for set-style assertions."""
    return {tuple(xy) for xy in np.argwhere(mask)}


def test_opponent_reachability_starts_at_current_positions():
    field = np.full((6, 6), G.WALL, dtype=int)
    field[1:5, 1:5] = G.FREE

    reachable = P.opponent_reachability(field, [(2, 2)], horizon=3)

    assert tiles_of(reachable[0]) == {(2, 2)}
    assert len(reachable) == 4  # horizon + 1, including t=0


def test_opponent_reachability_grows_but_stays_within_walls():
    field = np.full((6, 6), G.WALL, dtype=int)
    field[1:5, 1:5] = G.FREE

    reachable = P.opponent_reachability(field, [(2, 2)], horizon=5)

    assert tiles_of(reachable[1]) == {(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)}
    # The 4x4 open pocket has 16 tiles; nothing outside it is ever reachable.
    pocket = {(x, y) for x in range(1, 5) for y in range(1, 5)}
    assert tiles_of(reachable[-1]) <= pocket


def test_survival_profile_contested_reflects_overlapping_opponent_reach():
    """Fleeing a bomb (so the search actually walks the frontier forward,
    rather than certifying safety on the spot) past a tile a nearby opponent
    could also reach should read as contested; a distant opponent should not
    move the needle."""
    field = corridor(7)
    bombs = [((1, 1), 1)]
    passable = G.free_mask(field, bombs=bombs)
    danger = G.danger_map(field, bombs)
    start = (3, 1)

    nearby_opponent = P.opponent_reachability(field, [(6, 1)])
    contested = P.survival_profile(
        field, start, RIGHT, danger, passable, opponent_reachable=nearby_opponent
    )

    far_opponent = P.opponent_reachability(field, [(200, 200)])
    uncontested = P.survival_profile(
        field, start, RIGHT, danger, passable, opponent_reachable=far_opponent
    )

    assert contested.contested > uncontested.contested
    assert uncontested.contested == pytest.approx(0.0)



# --------------------------------------------------------------------------
# Robust bomb gate (kit.safety.BOMB_GATES)
# --------------------------------------------------------------------------


def test_survival_profile_reports_exit_step_and_terminal_width():
    from shared.kit import safety

    # Corridor x=1..9. Bombing at x=1 leaves one way out, along the corridor.
    field = corridor(9)
    passable = G.free_mask(field)
    danger = G.danger_map(field, [])
    profile = P.survival_profile(field, (1, 1), BOMB, danger, passable)
    assert profile.safe
    assert profile.exit_step == s.BOMB_POWER + 1  # the first tile beyond the blast
    assert profile.terminal_width == 1

    # Open room: many end tiles, out of the blast just as quickly.
    room = np.full((9, 9), G.WALL, dtype=int)
    room[1:8, 1:8] = G.FREE
    passable = G.free_mask(room)
    danger = G.danger_map(room, [])
    profile = P.survival_profile(room, (4, 4), BOMB, danger, passable)
    assert profile.safe
    assert profile.exit_step <= safety.ROBUST_EXIT_STEP
    assert profile.terminal_width >= safety.ROBUST_TERMINAL_WIDTH


def test_robust_gate_forbids_a_single_file_escape_but_not_an_open_one():
    from shared.kit import safety

    field = corridor(9)
    passable = G.free_mask(field)
    danger = G.danger_map(field, [])
    escape = safety.action_mask(field, (1, 1), danger, passable, True, mode="hard")
    robust = safety.action_mask(
        field, (1, 1), danger, passable, True, mode="hard", bomb_gate="robust"
    )
    assert escape[BOMB], "a corridor escape is still an escape"
    assert not robust[BOMB], "but it is not a redundant one"
    # Only BOMB differs: the gate never touches movement.
    assert np.array_equal(escape[:BOMB], robust[:BOMB])

    room = np.full((9, 9), G.WALL, dtype=int)
    room[1:8, 1:8] = G.FREE
    passable = G.free_mask(room)
    danger = G.danger_map(room, [])
    robust = safety.action_mask(
        room, (4, 4), danger, passable, True, mode="hard", bomb_gate="robust"
    )
    assert robust[BOMB]


def test_bomb_gate_is_ignored_outside_hard_mode_and_validated():
    from shared.kit import safety

    field = corridor(9)
    passable = G.free_mask(field)
    danger = G.danger_map(field, [])
    soft = safety.action_mask(field, (1, 1), danger, passable, True, mode="soft", bomb_gate="robust")
    assert soft[BOMB]
    with pytest.raises(ValueError):
        safety.action_mask(field, (1, 1), danger, passable, True, mode="hard", bomb_gate="strict")
