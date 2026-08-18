"""D4 symmetry tests.

The contract that matters: transforming the board and transforming the action
must describe the *same* physical move. If that ever breaks, augmentation
silently teaches the policy wrong labels, which is close to impossible to
diagnose from a learning curve.
"""

import numpy as np
import pytest

from shared.kit.actions import ACTION_DELTAS, MOVE_ACTIONS, N_ACTIONS
from shared.kit import symmetry as S


@pytest.mark.parametrize("name", S.TRANSFORMS)
def test_action_permutation_is_a_bijection(name):
    assert sorted(S.ACTION_PERMUTATION[name]) == list(range(N_ACTIONS))


@pytest.mark.parametrize("name", S.TRANSFORMS)
def test_wait_and_bomb_are_invariant(name):
    from shared.kit.actions import BOMB, WAIT

    assert S.transform_action(WAIT, name) == WAIT
    assert S.transform_action(BOMB, name) == BOMB


@pytest.mark.parametrize("name", S.TRANSFORMS)
@pytest.mark.parametrize("action", MOVE_ACTIONS)
def test_moving_then_transforming_equals_transforming_then_moving(name, action):
    """The core commutation property, checked on the board itself.

    Mark the agent at a tile and its destination on a grid, transform the grid,
    and confirm the transformed action still steps from the transformed origin
    to the transformed destination.
    """
    size = 7
    origin = (3, 2)
    dx, dy = ACTION_DELTAS[action]
    destination = (origin[0] + dx, origin[1] + dy)

    origin_grid = np.zeros((size, size), dtype=np.int8)
    origin_grid[origin] = 1
    destination_grid = np.zeros((size, size), dtype=np.int8)
    destination_grid[destination] = 1

    moved_origin = S.transform_grid(origin_grid, name)
    moved_destination = S.transform_grid(destination_grid, name)

    (ox,), (oy,) = np.nonzero(moved_origin)
    (tx,), (ty,) = np.nonzero(moved_destination)

    new_action = S.transform_action(action, name)
    ndx, ndy = ACTION_DELTAS[new_action]

    assert (int(ox) + ndx, int(oy) + ndy) == (int(tx), int(ty))


@pytest.mark.parametrize("name", S.TRANSFORMS)
def test_transform_preserves_shape_and_content(name):
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 5, size=(13, 9, 9)).astype(np.float32)

    out = S.transform_grid(obs, name)

    assert out.shape == obs.shape
    assert np.array_equal(np.sort(out, axis=None), np.sort(obs, axis=None))


def test_transform_action_array_moves_values_to_the_new_slots():
    values = np.arange(N_ACTIONS, dtype=np.float32)

    out = S.transform_action_array(values, "rot90")

    for action in range(N_ACTIONS):
        assert out[S.transform_action(action, "rot90")] == values[action]


def test_augment_yields_the_whole_orbit():
    rng = np.random.default_rng(1)
    obs = rng.random((3, 5, 5)).astype(np.float32)

    views = S.augment(obs, action=0)

    assert len(views) == S.N_TRANSFORMS
    assert all(grid.shape == obs.shape for grid, _ in views)


def test_canonical_form_is_orbit_invariant():
    """Every member of an orbit must canonicalise to the same representative.

    This is what makes Q-table canonicalisation sound: experience gathered in
    one corner is reused in the other three.
    """
    rng = np.random.default_rng(2)
    obs = rng.integers(0, 3, size=(2, 6, 6)).astype(np.int8)

    reference, _ = S.canonical_form(obs)

    for name in S.TRANSFORMS:
        rotated = S.transform_grid(obs, name)
        candidate, _ = S.canonical_form(rotated)
        assert np.array_equal(candidate, reference)
