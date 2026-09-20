"""Tests for the multi-seed ablation aggregator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from aggregate_seeds import mean, stdev, two_sided_p, welch  # noqa: E402


#: (t, dof) pairs at exactly the two-sided 5% critical value, from a t-table.
CRITICAL_VALUES = [(3.182, 3), (2.776, 4), (2.228, 10), (2.086, 20), (1.960, 10_000)]


@pytest.mark.parametrize("t, dof", CRITICAL_VALUES)
def test_p_value_matches_the_t_table_at_five_percent(t, dof):
    assert two_sided_p(t, dof) == pytest.approx(0.05, abs=5e-4)


def test_p_value_is_one_for_no_difference():
    assert two_sided_p(0.0, 4) == pytest.approx(1.0)


@pytest.mark.parametrize("t, dof", [(1.0, 30), (0.5, 7), (4.0, 12), (0.1, 3)])
def test_p_value_stays_a_probability(t, dof):
    """The bug that motivated this file produced a negative p."""
    assert 0.0 <= two_sided_p(t, dof) <= 1.0


def test_p_value_is_symmetric_in_the_sign_of_t():
    assert two_sided_p(2.5, 8) == pytest.approx(two_sided_p(-2.5, 8))


def test_p_value_falls_as_the_effect_grows():
    ladder = [two_sided_p(t, 6) for t in (0.5, 1.0, 2.0, 4.0)]
    assert ladder == sorted(ladder, reverse=True)


def test_welch_separates_well_separated_arms():
    baseline = [41.55, 41.20, 42.10]
    better = [45.85, 46.30, 45.40]
    t, dof = welch(better, baseline)
    assert t > 0
    assert two_sided_p(t, dof) < 0.01


def test_welch_does_not_separate_overlapping_arms():
    baseline = [41.55, 38.20, 44.90]
    other = [43.30, 39.80, 45.10]
    t, dof = welch(other, baseline)
    assert two_sided_p(t, dof) > 0.05


def test_welch_needs_two_seeds():
    """A single-seed arm must report "no variance estimate", not a p-value."""
    assert welch([41.5], [45.8, 46.2, 45.4]) is None


def test_stdev_is_the_sample_standard_deviation():
    assert stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(2.13809, abs=1e-4)
    assert stdev([3.0]) == 0.0
    assert mean([1.0, 2.0, 6.0]) == pytest.approx(3.0)
