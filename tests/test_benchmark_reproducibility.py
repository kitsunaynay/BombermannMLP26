"""Reproducibility is a grading criterion. A benchmark must give the same answer twice."""

import random

import numpy as np
import pytest

from blib.benchmark import MatchConfig, benchmark_fast, play_rounds_fast

MATCH = ("attackontensor_ql", "rule_based_agent", "rule_based_agent")


def _key(summary):
    agent = summary["per_agent"]["attackontensor_ql_0"]
    return tuple(round(agent[field], 9) for field in
                 ("score_mean", "coins_mean", "kills_mean", "survival_rate", "suicide_rate"))


@pytest.fixture(autouse=True)
def unpinned_agent_seed(monkeypatch):
    """The benchmark must pin the seed itself, not inherit one from the shell."""
    monkeypatch.delenv("AOT_QL_SEED", raising=False)
    monkeypatch.delenv("AOT_PPO_SEED", raising=False)


def test_repeated_benchmarks_agree():
    config = MatchConfig(agents=MATCH, n_seeds=3, track_latency=False)
    assert _key(benchmark_fast(config)) == _key(benchmark_fast(config))


def test_result_survives_a_disturbed_global_rng():
    """Whatever ran before must not change the answer."""
    config = MatchConfig(agents=MATCH, n_seeds=3, track_latency=False)
    random.seed(1); np.random.seed(1)
    first = _key(benchmark_fast(config))
    random.seed(999); np.random.seed(999)
    [random.random() for _ in range(50)]
    assert _key(benchmark_fast(config)) == first


def test_different_arena_seeds_still_differ():
    """Determinism must not have been bought by ignoring the seed."""
    a = benchmark_fast(MatchConfig(agents=MATCH, n_seeds=4, base_seed=1, track_latency=False))
    b = benchmark_fast(MatchConfig(agents=MATCH, n_seeds=4, base_seed=2, track_latency=False))
    assert _key(a) != _key(b)


def test_rounds_within_a_seed_are_independently_seeded():
    """A round's outcome must not depend on how many rounds preceded it."""
    one = play_rounds_fast(MATCH, "classic", seed=7, n_rounds=1, track_latency=False)
    three = play_rounds_fast(MATCH, "classic", seed=7, n_rounds=3, track_latency=False)
    first_of_three = [stat for stat in three if stat.round_index == 0]
    assert [s.score for s in one] == [s.score for s in first_of_three]
