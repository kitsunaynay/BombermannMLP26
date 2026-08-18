"""Tests for tracking, metrics, curriculum and benchmarking."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from blib.benchmark import MatchConfig, _stats_from_main_json, format_table, play_rounds_fast, summarise
from blib.curriculum import STAGES, Gate, describe_stage, get_stage, report_gate
from blib.metrics import (
    EpisodeStats,
    aggregate,
    bootstrap_ci,
    coin_efficiency,
    mark_winners,
    stats_from_agent,
)
from blib.tracking import CsvJsonLogger, JsonSummaryLogger, MultiLogger, make_run_id


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------


def test_csv_logger_writes_rows_and_config(tmp_path):
    logger = CsvJsonLogger(tmp_path / "run")
    logger.log_config({"alpha": 0.1})
    logger.log({"round": 1, "score": 3.0})
    logger.log({"round": 2, "score": 5.0})
    logger.close()

    assert json.loads((tmp_path / "run" / "config.json").read_text())["alpha"] == 0.1

    text = (tmp_path / "run" / "metrics.csv").read_text().strip().splitlines()
    assert text[0] == "round,score"
    assert len(text) == 3


def test_csv_logger_accepts_columns_that_appear_late(tmp_path):
    """A PPO update metric first appears many rounds in; it must still get a column."""
    logger = CsvJsonLogger(tmp_path / "run")
    logger.log({"round": 1})
    logger.log({"round": 2, "entropy": 1.2})
    logger.close()

    lines = (tmp_path / "run" / "metrics.csv").read_text().strip().splitlines()
    assert lines[0] == "round,entropy"
    assert lines[1] == "1,"  # blank rather than dropped


def test_multi_logger_survives_a_failing_sink(tmp_path):
    """Tracking must never take a training run down with it."""

    class Broken:
        def log_config(self, config):
            raise RuntimeError("boom")

        def log(self, metrics, step=None):
            raise RuntimeError("boom")

        def flush(self):
            raise RuntimeError("boom")

        def close(self):
            raise RuntimeError("boom")

    good = CsvJsonLogger(tmp_path / "run")
    logger = MultiLogger([Broken(), good])

    logger.log_config({"a": 1})
    logger.log({"round": 1})
    logger.close()

    assert (tmp_path / "run" / "metrics.csv").exists()


def test_json_summary_logger(tmp_path):
    logger = JsonSummaryLogger(tmp_path / "out.json")
    logger.log_config({"seeds": 5})
    logger.log({"agent": "x", "score": 1})
    logger.close()

    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["config"]["seeds"] == 5
    assert payload["entries"][0]["agent"] == "x"


def test_run_ids_are_sortable_and_prefixed():
    run_id = make_run_id("ppo")
    assert run_id.startswith("ppo-")
    assert len(run_id) > len("ppo-")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_stats_from_agent_reads_framework_counters():
    agent = SimpleNamespace(
        name="a",
        score=7,
        dead=False,
        statistics={"coins": 2, "kills": 1, "steps": 40, "invalid": 3},
    )

    stats = stats_from_agent(agent, seed=1)

    assert stats.score == 7.0
    assert stats.coins == 2 and stats.kills == 1
    assert stats.survived == 1
    assert stats.invalid_rate == pytest.approx(3 / 40)


def test_mark_winners_handles_ties():
    stats = [EpisodeStats(agent="a", score=5), EpisodeStats(agent="b", score=5),
             EpisodeStats(agent="c", score=1)]

    mark_winners(stats)

    assert [s.won for s in stats] == [1, 1, 0], "tied agents both count as winners"


def test_bootstrap_ci_brackets_the_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    low, high = bootstrap_ci(values, seed=0)
    assert low <= np.mean(values) <= high


def test_bootstrap_ci_degenerate_inputs():
    assert bootstrap_ci([]) == (float("nan"), float("nan")) or np.isnan(bootstrap_ci([])[0])
    assert bootstrap_ci([2.0]) == (2.0, 2.0)


def test_aggregate_produces_means_and_intervals():
    stats = [EpisodeStats(agent="a", score=float(i), coins=i, steps=10) for i in range(10)]

    summary = aggregate(stats)

    assert summary["n_rounds"] == 10
    assert summary["score_mean"] == pytest.approx(4.5)
    assert summary["score_ci_low"] <= 4.5 <= summary["score_ci_high"]
    assert "win_rate" in summary and "survival_rate" in summary


def test_aggregate_on_empty_input():
    assert aggregate([]) == {"n_rounds": 0}


def test_latency_summary_is_included_when_present():
    stats = [EpisodeStats(agent="a", latencies_ms=[1.0, 2.0, 3.0])]
    summary = aggregate(stats)

    assert summary["latency_p95_ms"] > 0
    assert summary["latency_budget_ms"] == pytest.approx(500.0)
    assert summary["latency_headroom"] > 1


def test_coin_efficiency_is_a_fraction():
    stats = [EpisodeStats(agent="a", coins=25)]
    assert coin_efficiency(stats, coins_available=50) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------


def test_stages_cover_tasks_one_to_four():
    assert [stage.index for stage in STAGES] == [1, 2, 3, 4]
    assert get_stage(1).scenario == "coin-heaven"
    assert "rule_based_agent" in get_stage(4).opponents


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="Unknown curriculum stage"):
        get_stage(99)


def test_gate_evaluates_each_condition():
    gate = Gate("demo", {"coins_mean": (">=", 45.0), "suicide_rate": ("<=", 0.05)})

    passed, results = gate.evaluate({"coins_mean": 49.0, "suicide_rate": 0.01})
    assert passed and all(results.values())

    passed, results = gate.evaluate({"coins_mean": 49.0, "suicide_rate": 0.5})
    assert not passed and results["coins_mean"] and not results["suicide_rate"]


def test_gate_fails_on_a_missing_metric():
    """A metric that was never measured must not silently pass."""
    gate = Gate("demo", {"never_measured": (">=", 1.0)})
    passed, results = gate.evaluate({"score_mean": 10.0})
    assert not passed and results["never_measured"] is False


def test_stage_descriptions_render():
    for stage in STAGES:
        text = describe_stage(stage)
        assert stage.name in text and stage.scenario in text

    assert "PASS" in report_gate(
        get_stage(1), {"coins_mean": 49.0, "invalid_rate_mean": 0.0, "suicide_rate": 0.0}
    )


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------


def test_match_config_seeds_are_deterministic():
    config = MatchConfig(agents=["a"], n_seeds=5)
    assert config.seeds == MatchConfig(agents=["b"], n_seeds=5).seeds
    assert len(set(config.seeds)) == 5


@pytest.mark.slow
def test_play_rounds_fast_produces_stats_for_every_agent():
    stats = play_rounds_fast(
        ["random_agent", "peaceful_agent"], "coin-heaven", seed=3, n_rounds=2
    )

    assert len({stat.agent for stat in stats}) == 2
    assert len(stats) == 4, "two agents x two rounds"
    assert all(stat.latencies_ms for stat in stats), "latency tracking is on by default"
    # Exactly one winner per round unless the round was tied.
    assert sum(stat.won for stat in stats) >= 2


def test_duplicate_agents_get_distinct_names():
    stats = play_rounds_fast(
        ["random_agent", "random_agent"], "coin-heaven", seed=1, n_rounds=1,
        track_latency=False,
    )
    assert {stat.agent for stat in stats} == {"random_agent_0", "random_agent_1"}


def test_stats_from_main_json_normalises_by_round_count():
    payload = {
        "by_agent": {"a": {"score": 20, "coins": 10, "steps": 400, "kills": 2}},
        "by_round": {"r1": {}, "r2": {}},
    }

    stats = _stats_from_main_json(payload, seed=1)

    assert len(stats) == 1
    assert stats[0].score == pytest.approx(10.0), "20 points over 2 rounds"
    assert stats[0].coins == 5


def test_format_table_renders_without_latency():
    stats = [EpisodeStats(agent="a", score=3.0), EpisodeStats(agent="b", score=1.0)]
    summary = summarise(stats, MatchConfig(agents=["a", "b"], n_seeds=1), backend="fast")

    text = format_table(summary)

    assert "agent" in text and "a" in text and "b" in text
    assert text.index("\na ") < text.index("\nb "), "sorted by score, best first"


def test_main_backend_does_not_report_survival_as_zero():
    """The framework's stats JSON has no per-agent survival field.

    Printing 0.0% there would read as "never survived" when it actually means
    "not measured", which is exactly the kind of number that ends up in a report.
    """
    stats = [EpisodeStats(agent="a", score=3.0), EpisodeStats(agent="b", score=1.0)]
    config = MatchConfig(agents=["a", "b"], n_seeds=1)

    fast_table = format_table(summarise(stats, config, backend="fast"))
    main_table = format_table(summarise(stats, config, backend="main"))

    assert "0.0" in fast_table, "the fast backend does measure survival"
    header, *rows = [line for line in main_table.splitlines() if line.strip()][2:]
    for row in [header, *rows]:
        # The survival column is a bare dash on the main backend.
        assert "     -" in row or row.startswith("-"), row
