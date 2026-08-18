"""Report asset generation.

These run headless (matplotlib Agg) and assert files are produced with sane
content rather than inspecting pixels. The point is that
``tools/make_report_assets.py`` never fails on a partially-populated results
directory -- during a project you regenerate assets constantly, often while a
training run is midway through writing its CSV.
"""

import csv
import json

import pytest

from blib import plots
from blib.benchmark import MatchConfig, summarise
from blib.metrics import EpisodeStats

pytestmark = pytest.mark.skipif(not plots.HAVE_MATPLOTLIB, reason="matplotlib not installed")


def write_training_csv(path, rounds=120, ppo=False):
    fields = ["round", "coins", "score", "crates", "suicides", "steps", "total_reward"]
    if ppo:
        fields += ["entropy", "approx_kl", "clip_fraction", "explained_variance",
                   "policy_loss", "value_loss", "updates"]

    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(rounds):
            row = {
                "round": index + 1,
                "coins": index * 0.3,
                "score": index * 0.2,
                "crates": index * 0.1,
                "suicides": max(0, 1 - index / rounds),
                "steps": 100 + index,
                "total_reward": index - 20,
            }
            if ppo:
                # Only every tenth round carries an update, as in a real run.
                has_update = index % 10 == 0
                row.update(
                    entropy=1.2 if has_update else 0.0,
                    approx_kl=0.01 if has_update else 0.0,
                    clip_fraction=0.1 if has_update else 0.0,
                    explained_variance=0.5 if has_update else 0.0,
                    policy_loss=-0.01 if has_update else 0.0,
                    value_loss=1.0 if has_update else 0.0,
                    updates=8 if has_update else 0,
                )
            writer.writerow(row)
    return path


def make_summary():
    stats = [
        EpisodeStats(agent="attackontensor_ql", score=float(i), coins=i, steps=100,
                     survived=1, latencies_ms=[0.3, 0.4])
        for i in range(8)
    ] + [
        EpisodeStats(agent="rule_based_agent", score=float(i) + 2, coins=i, steps=100,
                     survived=1, latencies_ms=[1.1, 1.3])
        for i in range(8)
    ]
    return summarise(stats, MatchConfig(agents=["a", "b"], n_seeds=8), backend="fast")


def test_training_curves_are_written(tmp_path):
    csv_path = write_training_csv(tmp_path / "train_metrics.csv")
    output = tmp_path / "curves.png"

    assert plots.plot_training_curves(csv_path, output) is not None
    assert output.stat().st_size > 1000


def test_training_curves_on_an_empty_csv(tmp_path):
    """A run that has not written a row yet must not crash the generator."""
    path = tmp_path / "empty.csv"
    path.write_text("round,coins\n")

    assert plots.plot_training_curves(path, tmp_path / "out.png") is None


def test_ppo_diagnostics_use_only_rows_with_updates(tmp_path):
    csv_path = write_training_csv(tmp_path / "ppo_metrics.csv", rounds=100, ppo=True)
    output = tmp_path / "diag.png"

    assert plots.plot_ppo_diagnostics(csv_path, output) is not None
    assert output.exists()


def test_benchmark_comparison_and_latency_plots(tmp_path):
    summary = make_summary()

    comparison = plots.plot_benchmark_comparison(summary, tmp_path / "bench.png")
    latency = plots.plot_latency_histogram(summary, tmp_path / "lat.png")

    assert comparison is not None and latency is not None


def test_plots_handle_an_empty_summary(tmp_path):
    assert plots.plot_benchmark_comparison({"per_agent": {}}, tmp_path / "x.png") is None
    assert plots.plot_latency_histogram({"rows": []}, tmp_path / "y.png") is None


def test_markdown_table_lists_every_agent():
    table = plots.markdown_table(make_summary())

    assert "attackontensor_ql" in table and "rule_based_agent" in table
    assert table.count("\n") >= 3  # header, separator, two agents


def test_latex_table_escapes_special_characters():
    table = plots.latex_table(make_summary(), caption="Test", label="tab:x")

    assert "\\begin{table}" in table and "\\end{table}" in table
    assert "attackontensor\\_ql" in table, "underscores must be escaped for LaTeX"
    assert "\\%" in table


def test_moving_average_smooths_and_shortens():
    import numpy as np

    values = np.arange(100, dtype=float)
    smoothed = plots.moving_average(values, 10)

    assert len(smoothed) == 91
    assert smoothed[0] == pytest.approx(4.5)


def test_moving_average_passes_short_series_through():
    import numpy as np

    values = np.arange(5, dtype=float)
    assert len(plots.moving_average(values, 10)) == 5


def test_column_skips_blanks_and_non_numeric():
    rows = [{"a": "1"}, {"a": ""}, {"a": "oops"}, {"a": "3"}]
    assert list(plots.column(rows, "a")) == [1.0, 3.0]


def test_make_report_assets_on_an_empty_results_dir(tmp_path):
    """Must exit cleanly, not crash, when there is nothing to plot yet."""
    import sys

    sys.path.insert(0, str(plots.Path(__file__).resolve().parent.parent / "tools"))
    import make_report_assets

    code = make_report_assets.main(
        ["--results-dir", str(tmp_path / "results"), "--output", str(tmp_path / "out")]
    )
    assert code == 1, "a missing results directory is reported, not ignored"

    (tmp_path / "results").mkdir()
    code = make_report_assets.main(
        ["--results-dir", str(tmp_path / "results"), "--output", str(tmp_path / "out")]
    )
    assert code == 0


def test_make_report_assets_produces_tables(tmp_path):
    import sys

    sys.path.insert(0, str(plots.Path(__file__).resolve().parent.parent / "tools"))
    import make_report_assets

    results = tmp_path / "results" / "run"
    results.mkdir(parents=True)
    write_training_csv(results / "train_metrics.csv")
    (results / "bench.json").write_text(json.dumps(make_summary(), default=str))

    output = tmp_path / "assets"
    assert make_report_assets.main(
        ["--results-dir", str(tmp_path / "results"), "--output", str(output)]
    ) == 0

    assert (output / "tables.md").exists()
    assert (output / "tables.tex").exists()
    assert any(output.glob("training_*.png"))
    assert any(output.glob("benchmark_*.png"))
