"""Figures and tables for the report.

The brief makes *Experiments and Results* the most heavily weighted section and
asks for "training progress diagrams, performance comparisons between your
agents and the predefined ones". Everything here reads the CSV and JSON that
training and benchmarking already wrote, so every figure regenerates from files
in the repository -- no notebook state, no manual steps.

Matplotlib only, with the default style and no seaborn, so the figures render
identically on a machine that only has the packages listed in requirements.txt.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # headless: never try to open a window
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAVE_MATPLOTLIB = False

FIGURE_SIZE = (8.0, 4.5)
DPI = 150


def _require_matplotlib() -> None:
    if not HAVE_MATPLOTLIB:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting; pip install matplotlib")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def column(rows: Sequence[Dict[str, str]], name: str) -> np.ndarray:
    """Numeric column, with blanks and non-numeric entries dropped."""
    values = []
    for row in rows:
        raw = row.get(name, "")
        if raw in ("", None):
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return np.asarray(values, dtype=float)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Smoothing for noisy per-episode curves.

    Round-to-round variance on random arenas is large enough that a raw curve is
    unreadable; the smoothed line is what shows the trend.
    """
    if len(values) < window or window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# --------------------------------------------------------------------------
# Training curves
# --------------------------------------------------------------------------

#: Metrics worth a panel, with a readable label and whether lower is better.
CURVE_PANELS = (
    ("coins", "Coins collected", False),
    ("score", "Score", False),
    ("crates", "Crates destroyed", False),
    ("suicides", "Suicides per round", True),
    ("steps", "Steps survived", False),
    ("total_reward", "Shaped return", False),
)


def plot_training_curves(
    metrics_csv: Path,
    output: Path,
    title: str = "Training progress",
    smooth: int = 25,
) -> Optional[Path]:
    """Grid of per-round training curves from a train_metrics.csv."""
    _require_matplotlib()
    rows = read_csv(metrics_csv)
    if not rows:
        return None

    panels = [(key, label, lower) for key, label, lower in CURVE_PANELS if column(rows, key).size]
    if not panels:
        return None

    columns = 3
    n_rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(n_rows, columns, figsize=(4.5 * columns, 3.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    rounds = column(rows, "round")
    for axis, (key, label, lower_is_better) in zip(axes, panels):
        values = column(rows, key)
        x = rounds[: len(values)] if rounds.size >= len(values) else np.arange(len(values))

        axis.plot(x, values, alpha=0.25, linewidth=0.8, color="tab:blue")
        smoothed = moving_average(values, smooth)
        if len(smoothed) < len(values):
            axis.plot(x[len(values) - len(smoothed):], smoothed, linewidth=2.0, color="tab:blue")

        axis.set_title(label + (" (lower is better)" if lower_is_better else ""))
        axis.set_xlabel("round")
        axis.grid(alpha=0.3)

    for axis in axes[len(panels):]:
        axis.set_visible(False)

    figure.suptitle(title)
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return Path(output)


def plot_ppo_diagnostics(metrics_csv: Path, output: Path, title: str = "PPO diagnostics"):
    """Entropy, KL, clip fraction and explained variance.

    These four are the ones that say *why* a PPO run is failing: collapsed
    entropy, a policy stepping outside its trust region, or a critic that never
    learned. Rows without an update are dropped -- an update only runs when the
    rollout buffer fills, so treating every round as a sample would flatten the
    curves with repeated values.
    """
    _require_matplotlib()
    rows = read_csv(metrics_csv)
    if "updates" in (rows[0] if rows else {}):
        rows = [row for row in rows if float(row.get("updates") or 0) > 0]
    if not rows:
        return None

    panels = (
        ("entropy", "Policy entropy (nats)"),
        ("approx_kl", "Approximate KL"),
        ("clip_fraction", "Clip fraction"),
        ("explained_variance", "Explained variance"),
        ("policy_loss", "Policy loss"),
        ("value_loss", "Value loss"),
    )
    available = [(key, label) for key, label in panels if column(rows, key).size]
    if not available:
        return None

    columns = 3
    n_rows = int(np.ceil(len(available) / columns))
    figure, axes = plt.subplots(n_rows, columns, figsize=(4.5 * columns, 3.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, (key, label) in zip(axes, available):
        values = column(rows, key)
        axis.plot(np.arange(len(values)), values, linewidth=1.4, color="tab:orange")
        axis.set_title(label)
        axis.set_xlabel("update")
        axis.grid(alpha=0.3)

        if key == "entropy":
            # The failure detector, drawn so a collapse is visible at a glance.
            axis.axhline(0.2, color="red", linestyle="--", linewidth=1.0, label="collapse threshold")
            axis.legend(fontsize="small")

    for axis in axes[len(available):]:
        axis.set_visible(False)

    figure.suptitle(title)
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return Path(output)


# --------------------------------------------------------------------------
# Benchmark comparisons
# --------------------------------------------------------------------------


def plot_benchmark_comparison(
    summary: Dict,
    output: Path,
    metric: str = "score",
    title: Optional[str] = None,
):
    """Bar chart with bootstrap confidence intervals as error bars."""
    _require_matplotlib()
    per_agent = summary.get("per_agent", {})
    if not per_agent:
        return None

    names = sorted(per_agent, key=lambda n: -per_agent[n].get(f"{metric}_mean", 0.0))
    means = [per_agent[n].get(f"{metric}_mean", 0.0) for n in names]
    lows = [per_agent[n].get(f"{metric}_ci_low", m) for n, m in zip(names, means)]
    highs = [per_agent[n].get(f"{metric}_ci_high", m) for n, m in zip(names, means)]

    # errorbar wants distances from the bar top, not absolute bounds.
    errors = np.array([
        [max(0.0, m - lo) for m, lo in zip(means, lows)],
        [max(0.0, hi - m) for m, hi in zip(means, highs)],
    ])

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    positions = np.arange(len(names))
    colours = ["tab:blue" if "attackontensor" in n else "tab:gray" for n in names]

    axis.bar(positions, means, yerr=errors, capsize=4, color=colours)
    axis.set_xticks(positions)
    axis.set_xticklabels([n.replace("attackontensor_", "AoT-") for n in names], rotation=20, ha="right")
    axis.set_ylabel(metric.replace("_", " "))
    axis.set_title(
        title
        or f"{metric} on '{summary.get('scenario')}' "
        f"({summary.get('n_seeds')} seeds, 95% bootstrap CI)"
    )
    axis.grid(alpha=0.3, axis="y")

    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return Path(output)


def plot_latency_histogram(summary: Dict, output: Path):
    """Per-step inference latency against the 0.5 s tournament budget."""
    _require_matplotlib()
    rows = summary.get("rows", [])
    by_agent: Dict[str, List[float]] = {}
    for row in rows:
        value = row.get("latency_mean_ms")
        if value is not None:
            by_agent.setdefault(row["agent"], []).append(float(value))
    if not by_agent:
        return None

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for name, values in sorted(by_agent.items()):
        axis.hist(values, bins=25, alpha=0.55, label=name.replace("attackontensor_", "AoT-"))

    axis.set_xlabel("mean act() latency per round (ms)")
    axis.set_ylabel("rounds")
    axis.set_title("Inference latency (tournament budget: 500 ms per step)")
    axis.legend(fontsize="small")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return Path(output)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

TABLE_COLUMNS = (
    ("score_mean", "Score", "{:.2f}"),
    ("win_rate", "Win %", "{:.1%}"),
    ("survival_rate", "Survival %", "{:.1%}"),
    ("coins_mean", "Coins", "{:.2f}"),
    ("crates_mean", "Crates", "{:.2f}"),
    ("kills_mean", "Kills", "{:.2f}"),
    ("suicide_rate", "Suicides", "{:.2f}"),
)


def _rows_for_table(summary: Dict):
    per_agent = summary.get("per_agent", {})
    for name in sorted(per_agent, key=lambda n: -per_agent[n].get("score_mean", 0.0)):
        values = per_agent[name]
        cells = [name]
        for key, _, fmt in TABLE_COLUMNS:
            cells.append(fmt.format(values.get(key, 0.0)))
        ci = (
            f"[{values.get('score_ci_low', 0):.2f}, {values.get('score_ci_high', 0):.2f}]"
        )
        cells.append(ci)
        yield cells


def markdown_table(summary: Dict) -> str:
    headers = ["Agent"] + [label for _, label, _ in TABLE_COLUMNS] + ["Score 95% CI"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for cells in _rows_for_table(summary):
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_table(summary: Dict, caption: str = "", label: str = "tab:benchmark") -> str:
    headers = ["Agent"] + [lab for _, lab, _ in TABLE_COLUMNS] + ["Score 95\\% CI"]
    column_spec = "l" + "r" * (len(headers) - 1)

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\begin{{tabular}}{{{column_spec}}}",
        "    \\hline",
        "    " + " & ".join(h.replace("%", "\\%") for h in headers) + " \\\\",
        "    \\hline",
    ]
    for cells in _rows_for_table(summary):
        escaped = [cell.replace("_", "\\_").replace("%", "\\%") for cell in cells]
        lines.append("    " + " & ".join(escaped) + " \\\\")
    lines += [
        "    \\hline",
        "  \\end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def load_summary(path: Path) -> Dict:
    return json.loads(Path(path).read_text())
