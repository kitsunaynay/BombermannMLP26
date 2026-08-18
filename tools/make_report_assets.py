#!/usr/bin/env python3
"""Regenerate every figure and table for the report.

Walks ``results/`` for training metric CSVs and benchmark JSONs and writes
figures, a Markdown table file and a LaTeX table file into ``report_assets/``.
Everything is derived from committed files, so a reader can reproduce every
number in the report by re-running this script -- which is the replication
standard the brief asks for.

Usage::

    python tools/make_report_assets.py
    python tools/make_report_assets.py --results-dir results --output report_assets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from blib import plots  # noqa: E402
from blib.paths import display_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--output", default=str(REPO_ROOT / "report_assets"))
    parser.add_argument("--smooth", type=int, default=25, help="moving-average window")
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.is_dir():
        print(f"No results directory at {results_dir}", file=sys.stderr)
        return 1

    written = []

    # -- training curves ---------------------------------------------------
    for csv_path in sorted(results_dir.rglob("*metrics*.csv")):
        relative = csv_path.relative_to(results_dir)
        stem = str(relative.parent).replace("/", "_") or csv_path.stem

        curves = output_dir / f"training_{stem}.{args.format}"
        if plots.plot_training_curves(csv_path, curves, title=f"Training: {stem}", smooth=args.smooth):
            written.append(curves)

        # PPO-only diagnostics; harmlessly skipped for Q-learning runs.
        rows = plots.read_csv(csv_path)
        if rows and "entropy" in rows[0]:
            diagnostics = output_dir / f"ppo_diagnostics_{stem}.{args.format}"
            if plots.plot_ppo_diagnostics(csv_path, diagnostics, title=f"PPO: {stem}"):
                written.append(diagnostics)

    # -- benchmarks --------------------------------------------------------
    markdown_sections = []
    latex_sections = []

    for json_path in sorted(results_dir.rglob("*.json")):
        try:
            summary = plots.load_summary(json_path)
        except Exception:  # noqa: BLE001 - config.json and summary.json also live here
            continue
        if not isinstance(summary, dict) or "per_agent" not in summary:
            continue

        stem = json_path.stem[:80]

        comparison = output_dir / f"benchmark_{stem}.{args.format}"
        if plots.plot_benchmark_comparison(summary, comparison):
            written.append(comparison)

        latency = output_dir / f"latency_{stem}.{args.format}"
        if plots.plot_latency_histogram(summary, latency):
            written.append(latency)

        heading = (
            f"### {summary.get('scenario', '?')} "
            f"({summary.get('n_seeds', '?')} seeds, backend={summary.get('backend', '?')})"
        )
        markdown_sections.append(f"{heading}\n\n{plots.markdown_table(summary)}\n")
        latex_sections.append(
            plots.latex_table(
                summary,
                caption=f"Performance on {summary.get('scenario')} over "
                        f"{summary.get('n_seeds')} seeds.",
                label=f"tab:{stem[:40]}",
            )
        )

    if markdown_sections:
        table_md = output_dir / "tables.md"
        table_md.write_text("# Benchmark results\n\n" + "\n".join(markdown_sections))
        written.append(table_md)

        table_tex = output_dir / "tables.tex"
        table_tex.write_text("\n\n".join(latex_sections))
        written.append(table_tex)

    if not written:
        print("Nothing to do: no training CSVs or benchmark JSONs found under "
              f"{results_dir}. Run tools/train_ql.py or tools/benchmark.py first.")
        return 0

    print(f"Wrote {len(written)} asset(s) to {display_path(output_dir)}/:")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
