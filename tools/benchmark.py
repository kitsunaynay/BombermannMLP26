#!/usr/bin/env python3
"""Evaluate agents against the provided baselines over N seeds.

Every agent plays the *same* arena seeds, so a difference in the table reflects
the agents rather than the luck of the draw. Results land in
``results/benchmarks/<name>.json`` for ``tools/make_report_assets.py`` to turn
into tables and figures.

Examples::

    # Our two agents against the strong baseline, quick in-process backend
    python tools/benchmark.py --agents attackontensor_ql rule_based_agent --seeds 30

    # Certified numbers through the real framework (slower, tournament-exact)
    python tools/benchmark.py --agents attackontensor_ppo rule_based_agent \\
        --seeds 20 --backend main

    # A curriculum stage's exact matchup
    python tools/benchmark.py --agent attackontensor_ql --stage 4 --seeds 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from blib.benchmark import MatchConfig, benchmark_fast, benchmark_main, format_table  # noqa: E402
from blib.curriculum import STAGES, describe_stage, get_stage, report_gate  # noqa: E402
from blib.paths import display_path  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "benchmarks"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--agents", nargs="+", help="explicit agent list (max 4)")
    group.add_argument("--agent", help="one agent; opponents come from --stage")
    group.add_argument("--list-stages", action="store_true", help="show the curriculum")

    parser.add_argument("--stage", type=int, choices=[s.index for s in STAGES],
                        help="curriculum stage; sets scenario and opponents")
    parser.add_argument("--scenario", default=None, help="override the scenario")
    parser.add_argument("--seeds", type=int, default=20, help="number of arena seeds")
    parser.add_argument("--rounds-per-seed", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260921)
    parser.add_argument("--backend", choices=("fast", "main"), default="fast")
    parser.add_argument("--no-latency", action="store_true", help="skip act() timing")
    parser.add_argument("--output", default=None, help="path for the results JSON")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_stages:
        for stage in STAGES:
            print(describe_stage(stage))
            print()
        return 0

    stage = get_stage(args.stage) if args.stage else None

    if args.agent:
        if stage is None:
            print("--agent requires --stage (it supplies the opponents)", file=sys.stderr)
            return 2
        agents = [args.agent, *stage.opponents]
    else:
        agents = list(args.agents)

    if len(agents) > 4:
        print(f"At most {4} agents can play; got {len(agents)}", file=sys.stderr)
        return 2

    scenario = args.scenario or (stage.scenario if stage else "classic")

    config = MatchConfig(
        agents=agents,
        scenario=scenario,
        n_seeds=args.seeds,
        rounds_per_seed=args.rounds_per_seed,
        base_seed=args.base_seed,
        track_latency=not args.no_latency and args.backend == "fast",
    )

    if not args.quiet:
        print(f"Benchmarking {' vs '.join(agents)} on '{scenario}' "
              f"({args.seeds} seeds x {args.rounds_per_seed} rounds, backend={args.backend})")

    runner = benchmark_fast if args.backend == "fast" else benchmark_main
    summary = runner(config, progress=not args.quiet)

    print()
    print(format_table(summary))

    if stage is not None:
        evaluated = summary["per_agent"].get(_display_name(agents, 0, args.backend))
        if evaluated:
            print()
            print(report_gate(stage, evaluated))

    output = Path(args.output) if args.output else _default_output(agents, scenario, args.backend)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {display_path(output)}")
    return 0


def _display_name(agents, index: int, backend: str) -> str:
    """Name the backend will have used for the agent at ``index``.

    The fast backend suffixes every agent with its slot to keep names unique;
    ``main.py`` only disambiguates when a code name repeats (environment.py:342).
    """
    if backend == "fast":
        return f"{agents[index]}_{index}"
    if list(agents).count(agents[index]) > 1:
        return f"{agents[index]}_{index}"
    return agents[index]


def _default_output(agents, scenario: str, backend: str) -> Path:
    stem = "_vs_".join(agents)[:120]
    return DEFAULT_OUTPUT_DIR / f"{stem}__{scenario}__{backend}.json"


if __name__ == "__main__":
    raise SystemExit(main())
