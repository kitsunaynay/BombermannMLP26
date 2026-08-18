#!/usr/bin/env python3
"""Curriculum-driven training for the Q-learning agent.

Drives the framework's own training loop (``main.py play --train 1``) rather
than a private one. That loop is already proven correct and is the one the brief
describes, and the Q-learning agent is cheap enough that its throughput is not
the bottleneck -- roughly 30 rounds/s solo.

Configuration is passed as ``AOT_QL_*`` environment variables, so no
tournament-critical file is touched.

Examples::

    python tools/train_ql.py --stage 1                      # coin-heaven
    python tools/train_ql.py --stage 2 --episodes 6000      # crates
    python tools/train_ql.py --all                          # tasks 1-4 in order
    python tools/train_ql.py --stage 4 --resume --no-benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from blib.benchmark import MatchConfig, benchmark_fast, format_table  # noqa: E402
from blib.curriculum import STAGES, describe_stage, get_stage, report_gate  # noqa: E402
from blib.paths import display_path  # noqa: E402
from blib.tracking import make_run_id  # noqa: E402

AGENT = "attackontensor_ql"
AGENT_DIR = REPO_ROOT / "agent_code" / AGENT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", type=int, choices=[s.index for s in STAGES])
    group.add_argument("--all", action="store_true", help="run every stage in order")
    group.add_argument("--list-stages", action="store_true")

    parser.add_argument("--episodes", type=int, default=None, help="override the stage default")
    parser.add_argument("--resume", action="store_true", help="keep the existing Q-table")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--safety-mode", choices=("none", "soft", "hard"), default="soft")
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--no-custom-events", action="store_true")
    parser.add_argument("--single-q", action="store_true", help="disable Double Q-learning")
    parser.add_argument("--variant", choices=("compact", "full"), default=None)
    parser.add_argument("--benchmark-seeds", type=int, default=20)
    parser.add_argument("--select-seeds", type=int, default=12,
                        help="held-out seeds used to choose the best checkpoint")
    parser.add_argument("--no-select-best", action="store_true",
                        help="keep the final table instead of the best snapshot")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--python", default=sys.executable)
    return parser


def stage_environment(args, stage, run_dir: Path) -> dict:
    """Environment variables configuring the agent for this stage."""
    environment = dict(os.environ)
    environment.update(
        AOT_QL_VARIANT=args.variant or stage.variant,
        AOT_QL_EPSILON_START=str(args.epsilon_start),
        AOT_QL_EPSILON_END=str(args.epsilon_end),
        # Decay over roughly the first fifth of the run, so most episodes are
        # spent refining rather than exploring.
        AOT_QL_EPSILON_DECAY_ROUNDS=str(max(1, (args.episodes or stage.episodes) // 5)),
        AOT_QL_ALPHA=str(args.alpha),
        AOT_QL_GAMMA=str(args.gamma),
        AOT_QL_N_STEP=str(args.n_step),
        AOT_QL_SAFETY_MODE=args.safety_mode,
        AOT_QL_USE_SYMMETRY=str(not args.no_symmetry).lower(),
        AOT_QL_USE_POTENTIAL_SHAPING=str(not args.no_shaping).lower(),
        AOT_QL_USE_CUSTOM_EVENTS=str(not args.no_custom_events).lower(),
        AOT_QL_DOUBLE_Q=str(not args.single_q).lower(),
        AOT_QL_METRICS_FILE=str(run_dir / "train_metrics.csv"),
        AOT_QL_CHECKPOINT_EVERY="250",
    )
    return environment


def run_stage(args, stage, run_id: str) -> int:
    episodes = args.episodes or stage.episodes
    run_dir = REPO_ROOT / "results" / run_id / stage.label
    run_dir.mkdir(parents=True, exist_ok=True)

    model = AGENT_DIR / "q_table.pkl"
    if not args.resume and model.exists():
        model.unlink()
        print(f"Removed {model.name} (use --resume to continue from it)")

    print()
    print(describe_stage(stage))
    print(f"  episodes  : {episodes}")
    print(f"  results   : {display_path(run_dir)}")
    print()

    command = [
        args.python, "main.py", "play", "--no-gui",
        "--n-rounds", str(episodes),
        "--scenario", stage.scenario,
        "--agents", AGENT, *stage.opponents,
        "--train", "1",
        # Deliberately NOT passing --continue-without-training. A dead agent
        # receives no further callbacks (environment.py:469), so playing the
        # round out after it dies produces no experience at all -- it only
        # burns wall-clock. Ending the round early is strictly faster and
        # changes nothing about what the agent learns.
    ]

    completed = subprocess.run(command, cwd=REPO_ROOT, env=stage_environment(args, stage, run_dir))
    if completed.returncode != 0:
        print(f"Training failed for {stage.label}", file=sys.stderr)
        return completed.returncode

    if args.no_benchmark:
        return 0

    if not args.no_select_best:
        select_best_checkpoint(args, stage, run_dir)

    print(f"\nEvaluating {stage.label} over {args.benchmark_seeds} seeds...")
    config = MatchConfig(
        agents=[AGENT, *stage.opponents],
        scenario=stage.scenario,
        n_seeds=args.benchmark_seeds,
    )
    summary = benchmark_fast(config)
    print()
    print(format_table(summary))
    print()
    print(report_gate(stage, summary["per_agent"][f"{AGENT}_0"]))

    (run_dir / "benchmark.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


#: Selection uses a different base seed from the final report, so the table is
#: not chosen on the same arenas it is then scored on.
SELECTION_BASE_SEED = 777_000_1


def select_best_checkpoint(args, stage, run_dir: Path) -> None:
    """Promote the best snapshot to be the live model.

    A constant learning rate makes tabular Q-values track rather than converge,
    so the last table a run writes is not reliably its best -- measured on Task 2
    the final table scored 16.3 coins where a mid-run one scored 24.9. Selection
    runs on its own seed set (`SELECTION_BASE_SEED`); the gate afterwards uses
    the normal evaluation seeds, so the reported number is not the one the
    checkpoint was picked on.
    """
    snapshot_dir = AGENT_DIR / "checkpoints"
    snapshots = sorted(snapshot_dir.glob("q_table_r*.pkl"))
    model = AGENT_DIR / "q_table.pkl"
    if model.exists():
        snapshots.append(model)

    if len(snapshots) < 2:
        print("\nOnly one checkpoint; skipping selection.")
        return

    print(f"\nSelecting among {len(snapshots)} checkpoints "
          f"({args.select_seeds} held-out seeds each)...")

    results = []
    for path in snapshots:
        os.environ["AOT_QL_MODEL_FILE"] = str(path)
        summary = benchmark_fast(
            MatchConfig(
                agents=[AGENT, *stage.opponents],
                scenario=stage.scenario,
                n_seeds=args.select_seeds,
                base_seed=SELECTION_BASE_SEED,
                track_latency=False,
            )
        )
        values = summary["per_agent"][f"{AGENT}_0"]
        # Score first, then break ties toward staying alive: a table that scores
        # the same while dying less is the better tournament agent.
        key = (values.get("score_mean", 0.0), values.get("survival_rate", 0.0))
        results.append((key, path, values))
        print(f"  {path.name:<28} score={values.get('score_mean', 0):6.2f} "
              f"surv={values.get('survival_rate', 0):5.1%} "
              f"suic={values.get('suicide_rate', 0):5.2f}")

    os.environ.pop("AOT_QL_MODEL_FILE", None)
    results.sort(key=lambda item: item[0], reverse=True)
    best_key, best_path, _ = results[0]

    if best_path != model:
        shutil.copy2(best_path, model)
        print(f"  -> promoted {best_path.name} (score {best_key[0]:.2f})")
    else:
        print(f"  -> final table was already the best (score {best_key[0]:.2f})")

    (run_dir / "checkpoint_selection.json").write_text(
        json.dumps(
            [{"checkpoint": p.name, "score": k[0], "survival": k[1]} for k, p, _ in results],
            indent=2,
        )
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_stages:
        for stage in STAGES:
            print(describe_stage(stage))
            print()
        return 0

    run_id = args.run_id or make_run_id("ql")
    stages = STAGES if args.all else [get_stage(args.stage)]

    if args.all and not args.resume:
        # Stages chain: each one continues from the table the previous produced.
        print("Running the full curriculum; stages after the first resume automatically.")

    for index, stage in enumerate(stages):
        stage_args = argparse.Namespace(**vars(args))
        if args.all and index > 0:
            stage_args.resume = True
        code = run_stage(stage_args, stage, run_id)
        if code != 0:
            return code

    print(f"\nDone. Results under results/{run_id}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
