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
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from blib.benchmark import MatchConfig, benchmark_fast, format_table  # noqa: E402
from blib.curriculum import STAGES, describe_stage, get_stage, report_gate  # noqa: E402
from blib.paths import display_path  # noqa: E402
from blib.seeding import derive_seed  # noqa: E402
from blib.tracking import WandbLogger, make_run_id  # noqa: E402

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
    parser.add_argument("--alpha-decay", type=float, default=0.0,
                        help="visit-count learning-rate schedule: alpha / "
                             "(1 + alpha_decay * visits), floored at --alpha-min. "
                             "0 keeps alpha constant. Implemented at train.py:184 "
                             "but unreachable from this tool until now, so every "
                             "run before Phase 8 trained with constant alpha.")
    parser.add_argument("--alpha-min", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--safety-mode", choices=("none", "soft", "hard"), default="soft")
    parser.add_argument("--opponent-bomb-lookahead", action="store_true",
                        help="the hard escape search assumes every armed "
                             "opponent bombs from where it stands. Only "
                             "meaningful with --safety-mode hard.")
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--no-custom-events", action="store_true")
    parser.add_argument("--single-q", action="store_true", help="disable Double Q-learning")
    parser.add_argument("--variant", choices=("compact", "full"), default=None)
    parser.add_argument("--checkpoint-every", type=int, default=250,
                        help="rounds between snapshots. Selection can only "
                             "promote a snapshot it took, and the policy "
                             "oscillates between adjacent ones, so longer "
                             "stages want finer spacing.")
    parser.add_argument("--benchmark-seeds", type=int, default=None,
                        help="reporting seeds; defaults per scenario, see "
                             "EVALUATION_SEEDS")
    parser.add_argument("--select-seeds", type=int, default=None,
                        help="held-out seeds used to choose the best "
                             "checkpoint; defaults per scenario, see "
                             "EVALUATION_SEEDS")
    parser.add_argument("--no-select-best", action="store_true",
                        help="keep the final table instead of the best snapshot")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="make the run replayable: seeds the world's arena "
                             "generator and the agent's exploration RNG. Each "
                             "stage derives its own child seed from this one.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--promote", action="store_true",
                        help="copy the winning table over q_table.pkl, the "
                             "artifact the tournament agent loads")
    parser.add_argument("--wandb", action="store_true",
                        help="mirror train_metrics.csv to Weights & Biases after the run")
    parser.add_argument("--python", default=sys.executable)
    return parser


def artifact_dir(run_id: str) -> Path:
    """Per-run directory for this agent's tables and snapshots."""
    directory = AGENT_DIR / "checkpoints" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def stage_environment(args, stage, run_dir: Path, work_dir: Path) -> dict:
    """Environment variables configuring the agent for this stage."""
    environment = dict(os.environ)
    environment.update(
        AOT_QL_MODEL_FILE=str(work_dir / "q_table.pkl"),
        AOT_QL_SNAPSHOT_DIR=str(work_dir),
        AOT_QL_VARIANT=args.variant or stage.variant,
        AOT_QL_EPSILON_START=str(args.epsilon_start),
        AOT_QL_EPSILON_END=str(args.epsilon_end),
        # Decay over roughly the first fifth of the run, so most episodes are
        # spent refining rather than exploring.
        AOT_QL_EPSILON_DECAY_ROUNDS=str(max(1, (args.episodes or stage.episodes) // 5)),
        AOT_QL_ALPHA=str(args.alpha),
        AOT_QL_ALPHA_DECAY=str(args.alpha_decay),
        AOT_QL_ALPHA_MIN=str(args.alpha_min),
        AOT_QL_GAMMA=str(args.gamma),
        AOT_QL_N_STEP=str(args.n_step),
        AOT_QL_SAFETY_MODE=args.safety_mode,
        AOT_QL_OPPONENT_BOMB_LOOKAHEAD=str(args.opponent_bomb_lookahead).lower(),
        AOT_QL_USE_SYMMETRY=str(not args.no_symmetry).lower(),
        AOT_QL_USE_POTENTIAL_SHAPING=str(not args.no_shaping).lower(),
        AOT_QL_USE_CUSTOM_EVENTS=str(not args.no_custom_events).lower(),
        AOT_QL_DOUBLE_Q=str(not args.single_q).lower(),
        AOT_QL_METRICS_FILE=str(run_dir / "train_metrics.csv"),
        AOT_QL_CHECKPOINT_EVERY=str(args.checkpoint_every),
    )
    if args.seed is not None:
        # Only the training subprocess. Selection and the gate benchmark run in
        # the parent on their own fixed seed sets (blib.seeding), and those must
        # stay identical across arms or the arms are no longer comparable.
        environment["AOT_QL_SEED"] = str(stage_seed(args.seed, stage, "agent"))
    return environment


def stage_seed(base_seed: int, stage, role: str) -> int:
    """A stable child seed, distinct per stage and per stochastic component.

    Without the stage label every stage of a run would replay the same arena
    stream; without the role label the world and the agent would draw from the
    same seed, which correlates exploration with the map it explores.
    """
    return derive_seed(base_seed, "ql", stage.label, role)


def _git_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001 - provenance must never abort a run
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _file_digest(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_run_manifest(args, stage, run_id, run_dir, work_dir, episodes, resumed) -> None:
    """Record what produced this run, next to the numbers it produced.

    Without it a directory in `results/` cannot be attributed to an arm except
    by parsing its name, and the table a stage resumed from is not recorded
    anywhere. Both gaps bit the Phase 7-10 sweeps.
    """
    manifest = {
        "run_id": run_id,
        "stage": stage.index,
        "stage_label": stage.label,
        "scenario": stage.scenario,
        "opponents": list(stage.opponents),
        "episodes": episodes,
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "git_dirty": _git_is_dirty(),
        "python": args.python,
        "args": vars(args),
        "derived_seeds": {
            "world": stage_seed(args.seed, stage, "world") if args.seed is not None else None,
            "agent": stage_seed(args.seed, stage, "agent") if args.seed is not None else None,
            "selection_base": SELECTION_BASE_SEED,
            "reporting_base": MatchConfig.base_seed,
        },
        "resumed_from": {
            "path": str(resumed) if args.resume else None,
            "md5": _file_digest(resumed) if args.resume else None,
        },
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2, default=str))


def _git_is_dirty() -> Optional[bool]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def run_stage(args, stage, run_id: str) -> int:
    episodes = args.episodes or stage.episodes
    run_dir = REPO_ROOT / "results" / run_id / stage.label
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve here rather than in the parser: the count depends on the stage's
    # scenario, and `--all` runs stages with different ones under one args.
    args.select_seeds, args.benchmark_seeds = evaluation_seed_counts(args, stage)

    # Artifacts live under the run id, so concurrent runs (ablation sweeps) do
    # not fight over one table, and no run can replace the shipped q_table.pkl
    # unless --promote says so. Stages of one run share the directory, which is
    # how the curriculum chains.
    work_dir = artifact_dir(run_id)
    model = work_dir / "q_table.pkl"
    shipped = AGENT_DIR / "q_table.pkl"

    if not args.resume and model.exists():
        model.unlink()
        print(f"Removed {display_path(model)} (use --resume to continue from it)")
    if args.resume and not model.exists() and shipped.is_file():
        shutil.copy2(shipped, model)
        print(f"Seeded run from {display_path(shipped)}")

    print()
    print(describe_stage(stage))
    print(f"  episodes  : {episodes}")
    print(f"  results   : {display_path(run_dir)}")
    print(f"  artifacts : {display_path(work_dir)}")
    print(f"  eval seeds: {args.select_seeds} selection, "
          f"{args.benchmark_seeds} reporting")
    print()

    write_run_manifest(args, stage, run_id, run_dir, work_dir, episodes, model)

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
    if args.seed is not None:
        command += ["--seed", str(stage_seed(args.seed, stage, "world"))]

    logger = open_wandb(args, stage, run_id)
    # The stage env only reaches the training subprocess. Selection and the gate
    # benchmark run in *this* process, so without this they would evaluate every
    # arm under the default mask instead of the one it trained with.
    os.environ["AOT_QL_SAFETY_MODE"] = args.safety_mode
    os.environ["AOT_QL_OPPONENT_BOMB_LOOKAHEAD"] = str(args.opponent_bomb_lookahead).lower()
    try:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, env=stage_environment(args, stage, run_dir, work_dir)
        )
        if completed.returncode != 0:
            print(f"Training failed for {stage.label}", file=sys.stderr)
            return completed.returncode

        replay_metrics(logger, run_dir / "train_metrics.csv")

        if args.no_benchmark:
            return 0

        best = model
        if not args.no_select_best:
            best = select_best_checkpoint(args, stage, run_dir, work_dir)

        if args.promote:
            shutil.copy2(best, shipped)
            print(f"Promoted {best.name} -> {display_path(shipped)}")
        else:
            os.environ["AOT_QL_MODEL_FILE"] = str(best)

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
        values = summary["per_agent"][f"{AGENT}_0"]
        print(report_gate(stage, values))

        (run_dir / "benchmark.json").write_text(json.dumps(summary, indent=2, default=str))

        if logger is not None:
            # Benchmark numbers are what the report quotes, so they go to W&B as
            # summary values rather than another point on the training curve.
            logger.log({f"eval/{k}": v for k, v in values.items()
                        if isinstance(v, (int, float))})
        return 0
    finally:
        os.environ.pop("AOT_QL_MODEL_FILE", None)
        os.environ.pop("AOT_QL_SAFETY_MODE", None)
        if logger is not None:
            logger.close()


def open_wandb(args, stage, run_id: str):
    """W&B run for one stage, or None when --wandb is off.

    The agent writes its own CSV and knows nothing about W&B: training runs as a
    `main.py` subprocess, and adding a wandb import to the submitted directory
    would put a tournament dependency on a dev-only tool. So the CSV is the
    source of truth and this replays it afterwards.
    """
    if not args.wandb:
        return None
    logger = WandbLogger(
        project="attackontensor-bomberman",
        run_name=f"{run_id}-{stage.label}",
        config={
            "agent": AGENT, "stage": stage.index, "scenario": stage.scenario,
            "opponents": list(stage.opponents),
            "episodes": args.episodes or stage.episodes,
            "variant": args.variant or stage.variant,
            "alpha": args.alpha, "alpha_decay": args.alpha_decay,
            "alpha_min": args.alpha_min,
            "gamma": args.gamma, "n_step": args.n_step,
            "epsilon_start": args.epsilon_start, "epsilon_end": args.epsilon_end,
            "safety_mode": args.safety_mode,
            "opponent_bomb_lookahead": args.opponent_bomb_lookahead,
            "symmetry": not args.no_symmetry,
            "potential_shaping": not args.no_shaping,
            "custom_events": not args.no_custom_events,
            "double_q": not args.single_q,
            "seed": args.seed,
        },
    )
    if not logger.active:
        print("W&B unavailable; continuing with local CSV only.")
        return None
    return logger


def replay_metrics(logger, metrics_path: Path) -> None:
    """Push each row of the agent's training CSV to W&B, in round order."""
    if logger is None or not metrics_path.is_file():
        return
    with metrics_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        values = {}
        for key, raw in row.items():
            if raw in ("", None):
                continue
            try:
                values[key] = float(raw)
            except ValueError:
                pass
        step = int(values.pop("round", 0)) or None
        if values:
            logger.log({f"train/{k}": v for k, v in values.items()}, step=step)
    print(f"Mirrored {len(rows)} training rows to W&B.")


#: Selection uses a different base seed from the final report, so the table is
#: not chosen on the same arenas it is then scored on.
SELECTION_BASE_SEED = 777_000_1

#: Evaluation seed counts, per scenario: (selection, reporting).
#:
#: `classic` needs far more than the rest. Per-arena scores on one stage-3 run
#: ranged 1 to 19 with sd 4.75 on a mean of 7.33, which bootstraps to a +-2.67
#: half-width at 12 seeds and +-1.72 at 30 -- wider than any arm difference we
#: have measured. `loot-crate` and `coin-heaven` are much quieter objectives
#: (selection 42.83 vs benchmark 42.30 over 13 candidates), so they keep the
#: original counts and every Task-1/2 number stays comparable with the runs
#: already in `results/`.
EVALUATION_SEEDS = {
    "classic": (40, 60),
    None: (12, 20),
}


def evaluation_seed_counts(args, stage):
    """Resolve (select_seeds, benchmark_seeds), explicit flags winning."""
    select, benchmark = EVALUATION_SEEDS.get(stage.scenario, EVALUATION_SEEDS[None])
    if args.select_seeds is not None:
        select = args.select_seeds
    if args.benchmark_seeds is not None:
        benchmark = args.benchmark_seeds
    return select, benchmark


def select_best_checkpoint(args, stage, run_dir: Path, work_dir: Path) -> Path:
    """Score the run's snapshots on held-out seeds and return the best.

    A constant learning rate makes tabular Q-values track rather than converge,
    so the last table a run writes is not reliably its best -- measured on Task 2
    the final table scored 16.3 coins where a mid-run one scored 24.9. Selection
    runs on its own seed set (`SELECTION_BASE_SEED`); the gate afterwards uses
    the normal evaluation seeds, so the reported number is not the one the
    checkpoint was picked on.
    """
    snapshots = sorted(work_dir.glob("q_table_r*.pkl"))
    model = work_dir / "q_table.pkl"
    if model.exists():
        snapshots.append(model)

    if len(snapshots) < 2:
        print("\nOnly one checkpoint; skipping selection.")
        return model

    print(f"\nSelecting among {len(snapshots)} checkpoints "
          f"({args.select_seeds} held-out seeds each)...")

    # `snapshots` is in training order (ascending round, run's final table last),
    # so the index is a proxy for how much experience a table carries. Including
    # it in the key breaks score/survival ties toward the *more trained* table.
    # Without it a stable sort keeps glob order and ties go to the earliest
    # snapshot, which on Task 1 promoted a 92-state table over an equally
    # perfect 130-state one.
    results = []
    for index, path in enumerate(snapshots):
        # Resolved, not relative: model_path is AGENT_DIR / model_file, so a
        # relative value silently resolves under the agent directory and the
        # agent plays from an empty table with only a log line.
        os.environ["AOT_QL_MODEL_FILE"] = str(path.resolve())
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
        key = (values.get("score_mean", 0.0), values.get("survival_rate", 0.0), index)
        results.append((key, path, values))
        print(f"  {path.name:<28} score={values.get('score_mean', 0):6.2f} "
              f"surv={values.get('survival_rate', 0):5.1%} "
              f"suic={values.get('suicide_rate', 0):5.2f}")

    os.environ.pop("AOT_QL_MODEL_FILE", None)
    results.sort(key=lambda item: item[0], reverse=True)
    best_key, best_path, _ = results[0]

    print(f"  -> best {best_path.name} (score {best_key[0]:.2f})")

    (run_dir / "checkpoint_selection.json").write_text(
        json.dumps(
            [{"checkpoint": p.name, "score": k[0], "survival": k[1]} for k, p, _ in results],
            indent=2,
        )
    )
    return best_path


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
