"""
Select the best Double DQN checkpoint for each training seed.

Each checkpoint is evaluated on the same held-out arena seeds using the
ordinary beta_dqn inference path. Selection follows the same logic as the
PPO trainer: stage gate first, then score, survival, and training progress.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from blib.benchmark import MatchConfig, benchmark_fast  # noqa: E402
from blib.curriculum import get_stage  # noqa: E402

SELECTION_BASE_SEED = 883_000_2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="DQN run IDs under agent_code/beta_dqn/checkpoints",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--select-seeds",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--select-by",
        choices=("score", "gate"),
        default="gate",
    )

    return parser


def select_run(
    run_id: str,
    stage,
    select_seeds: int,
    select_by: str,
):
    checkpoint_dir = (
        REPO_ROOT
        / "agent_code"
        / "beta_dqn"
        / "checkpoints"
        / run_id
    )

    snapshots = sorted(
        checkpoint_dir.glob("dqn_step*.pt")
    )

    final_path = checkpoint_dir / "dqn_final.pt"

    # dqn_final.pt and dqn_step100000.pt contain the same final
    # training state in our trainer. Do not evaluate it twice.
    if final_path.is_file():
        final_step = None

        if snapshots:
            final_step = snapshots[-1]

        if final_step is None:
            snapshots.append(final_path)

    if not snapshots:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir}"
        )

    print()
    print("=" * 72)
    print(f"Run: {run_id}")
    print(
        f"Scoring {len(snapshots)} checkpoints on "
        f"{select_seeds} held-out seeds"
    )
    print("=" * 72)

    agent = "beta_dqn"
    results = []

    for index, path in enumerate(snapshots):
        os.environ["AOT_DQN_MODEL_FILE"] = str(
            path.resolve()
        )

        summary = benchmark_fast(
            MatchConfig(
                agents=[
                    agent,
                    *stage.opponents,
                ],
                scenario=stage.scenario,
                n_seeds=select_seeds,
                base_seed=SELECTION_BASE_SEED,
                track_latency=False,
            )
        )

        values = summary["per_agent"][
            f"{agent}_0"
        ]

        passed, _ = stage.gate.evaluate(values)

        key = (
            (
                1 if passed else 0
            )
            if select_by == "gate"
            else 0,
            values.get("score_mean", 0.0),
            values.get("survival_rate", 0.0),
            index,
        )

        results.append(
            (
                key,
                path,
                values,
                passed,
            )
        )

        print(
            f"  {path.name:<28} "
            f"score={values.get('score_mean', 0):6.2f} "
            f"surv={values.get('survival_rate', 0):5.1%} "
            f"suic={values.get('suicide_rate', 0):5.2f} "
            f"crates={values.get('crates_mean', 0):6.2f} "
            f"gate={'PASS' if passed else 'fail'}"
        )

    os.environ.pop(
        "AOT_DQN_MODEL_FILE",
        None,
    )

    results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_key, best_path, _, best_passed = results[0]

    print(
        f"  -> best {best_path.name} "
        f"(score {best_key[1]:.2f}, "
        f"gate {'PASS' if best_passed else 'fail'}, "
        f"select_by={select_by})"
    )

    selection = []

    for key, path, values, passed in results:
        selection.append(
            {
                "checkpoint": path.name,
                "score": key[1],
                "survival": key[2],
                "gate_pass": bool(passed),
                "suicide": values.get(
                    "suicide_rate"
                ),
                "win": values.get(
                    "win_rate"
                ),
                "kills": values.get(
                    "kills_mean"
                ),
                "crates": values.get(
                    "crates_mean"
                ),
            }
        )

    output_path = (
        checkpoint_dir
        / "checkpoint_selection.json"
    )

    output_path.write_text(
        json.dumps(
            selection,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "run_id": run_id,
        "best_checkpoint": best_path.name,
        "best_checkpoint_path": str(
            best_path.resolve()
        ),
        "score": best_key[1],
        "survival": best_key[2],
        "gate_pass": bool(best_passed),
    }


def main() -> int:
    args = build_parser().parse_args()

    stage = get_stage(args.stage)

    all_results = []

    try:
        for run_id in args.runs:
            result = select_run(
                run_id=run_id,
                stage=stage,
                select_seeds=args.select_seeds,
                select_by=args.select_by,
            )

            all_results.append(result)

    finally:
        os.environ.pop(
            "AOT_DQN_MODEL_FILE",
            None,
        )

    print()
    print("=" * 72)
    print("SELECTED CHECKPOINTS")
    print("=" * 72)

    for result in all_results:
        print(
            f"{result['run_id']}: "
            f"{result['best_checkpoint']} | "
            f"score={result['score']:.2f} | "
            f"survival={result['survival']:.1%} | "
            f"gate="
            f"{'PASS' if result['gate_pass'] else 'fail'}"
        )

    output_dir = (
        REPO_ROOT
        / "results"
        / "dqn_checkpoint_selection"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "stage2_selection.json"
    )

    output_path.write_text(
        json.dumps(
            all_results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
