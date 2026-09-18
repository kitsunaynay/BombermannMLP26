"""
    python tools/train_ppo.py --stage 1 --workers 8 --total-steps 500000
    python tools/train_ppo.py --stage 4 --workers 16 --device cuda:0 --wandb
    python tools/train_ppo.py --stage 4 --workers 16 --promote --benchmark-seeds 30
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from agent_code.attackontensor_ppo import tensorizer as T  # noqa: E402
from agent_code.attackontensor_ppo.config import AGENT_DIR, PPOConfig  # noqa: E402
from agent_code.attackontensor_ppo.kit.actions import ACTIONS  # noqa: E402
from agent_code.attackontensor_ppo.network import build_network  # noqa: E402
from agent_code.attackontensor_ppo.ppo import PPOLearner, RolloutBuffer, compute_gae  # noqa: E402
from blib.benchmark import MatchConfig, benchmark_fast, format_table  # noqa: E402
from blib.curriculum import STAGES, describe_stage, get_stage, report_gate  # noqa: E402
from blib.factories import make_ppo_reward_fn, make_ppo_transform  # noqa: E402
from blib.paths import display_path  # noqa: E402
from blib.seeding import seed_everything  # noqa: E402
from blib.tracking import make_logger, make_run_id  # noqa: E402
from blib.vec_env import DummyVecEnv, SubprocVecEnv, make_specs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=int, choices=[s.index for s in STAGES], default=1)
    parser.add_argument("--opponents", nargs="*", default=None,
                        help="override the stage's opponents (blib.opponents.OpponentSpec "
                             "strings, e.g. rule_based_agent:nobomb or "
                             "attackontensor_ppo@/path/to/frozen.pt)")
    parser.add_argument("--select-by", choices=("score", "gate"), default="score",
                        help="rank snapshots by raw score, or by passing the stage gate "
                             "first and score second")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--steps-per-worker", type=int, default=128,
                        help="rollout length per worker per iteration")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue from policy.pt")
    parser.add_argument("--observation", choices=("global", "ego"), default=None)
    parser.add_argument("--safety-mode", choices=("none", "soft", "hard"), default=None)
    parser.add_argument("--bomb-gate", choices=("escape", "robust"), default=None,
                        help="how the hard mask judges BOMB (kit.safety.BOMB_GATES)")
    parser.add_argument("--survival-channels", dest="survival_channels", action="store_true",
                        default=None, help="add the four survival-profile input planes")
    parser.add_argument("--no-survival-channels", dest="survival_channels",
                        action="store_false", help="base 13 planes only")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coefficient", type=float, default=None)
    parser.add_argument("--event-reward", action="append", default=None, metavar="EVENT=VALUE",
                        help="override one event's reward weight (repeatable); reaches "
                             "the workers as AOT_PPO_EVENT_REWARDS")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch threads for the learner's update (default: "
                             "min(4, workers)). On a CPU-only node the update is "
                             "the bottleneck; 16 threads is ~2.2x faster than 4")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="iterations")
    parser.add_argument("--log-every", type=int, default=1, help="iterations")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--select-seeds", type=int, default=12,
                        help="held-out seeds used to score each snapshot")
    parser.add_argument("--no-select-best", action="store_true",
                        help="skip scoring the snapshots after training")
    parser.add_argument("--promote", action="store_true",
                        help="copy the winning snapshot over policy.pt, the "
                             "artifact the tournament agent loads")
    parser.add_argument("--benchmark-seeds", type=int, default=0,
                        help="if set, evaluate the promoted policy and print the stage gate")
    parser.add_argument("--serial", action="store_true",
                        help="use DummyVecEnv (no processes) for debugging")
    return parser


#: Overrides that the *workers* need, not just the learner. `make_ppo_transform`
#: and `make_ppo_reward_fn` run inside each worker process and rebuild their own
#: `PPOConfig.load()`, so anything they consume has to travel as an environment
#: variable or it never leaves the parent. `learning_rate` and
#: `entropy_coefficient` are deliberately absent: they are used only by the
#: learner, in this process.
WORKER_VISIBLE = ("observation", "safety_mode", "survival_channels", "bomb_gate")


def parse_event_rewards(items) -> dict:
    """``["OPPONENT_ELIMINATED=3", "TRAPPED_OPPONENT=1"]`` -> ``{...: float}``."""
    parsed = {}
    for item in items or ():
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise SystemExit(f"--event-reward expects EVENT=VALUE, got {item!r}")
        parsed[name.strip()] = float(value)
    return parsed


def apply_overrides(args) -> PPOConfig:
    """CLI overrides win over config.json and the environment.

    Exports the worker-visible ones back into ``os.environ`` before returning.
    Without that, ``--safety-mode hard`` changed this process's config object
    and nothing else: every worker built its action mask from the default
    ``soft``, so the flag silently did nothing to training. Two arms launched
    on 2026-09-09 produced bit-identical curves, which is how it was found.
    """
    config = PPOConfig.load()
    if args.observation:
        config.observation = args.observation
    if args.safety_mode:
        config.safety_mode = args.safety_mode
    if args.survival_channels is not None:
        config.survival_channels = args.survival_channels
    if args.bomb_gate:
        config.bomb_gate = args.bomb_gate
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.entropy_coefficient is not None:
        config.entropy_coefficient = args.entropy_coefficient
    config.device = args.device
    config.seed = args.seed

    # Set before the vec env is built, so forked and spawned workers both see it.
    for field in WORKER_VISIBLE:
        os.environ[f"AOT_PPO_{field.upper()}"] = str(getattr(config, field))

    # Event weights are consumed by the reward function, which also runs in
    # the workers. The override travels as JSON on top of whatever the
    # environment already carried, so a stale variable cannot leak in either.
    overrides = parse_event_rewards(args.event_reward)
    if overrides:
        config.event_rewards.update(overrides)
        os.environ["AOT_PPO_EVENT_REWARDS"] = json.dumps(overrides)
    else:
        os.environ.pop("AOT_PPO_EVENT_REWARDS", None)

    return config


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    stage = get_stage(args.stage)
    if args.opponents is not None:
        stage = replace(stage, opponents=tuple(args.opponents))
    config = apply_overrides(args)

    run_id = args.run_id or make_run_id(f"ppo-task{stage.index}")
    seed_everything(args.seed)
    # The parent only ever does batched forward/backward passes; letting torch
    # also fan out across cores would fight the environment workers for them.
    torch.set_num_threads(max(1, args.threads if args.threads else min(4, args.workers)))

    # Snapshots live under the run id, so concurrent runs cannot overwrite each
    # other and an exploratory run cannot clobber policy.pt. Promotion into
    # policy.pt is a separate, explicit step (--promote).
    checkpoint_dir = AGENT_DIR / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(describe_stage(stage))
    print(f"\nrun_id={run_id}  workers={args.workers}  device={args.device}  "
          f"select_by={args.select_by}")
    print(f"checkpoints={display_path(checkpoint_dir)}")
    print(f"observation={config.observation} shape={T.observation_shape(config)} "
          f"safety={config.safety_mode}\n")

    # -- environments ------------------------------------------------------
    specs = make_specs(
        args.workers,
        base_seed=args.seed,
        opponents=stage.opponents,
        scenario=stage.scenario,
        reward_factory=make_ppo_reward_fn,
        transform_factory=make_ppo_transform,
    )
    vec_class = DummyVecEnv if args.serial else SubprocVecEnv
    envs = vec_class(specs)

    # -- learner -----------------------------------------------------------
    network = build_network(config, T.n_channels(config))
    if args.resume and config.model_path.is_file():
        payload = torch.load(config.model_path, map_location="cpu", weights_only=False)
        found = (payload.get("network") or {}).get("in_channels")
        if found is not None and int(found) != T.n_channels(config):
            raise SystemExit(
                f"{config.model_path} has {found} input planes but this run is "
                f"configured for {T.n_channels(config)}; pass "
                f"--{'' if found == T.N_CHANNELS else 'no-'}survival-channels"
            )
        network.load_state_dict(payload["state_dict"])
        print(f"Resumed from {config.model_path}")

    learner = PPOLearner(network, config, device=args.device)
    device = torch.device(args.device)

    logger = make_logger(run_id, config=config.to_dict(), use_wandb=args.wandb)

    observation_shape = T.observation_shape(config)
    n_actions = len(ACTIONS)
    steps_per_iteration = args.steps_per_worker * args.workers
    n_iterations = max(1, args.total_steps // steps_per_iteration)

    # Per-worker streams, so GAE can be computed independently down each column.
    observations = np.zeros((args.steps_per_worker, args.workers, *observation_shape), np.float32)
    masks = np.ones((args.steps_per_worker, args.workers, n_actions), bool)
    actions = np.zeros((args.steps_per_worker, args.workers), np.int64)
    log_probs = np.zeros((args.steps_per_worker, args.workers), np.float32)
    values = np.zeros((args.steps_per_worker, args.workers), np.float32)
    rewards = np.zeros((args.steps_per_worker, args.workers), np.float32)
    dones = np.zeros((args.steps_per_worker, args.workers), bool)

    current = envs.reset()
    episode_returns = np.zeros(args.workers, np.float32)
    finished_episodes = []
    total_steps = 0
    started = time.perf_counter()

    try:
        for iteration in range(1, n_iterations + 1):
            learner.progress = (iteration - 1) / n_iterations
            network.eval()

            for step in range(args.steps_per_worker):
                batch_observation = np.stack([entry["observation"] for entry in current])
                batch_mask = np.stack([entry["mask"] for entry in current])

                observation_tensor = torch.as_tensor(batch_observation, device=device)
                mask_tensor = torch.as_tensor(batch_mask, device=device)
                action, log_prob, value = network.act(observation_tensor, mask_tensor)

                action_np = action.cpu().numpy()
                observations[step] = batch_observation
                masks[step] = batch_mask
                actions[step] = action_np
                log_probs[step] = log_prob.cpu().numpy()
                values[step] = value.cpu().numpy()

                current, reward, done, infos = envs.step([ACTIONS[a] for a in action_np])
                rewards[step] = reward
                dones[step] = done

                episode_returns += reward
                for index, is_done in enumerate(done):
                    if is_done:
                        stats = infos[index].get("round_statistics", {})
                        finished_episodes.append(
                            {**stats, "episode_return": float(episode_returns[index])}
                        )
                        episode_returns[index] = 0.0

                total_steps += args.workers

            # -- advantages, per worker stream -----------------------------
            batch_observation = np.stack([entry["observation"] for entry in current])
            batch_mask = np.stack([entry["mask"] for entry in current])
            with torch.inference_mode():
                _, bootstrap = network(
                    torch.as_tensor(batch_observation, device=device),
                    torch.as_tensor(batch_mask, device=device),
                )
            bootstrap = bootstrap.cpu().numpy()

            advantages = np.zeros_like(rewards, dtype=np.float64)
            returns = np.zeros_like(rewards, dtype=np.float64)
            for worker in range(args.workers):
                advantages[:, worker], returns[:, worker] = compute_gae(
                    rewards[:, worker],
                    values[:, worker],
                    dones[:, worker],
                    float(bootstrap[worker]),
                    config.gamma,
                    config.gae_lambda,
                )

            # -- update ----------------------------------------------------
            buffer = RolloutBuffer(steps_per_iteration, observation_shape)
            buffer.observations[:] = observations.reshape(steps_per_iteration, *observation_shape)
            buffer.masks[:] = masks.reshape(steps_per_iteration, n_actions)
            buffer.actions[:] = actions.reshape(-1)
            buffer.log_probs[:] = log_probs.reshape(-1)
            buffer.values[:] = values.reshape(-1)
            buffer.rewards[:] = rewards.reshape(-1)
            buffer.dones[:] = dones.reshape(-1)
            buffer.size = steps_per_iteration

            network.train()
            metrics = learner.update(buffer, advantages.reshape(-1), returns.reshape(-1))

            # -- report ----------------------------------------------------
            if iteration % args.log_every == 0:
                elapsed = time.perf_counter() - started
                recent = finished_episodes[-50:]
                row = {
                    "iteration": iteration,
                    "total_steps": total_steps,
                    "steps_per_second": total_steps / max(elapsed, 1e-9),
                    "episodes": len(finished_episodes),
                    **metrics.as_dict(),
                }
                if recent:
                    for key in ("score", "coins", "crates", "kills", "suicides", "steps",
                                "survived", "episode_return"):
                        present = [float(item.get(key, 0.0)) for item in recent if key in item]
                        if present:
                            row[f"ep_{key}"] = float(np.mean(present))

                logger.log(row, step=total_steps)
                print(
                    f"it {iteration:4d}/{n_iterations} | steps {total_steps:>8,} "
                    f"({row['steps_per_second']:6.0f}/s) | "
                    f"ret {row.get('ep_episode_return', 0):+7.2f} | "
                    f"score {row.get('ep_score', 0):5.2f} | "
                    f"H {metrics.entropy:.3f} | KL {metrics.approx_kl:.4f} | "
                    f"EV {metrics.explained_variance:+.3f}",
                    flush=True,
                )

            if iteration % args.checkpoint_every == 0:
                snapshot = checkpoint_dir / f"policy_it{iteration:06d}.pt"
                learner.save(snapshot, extra={"run_id": run_id, "iteration": iteration})
                print(f"  checkpoint -> {snapshot.name}")

    except KeyboardInterrupt:
        print("\nInterrupted; saving before exit.")
    finally:
        final_path = checkpoint_dir / "policy_final.pt"
        learner.save(final_path, extra={"run_id": run_id})
        logger.close()
        envs.close()

    summary = {
        "run_id": run_id,
        "stage": stage.label,
        "total_steps": total_steps,
        "episodes": len(finished_episodes),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = REPO_ROOT / "results" / run_id / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))

    print(f"\nSaved {display_path(final_path)}")

    best = final_path
    if not args.no_select_best:
        best = select_best_checkpoint(args, stage, checkpoint_dir, run_dir=output.parent)

    if args.promote:
        shutil.copy2(best, config.model_path)
        print(f"Promoted {best.name} -> {display_path(config.model_path)}")
        if args.benchmark_seeds:
            evaluate_promoted(args, stage, output.parent)
    else:
        print(f"\npolicy.pt untouched. To ship this run:\n"
              f"  cp {display_path(best)} {display_path(config.model_path)}")

    print(f"Results under results/{run_id}/")
    return 0


#: Selection runs on its own seeds, so a snapshot is not chosen on the same
#: arenas it is later reported on.
SELECTION_BASE_SEED = 883_000_2


def select_best_checkpoint(args, stage, checkpoint_dir: Path, run_dir: Path) -> Path:
    """Score every snapshot on held-out seeds and return the best.

    The policy a PPO run ends on is not reliably the best one it passed through,
    for the same reason it is not for the Q-learning table. Scoring the saved
    artifacts is the only way to find that out.
    """
    snapshots = sorted(checkpoint_dir.glob("policy_it*.pt"))
    final_path = checkpoint_dir / "policy_final.pt"
    if final_path.is_file():
        snapshots.append(final_path)

    if len(snapshots) < 2:
        print("Only one checkpoint; skipping selection.")
        return snapshots[0] if snapshots else final_path

    print(f"\nScoring {len(snapshots)} checkpoints on {args.select_seeds} held-out seeds...")
    agent = "attackontensor_ppo"
    # Training order, final policy last, so the index stands in for how much
    # training a snapshot carries and breaks ties toward the more trained one.
    results = []
    for index, path in enumerate(snapshots):
        # The agent reads its weights path from the environment, so each
        # candidate is evaluated through the ordinary inference path.
        # Resolved, not relative: model_path is AGENT_DIR / model_file, so a
        # relative value silently resolves under the agent directory and the
        # agent plays from an empty table with only a log line.
        os.environ["AOT_PPO_MODEL_FILE"] = str(path.resolve())
        summary = benchmark_fast(
            MatchConfig(
                agents=[agent, *stage.opponents],
                scenario=stage.scenario,
                n_seeds=args.select_seeds,
                base_seed=SELECTION_BASE_SEED,
                track_latency=False,
            )
        )
        values = summary["per_agent"][f"{agent}_0"]
        passed, _ = stage.gate.evaluate(values)
        # gate mode: a snapshot that clears the stage's gate always outranks one
        # that doesn't, regardless of score (avoids picking a high-scoring but
        # high-suicide snapshot).
        key = (
            (1 if passed else 0) if args.select_by == "gate" else 0,
            values.get("score_mean", 0.0),
            values.get("survival_rate", 0.0),
            index,
        )
        results.append((key, path, values))
        print(f"  {path.name:<26} score={values.get('score_mean', 0):6.2f} "
              f"surv={values.get('survival_rate', 0):5.1%} "
              f"suic={values.get('suicide_rate', 0):5.2f} "
              f"gate={'PASS' if passed else 'fail'}")

    os.environ.pop("AOT_PPO_MODEL_FILE", None)
    results.sort(key=lambda item: item[0], reverse=True)
    best_key, best_path, _ = results[0]
    print(f"  -> best {best_path.name} (score {best_key[1]:.2f}, "
          f"gate {'PASS' if best_key[0] else 'fail'}, select_by={args.select_by})")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint_selection.json").write_text(
        json.dumps(
            [{"checkpoint": p.name, "score": k[1], "survival": k[2],
              "gate_pass": bool(stage.gate.evaluate(v)[0]),
              "suicide": v.get("suicide_rate"), "win": v.get("win_rate"),
              "kills": v.get("kills_mean")} for k, p, v in results],
            indent=2,
        )
    )
    return best_path


def evaluate_promoted(args, stage, run_dir: Path) -> None:
    """Score policy.pt on the reporting seeds and print the stage gate."""
    agent = "attackontensor_ppo"
    print(f"\nEvaluating promoted policy over {args.benchmark_seeds} seeds...")
    summary = benchmark_fast(
        MatchConfig(
            agents=[agent, *stage.opponents],
            scenario=stage.scenario,
            n_seeds=args.benchmark_seeds,
        )
    )
    print()
    print(format_table(summary))
    print()
    print(report_gate(stage, summary["per_agent"][f"{agent}_0"]))
    (run_dir / "benchmark.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
