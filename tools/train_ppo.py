#!/usr/bin/env python3
"""Parallel PPO training.

Collects rollouts from many :class:`blib.fast_env.BomberEnv` instances running in
worker processes, batches the forward pass in the parent, and writes the same
``policy.pt`` the tournament agent loads. The brief permits this explicitly --
*"multiprocessing or anything else that comes to your mind to improve training is
perfectly fine"* -- while the submitted agent stays single-process.

On hardware: this workload is environment-bound, not GPU-bound. The network is
about 17M multiply-accumulates per forward pass, which a CPU handles in under a
millisecond, whereas one step of the game with three ``rule_based_agent``
opponents costs roughly a millisecond of pure Python *per opponent*. So the way
to use several GPUs here is to run several independent seeds or hyperparameter
configurations at once (``--device cuda:N --seed S``), not to shard one tiny
network across them.

Examples::

    python tools/train_ppo.py --stage 1 --workers 8 --total-steps 500000
    python tools/train_ppo.py --stage 4 --workers 16 --device cuda:0 --wandb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from agent_code.attackontensor_ppo import tensorizer as T  # noqa: E402
from agent_code.attackontensor_ppo.config import PPOConfig  # noqa: E402
from agent_code.attackontensor_ppo.kit.actions import ACTIONS  # noqa: E402
from agent_code.attackontensor_ppo.network import build_network  # noqa: E402
from agent_code.attackontensor_ppo.ppo import PPOLearner, RolloutBuffer, compute_gae  # noqa: E402
from blib.curriculum import STAGES, describe_stage, get_stage  # noqa: E402
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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--steps-per-worker", type=int, default=128,
                        help="rollout length per worker per iteration")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue from policy.pt")
    parser.add_argument("--observation", choices=("global", "ego"), default=None)
    parser.add_argument("--safety-mode", choices=("none", "soft", "hard"), default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coefficient", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=20, help="iterations")
    parser.add_argument("--log-every", type=int, default=1, help="iterations")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--serial", action="store_true",
                        help="use DummyVecEnv (no processes) for debugging")
    return parser


def apply_overrides(args) -> PPOConfig:
    """CLI overrides win over config.json and the environment."""
    config = PPOConfig.load()
    if args.observation:
        config.observation = args.observation
    if args.safety_mode:
        config.safety_mode = args.safety_mode
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.entropy_coefficient is not None:
        config.entropy_coefficient = args.entropy_coefficient
    config.device = args.device
    config.seed = args.seed
    return config


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    stage = get_stage(args.stage)
    config = apply_overrides(args)

    run_id = args.run_id or make_run_id(f"ppo-task{stage.index}")
    seed_everything(args.seed)
    # The parent only ever does batched forward/backward passes; letting torch
    # also fan out across cores would fight the environment workers for them.
    torch.set_num_threads(max(1, min(4, args.workers)))

    print(describe_stage(stage))
    print(f"\nrun_id={run_id}  workers={args.workers}  device={args.device}")
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
    network = build_network(config, T.N_CHANNELS)
    if args.resume and config.model_path.is_file():
        payload = torch.load(config.model_path, map_location="cpu", weights_only=False)
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
                learner.save(config.model_path, extra={"run_id": run_id, "iteration": iteration})
                print(f"  checkpoint -> {config.model_path.name}")

    except KeyboardInterrupt:
        print("\nInterrupted; saving before exit.")
    finally:
        learner.save(config.model_path, extra={"run_id": run_id})
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

    print(f"\nSaved {display_path(config.model_path)}")
    print(f"Results under results/{run_id}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
