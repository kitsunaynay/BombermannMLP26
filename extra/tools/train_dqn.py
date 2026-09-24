"""
Train the CNN Double DQN agent with the shared Bomberman curriculum.

Example:
    python tools/train_dqn.py --stage 2 --workers 8 --total-steps 100000 \
        --seed 0 --safety-mode hard --run-id beta-dqn-s2-hard-s0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from agent_code.beta_dqn import tensorizer as T  # noqa: E402
from agent_code.beta_dqn.config import AGENT_DIR, DQNConfig  # noqa: E402
from agent_code.beta_dqn.dqn import (  # noqa: E402
    double_dqn_loss,
    update_target_network,
)
from agent_code.beta_dqn.kit.actions import ACTIONS  # noqa: E402
from agent_code.beta_dqn.network import QNetwork  # noqa: E402
from agent_code.beta_dqn.replay import ReplayBuffer  # noqa: E402
from blib.curriculum import STAGES, describe_stage, get_stage  # noqa: E402
from blib.factories import make_dqn_reward_fn, make_dqn_transform  # noqa: E402
from blib.paths import display_path  # noqa: E402
from blib.seeding import seed_everything  # noqa: E402
from blib.vec_env import DummyVecEnv, SubprocVecEnv, make_specs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=[stage.index for stage in STAGES],
        default=1,
    )
    parser.add_argument(
        "--opponents",
        nargs="*",
        default=None,
        help="override the stage's opponents",
    )

    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--safety-mode",
        choices=("none", "soft", "hard"),
        default=None,
    )
    parser.add_argument(
        "--bomb-gate",
        choices=("escape", "robust"),
        default=None,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--run-id",
        default=None,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume weights/optimizer/counters from a DQN checkpoint",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="use DummyVecEnv for debugging",
    )

    return parser


WORKER_VISIBLE = (
    "observation",
    "safety_mode",
    "survival_channels",
    "bomb_gate",
)


def apply_overrides(args) -> DQNConfig:
    config = DQNConfig.load()

    if args.safety_mode is not None:
        config.safety_mode = args.safety_mode

    if args.bomb_gate is not None:
        config.bomb_gate = args.bomb_gate

    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    config.device = args.device
    config.seed = args.seed

    for field in WORKER_VISIBLE:
        value = getattr(config, field)

        if isinstance(value, bool):
            value = "true" if value else "false"

        os.environ[
            f"AOT_DQN_{field.upper()}"
        ] = str(value)

    return config


def epsilon_at_step(
    config: DQNConfig,
    step: int,
) -> float:
    fraction = min(
        max(step, 0)
        / max(config.epsilon_decay_steps, 1),
        1.0,
    )

    return (
        config.epsilon_start
        + fraction
        * (
            config.epsilon_end
            - config.epsilon_start
        )
    )


def sample_batch(
    replay: ReplayBuffer,
    batch_size: int,
    device: str,
):
    batch = replay.sample(batch_size)

    states = torch.as_tensor(
        np.stack([transition.state for transition in batch]),
        dtype=torch.float32,
        device=device,
    )

    actions = torch.as_tensor(
        [transition.action for transition in batch],
        dtype=torch.long,
        device=device,
    )

    rewards = torch.as_tensor(
        [transition.reward for transition in batch],
        dtype=torch.float32,
        device=device,
    )

    next_states = torch.as_tensor(
        np.stack(
            [transition.next_state for transition in batch]
        ),
        dtype=torch.float32,
        device=device,
    )

    dones = torch.as_tensor(
        [transition.done for transition in batch],
        dtype=torch.float32,
        device=device,
    )

    next_masks = torch.as_tensor(
        np.stack(
            [transition.next_mask for transition in batch]
        ),
        dtype=torch.bool,
        device=device,
    )

    return (
        states,
        actions,
        rewards,
        next_states,
        dones,
        next_masks,
    )


def save_checkpoint(
    path: Path,
    online_network: QNetwork,
    target_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    config: DQNConfig,
    total_steps: int,
    training_updates: int,
    run_id: str,
    stage_index: int,
) -> None:
    payload = {
        "state_dict": online_network.state_dict(),
        "target_state_dict": target_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "environment_steps": total_steps,
        "training_updates": training_updates,
        "run_id": run_id,
        "stage": stage_index,
        "config": config.to_dict(),
    }

    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    online_network: QNetwork,
    target_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    device: str,
):
    # Replay is not stored in the checkpoint, so it starts empty after resume.
    payload = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    online_network.load_state_dict(
        payload["state_dict"]
    )

    if "target_state_dict" in payload:
        target_network.load_state_dict(
            payload["target_state_dict"]
        )
    else:
        update_target_network(
            online_network,
            target_network,
        )

    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(
            payload["optimizer_state_dict"]
        )

    total_steps = int(
        payload.get("environment_steps", 0)
    )
    training_updates = int(
        payload.get("training_updates", 0)
    )

    return total_steps, training_updates


def choose_actions(
    online_network: QNetwork,
    current,
    config: DQNConfig,
    total_steps: int,
):
    batch_observation = np.stack(
        [
            entry["observation"]
            for entry in current
        ]
    )

    batch_mask = np.stack(
        [
            entry["mask"]
            for entry in current
        ]
    )

    observation_tensor = torch.as_tensor(
        batch_observation,
        dtype=torch.float32,
        device=config.device,
    )

    mask_tensor = torch.as_tensor(
        batch_mask,
        dtype=torch.bool,
        device=config.device,
    )

    online_network.eval()

    with torch.no_grad():
        q_values = online_network(
            observation_tensor,
            mask_tensor,
        )

    greedy_actions = (
        q_values.argmax(dim=1)
        .cpu()
        .numpy()
    )

    epsilon = epsilon_at_step(
        config,
        total_steps,
    )

    action_indices = []

    for index, mask in enumerate(batch_mask):
        valid_actions = np.flatnonzero(mask)

        if len(valid_actions) == 0:
            action = ACTIONS.index("WAIT")

        elif np.random.random() < epsilon:
            action = int(
                np.random.choice(valid_actions)
            )

        else:
            action = int(
                greedy_actions[index]
            )

        action_indices.append(action)

    return (
        action_indices,
        batch_observation,
        batch_mask,
        epsilon,
    )


def train_update(
    online_network: QNetwork,
    target_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: DQNConfig,
):
    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        next_masks,
    ) = sample_batch(
        replay,
        config.batch_size,
        config.device,
    )

    online_network.train()

    optimizer.zero_grad(set_to_none=True)

    loss = double_dqn_loss(
        online_network=online_network,
        target_network=target_network,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        next_masks=next_masks,
        gamma=config.gamma,
    )

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        online_network.parameters(),
        config.max_grad_norm,
    )

    optimizer.step()

    return (
        float(loss.item()),
        float(grad_norm),
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    stage = get_stage(args.stage)

    if args.opponents is not None:
        stage = replace(
            stage,
            opponents=tuple(args.opponents),
        )

    config = apply_overrides(args)

    run_id = (
        args.run_id
        or make_run_id(
            f"dqn-task{stage.index}"
        )
    )

    seed_everything(args.seed)

    torch.set_num_threads(
        max(1, config.torch_threads)
    )

    checkpoint_dir = (
        AGENT_DIR
        / "checkpoints"
        / run_id
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(describe_stage(stage))
    print(
        f"\nrun_id={run_id} "
        f"workers={args.workers} "
        f"device={config.device}"
    )
    print(
        f"checkpoints="
        f"{display_path(checkpoint_dir)}"
    )
    print(
        f"observation={config.observation} "
        f"shape={T.observation_shape(config)} "
        f"safety={config.safety_mode}"
    )

    specs = make_specs(
        args.workers,
        base_seed=args.seed,
        opponents=stage.opponents,
        scenario=stage.scenario,
        reward_factory=make_dqn_reward_fn,
        transform_factory=make_dqn_transform,
    )

    vec_class = (
        DummyVecEnv
        if args.serial
        else SubprocVecEnv
    )

    envs = vec_class(specs)

    online_network = QNetwork(
        in_channels=T.n_channels(config),
        spatial_size=T.observation_shape(config)[1],
        channels=config.channels,
        hidden_dim=config.hidden_dim,
    ).to(config.device)

    target_network = QNetwork(
        in_channels=T.n_channels(config),
        spatial_size=T.observation_shape(config)[1],
        channels=config.channels,
        hidden_dim=config.hidden_dim,
    ).to(config.device)

    update_target_network(
        online_network,
        target_network,
    )

    target_network.eval()

    optimizer = torch.optim.Adam(
        online_network.parameters(),
        lr=config.learning_rate,
    )

    replay = ReplayBuffer(
        config.replay_capacity
    )

    total_steps = 0
    training_updates = 0

    if args.resume is not None:
        (
            total_steps,
            training_updates,
        ) = load_checkpoint(
            args.resume,
            online_network,
            target_network,
            optimizer,
            config.device,
        )

        print(
            f"resumed={display_path(args.resume)} "
            f"steps={total_steps:,} "
            f"updates={training_updates:,}"
        )
        print(
            "replay buffer starts empty after resume"
        )

    print(
        f"network={online_network.n_parameters:,} parameters "
        f"replay_capacity={config.replay_capacity:,}"
    )
    print(
        f"batch={config.batch_size} "
        f"warmup={config.replay_warmup:,} "
        f"train_frequency={config.train_frequency} "
        f"target_update={config.target_update_interval:,}"
    )
    print(
        f"epsilon={config.epsilon_start:.3f}"
        f"->{config.epsilon_end:.3f} "
        f"over {config.epsilon_decay_steps:,} steps"
    )

    # Continue until the requested total number of environment steps.
    target_total_steps = args.total_steps

    last_loss = None
    last_grad_norm = None

    episode_returns = np.zeros(
        args.workers,
        dtype=np.float64,
    )

    completed_returns = []

    next_log = (
        (total_steps // args.log_every) + 1
    ) * args.log_every

    next_checkpoint = (
        (total_steps // args.checkpoint_every) + 1
    ) * args.checkpoint_every

    try:
        current = envs.reset()

        print(
            f"Environment reset successful: "
            f"{len(current)} worker(s)"
        )

        while total_steps < target_total_steps:
            previous_steps = total_steps

            (
                action_indices,
                batch_observation,
                _batch_mask,
                epsilon,
            ) = choose_actions(
                online_network,
                current,
                config,
                total_steps,
            )

            next_current, rewards, dones, infos = (
                envs.step(
                    [
                        ACTIONS[action]
                        for action in action_indices
                    ]
                )
            )

            for index in range(args.workers):
                if dones[index]:
                    next_entry = infos[index].get(
                        "terminal_observation"
                    )
                else:
                    next_entry = next_current[index]

                # Terminal observations may be None. Since done=1,
                # the target never bootstraps from this placeholder.
                if next_entry is None:
                    next_observation = (
                        batch_observation[index]
                    )
                    next_mask = np.ones(
                        len(ACTIONS),
                        dtype=bool,
                    )
                else:
                    next_observation = next_entry[
                        "observation"
                    ]
                    next_mask = next_entry["mask"]

                replay.add(
                    state=batch_observation[index],
                    action=action_indices[index],
                    reward=float(rewards[index]),
                    next_state=next_observation,
                    done=bool(dones[index]),
                    next_mask=next_mask,
                )

                episode_returns[index] += float(
                    rewards[index]
                )

                if dones[index]:
                    completed_returns.append(
                        float(
                            episode_returns[index]
                        )
                    )
                    episode_returns[index] = 0.0

            current = next_current
            total_steps += args.workers
            # One vector step can cross several update intervals.

            first_due = (
                (
                    previous_steps
                    // config.train_frequency
                )
                + 1
            ) * config.train_frequency

            due_step = first_due

            while due_step <= total_steps:
                if (
                    due_step >= config.replay_warmup
                    and len(replay)
                    >= config.batch_size
                    and len(replay)
                    >= config.replay_warmup
                ):
                    (
                        last_loss,
                        last_grad_norm,
                    ) = train_update(
                        online_network,
                        target_network,
                        optimizer,
                        replay,
                        config,
                    )

                    training_updates += 1

                due_step += (
                    config.train_frequency
                )
            # Update target network at fixed environment-step intervals.

            if (
                total_steps
                // config.target_update_interval
                >
                previous_steps
                // config.target_update_interval
            ):
                update_target_network(
                    online_network,
                    target_network,
                )
                target_network.eval()
            # Logging

            if total_steps >= next_log:
                if completed_returns:
                    recent_returns = (
                        completed_returns[-50:]
                    )
                    mean_return = float(
                        np.mean(recent_returns)
                    )
                else:
                    mean_return = float("nan")

                loss_text = (
                    f"{last_loss:.4f}"
                    if last_loss is not None
                    else "N/A"
                )

                grad_text = (
                    f"{last_grad_norm:.3f}"
                    if last_grad_norm is not None
                    else "N/A"
                )

                print(
                    f"steps={total_steps:,} "
                    f"episodes={len(completed_returns):,} "
                    f"replay={len(replay):,} "
                    f"updates={training_updates:,} "
                    f"epsilon={epsilon:.4f} "
                    f"return50={mean_return:.2f} "
                    f"loss={loss_text} "
                    f"grad={grad_text}"
                )

                while next_log <= total_steps:
                    next_log += args.log_every
            # Save checkpoints during training.

            if total_steps >= next_checkpoint:
                checkpoint_path = (
                    checkpoint_dir
                    / f"dqn_step{total_steps:09d}.pt"
                )

                save_checkpoint(
                    checkpoint_path,
                    online_network,
                    target_network,
                    optimizer,
                    config,
                    total_steps,
                    training_updates,
                    run_id,
                    stage.index,
                )

                print(
                    f"checkpoint="
                    f"{display_path(checkpoint_path)}"
                )

                while next_checkpoint <= total_steps:
                    next_checkpoint += (
                        args.checkpoint_every
                    )
        # Save the final checkpoint and training summary.

        final_path = (
            checkpoint_dir
            / "dqn_final.pt"
        )

        save_checkpoint(
            final_path,
            online_network,
            target_network,
            optimizer,
            config,
            total_steps,
            training_updates,
            run_id,
            stage.index,
        )

        summary = {
            "run_id": run_id,
            "stage": stage.index,
            "scenario": stage.scenario,
            "opponents": list(stage.opponents),
            "seed": args.seed,
            "workers": args.workers,
            "environment_steps": total_steps,
            "training_updates": training_updates,
            "episodes": len(completed_returns),
            "epsilon": epsilon_at_step(
                config,
                total_steps,
            ),
            "last_loss": last_loss,
            "last_grad_norm": last_grad_norm,
            "mean_return_last_50": (
                float(
                    np.mean(
                        completed_returns[-50:]
                    )
                )
                if completed_returns
                else None
            ),
            "config": config.to_dict(),
        }

        summary_path = (
            checkpoint_dir
            / "summary.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("Training complete")
        print(
            f"steps={total_steps:,} "
            f"episodes={len(completed_returns):,} "
            f"updates={training_updates:,}"
        )
        print(
            f"final="
            f"{display_path(final_path)}"
        )
        print(
            f"summary="
            f"{display_path(summary_path)}"
        )

    finally:
        envs.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
