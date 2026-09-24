from __future__ import annotations

import atexit
from typing import List, Optional

import numpy as np
import torch

from . import rewards as R
from . import tensorizer as T
from .dqn import double_dqn_loss, update_target_network
from .kit.actions import ACTION_INDEX
from .replay import ReplayBuffer


def setup_training(self):
    self.network.train()

    self.target_network = type(self.network)(
        in_channels=T.n_channels(self.config),
        spatial_size=self.config.spatial_size,
        channels=self.config.channels,
        hidden_dim=self.config.hidden_dim,
    ).to(self.device)

    self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=self.config.learning_rate,
    )

    payload = self.checkpoint_payload

    if isinstance(payload, dict):
        target_state = payload.get("target_state_dict")
        optimizer_state = payload.get("optimizer_state_dict")

        if target_state is not None:
            self.target_network.load_state_dict(target_state)
        else:
            update_target_network(self.network, self.target_network)

        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        self.training_updates = int(
            payload.get("training_updates", 0)
        )
    else:
        update_target_network(self.network, self.target_network)
        self.training_updates = 0

    self.target_network.eval()

    self.replay_buffer = ReplayBuffer(
        capacity=self.config.replay_capacity,
    )

    self.last_loss = None
    self.last_grad_norm = None
    self._last_processed_step: Optional[int] = None
    self._last_events_len = 0

    atexit.register(_save_on_exit, self)

    self.logger.info(
        f"DQN training start | "
        f"buffer={self.config.replay_capacity} "
        f"warmup={self.config.replay_warmup} "
        f"batch={self.config.batch_size} "
        f"lr={self.config.learning_rate} "
        f"gamma={self.config.gamma}"
    )


def _save_checkpoint(self) -> None:
    payload = {
        "state_dict": self.network.state_dict(),
        "target_state_dict": self.target_network.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "environment_steps": self.environment_steps,
        "training_updates": self.training_updates,
    }

    torch.save(payload, self.config.model_path)


def _save_on_exit(self) -> None:
    try:
        _save_checkpoint(self)
        self.logger.info(
            f"Final DQN save -> {self.config.model_path}"
        )
    except Exception as error:
        self.logger.error(f"Final DQN save failed: {error}")


def _transition_from_states(
    self,
    old_game_state: dict,
    self_action: str,
    reward: float,
    new_game_state: Optional[dict],
    done: bool,
) -> None:
    key = (
        old_game_state["round"],
        old_game_state["step"],
    )

    cached = self._step_cache.get(key)

    if cached is not None:
        state = cached["observation"]
        action = cached["action"]
    else:
        state = T.state_to_tensor(
            old_game_state,
            self.config,
        )
        action = ACTION_INDEX[self_action]

    if new_game_state is None:
        next_state = np.zeros(
            T.observation_shape(self.config),
            dtype=np.float32,
        )

        next_mask = np.ones(6, dtype=bool)

    else:
        next_state = T.state_to_tensor(
            new_game_state,
            self.config,
        )

        next_mask = T.action_mask_for(
            new_game_state,
            self.config,
        )

    self.replay_buffer.add(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        next_mask=next_mask,
    )


def _sample_batch(self):
    transitions = self.replay_buffer.sample(
        self.config.batch_size
    )

    states = torch.as_tensor(
        np.stack([t.state for t in transitions]),
        dtype=torch.float32,
        device=self.device,
    )

    actions = torch.as_tensor(
        [t.action for t in transitions],
        dtype=torch.long,
        device=self.device,
    )

    rewards = torch.as_tensor(
        [t.reward for t in transitions],
        dtype=torch.float32,
        device=self.device,
    )

    next_states = torch.as_tensor(
        np.stack([t.next_state for t in transitions]),
        dtype=torch.float32,
        device=self.device,
    )

    dones = torch.as_tensor(
        [t.done for t in transitions],
        dtype=torch.float32,
        device=self.device,
    )

    next_masks = torch.as_tensor(
        np.stack([t.next_mask for t in transitions]),
        dtype=torch.bool,
        device=self.device,
    )

    return (
        states,
        actions,
        rewards,
        next_states,
        dones,
        next_masks,
    )


def _maybe_update(self) -> None:
    if len(self.replay_buffer) < self.config.replay_warmup:
        return

    if len(self.replay_buffer) < self.config.batch_size:
        return

    if self.environment_steps % self.config.train_frequency != 0:
        return

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        next_masks,
    ) = _sample_batch(self)

    self.network.train()

    loss = double_dqn_loss(
        online_network=self.network,
        target_network=self.target_network,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        next_masks=next_masks,
        gamma=self.config.gamma,
    )

    self.optimizer.zero_grad()
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        self.network.parameters(),
        self.config.max_grad_norm,
    )

    self.optimizer.step()

    self.training_updates += 1
    self.last_loss = float(loss.item())
    self.last_grad_norm = float(grad_norm)

    if (
        self.environment_steps
        % self.config.target_update_interval
        == 0
    ):
        update_target_network(
            self.network,
            self.target_network,
        )

        self.logger.info(
            f"Target network updated at "
            f"step {self.environment_steps}"
        )

    if self.training_updates % 100 == 0:
        self.logger.info(
            f"DQN update {self.training_updates} | "
            f"step={self.environment_steps} "
            f"buffer={len(self.replay_buffer)} "
            f"loss={loss.item():.4f} "
            f"grad={float(grad_norm):.3f}"
        )


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    if old_game_state is None:
        return

    reward, _ = R.compute_reward(
        old_game_state,
        self_action,
        new_game_state,
        events,
        self.config,
    )

    _transition_from_states(
        self,
        old_game_state,
        self_action,
        reward,
        new_game_state,
        done=False,
    )

    _maybe_update(self)

    self._last_processed_step = old_game_state["step"]
    self._last_events_len = len(events)


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    if last_game_state is None:
        return

    already_seen = (
        self._last_processed_step
        == last_game_state["step"]
    )

    if already_seen:
        new_events = list(
            events[self._last_events_len:]
        )

        if new_events and len(self.replay_buffer) > 0:
            bonus = R.reward_from_events(
                new_events,
                self.config,
            )

            self.replay_buffer.buffer[-1].reward += bonus

        if len(self.replay_buffer) > 0:
            self.replay_buffer.buffer[-1].done = True

    else:
        reward, _ = R.compute_reward(
            last_game_state,
            last_action,
            None,
            events,
            self.config,
        )

        _transition_from_states(
            self,
            last_game_state,
            last_action,
            reward,
            None,
            done=True,
        )

        _maybe_update(self)

    round_number = last_game_state["round"]
    epsilon = self.config.epsilon_end + (
        self.config.epsilon_start - self.config.epsilon_end
    ) * max(
        0.0,
        1.0 - self.environment_steps / self.config.epsilon_decay_steps,
    )

    loss_text = (
        f"{self.last_loss:.4f}"
        if self.last_loss is not None
        else "N/A"
    )

    grad_text = (
        f"{self.last_grad_norm:.3f}"
        if self.last_grad_norm is not None
        else "N/A"
    )

    print(
        f"\n[DQN] Round {round_number} finished | "
        f"env_steps={self.environment_steps} | "
        f"replay={len(self.replay_buffer)} | "
        f"updates={self.training_updates} | "
        f"epsilon={epsilon:.4f} | "
        f"loss={loss_text} | "
        f"grad={grad_text}",
        flush=True,
    )

    self._last_processed_step = None
    self._last_events_len = 0
    self._step_cache.clear()
