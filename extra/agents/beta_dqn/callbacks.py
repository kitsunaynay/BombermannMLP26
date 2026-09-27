from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch

from . import tensorizer as T
from .config import DQNConfig
from .kit.actions import ACTIONS, WAIT
from .network import QNetwork

state_to_features = T.state_to_tensor


def setup(self):
    self.config = DQNConfig.load()

    torch.set_num_threads(max(1, int(self.config.torch_threads)))
    torch.manual_seed(self.config.seed)
    random.seed(self.config.seed)
    np.random.seed(self.config.seed)

    self.device = torch.device(self.config.device)

    self.network = QNetwork(
        in_channels=T.n_channels(self.config),
        spatial_size=self.config.spatial_size,
        channels=self.config.channels,
        hidden_dim=self.config.hidden_dim,
    ).to(self.device)

    path = self.config.model_path
    self.checkpoint_payload = None
    self.environment_steps = 0

    if path.is_file():
        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            state_dict = payload.get("state_dict", payload)
            self.network.load_state_dict(state_dict)

            if isinstance(payload, dict):
                self.checkpoint_payload = payload
                self.environment_steps = int(
                    payload.get("environment_steps", 0)
                )

            self.logger.info(
                f"Loaded DQN from {path} | "
                f"environment_steps={self.environment_steps}"
            )

        except Exception as error:
            self.logger.warning(
                f"Could not load {path} ({error}); using fresh weights."
            )

    elif not self.train:
        self.logger.warning(
            f"No checkpoint at {path}; playing from random weights."
        )

    self.network.eval()

    self.round_index = 0
    self._step_cache = {}

    self.logger.info(
        f"Beta DQN ready | "
        f"shape={T.observation_shape(self.config)} "
        f"safety={self.config.safety_mode} "
        f"params={self.network.n_parameters:,}"
    )


def epsilon_at_step(self) -> float:
    if not self.train:
        return 0.0

    fraction = min(
        self.environment_steps / self.config.epsilon_decay_steps,
        1.0,
    )

    return (
        self.config.epsilon_start
        + fraction
        * (self.config.epsilon_end - self.config.epsilon_start)
    )


def choose_action(self, game_state: dict) -> Tuple[str, int, dict]:
    observation = T.state_to_tensor(game_state, self.config)
    mask = T.action_mask_for(game_state, self.config)

    valid_actions = np.flatnonzero(mask)

    if len(valid_actions) == 0:
        valid_actions = np.array([WAIT])

    epsilon = epsilon_at_step(self)

    if self.train and random.random() < epsilon:
        index = int(random.choice(valid_actions))

    else:
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        mask_tensor = torch.as_tensor(
            mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        with torch.inference_mode():
            q_values = self.network(
                observation_tensor,
                action_mask=mask_tensor,
            )

        index = int(q_values.argmax(dim=1).item())

    step_data = {
        "observation": observation,
        "action": index,
        "mask": mask,
        "epsilon": epsilon,
    }

    return ACTIONS[index], index, step_data


def act(self, game_state: dict) -> str:
    if game_state is None:
        return ACTIONS[WAIT]

    if game_state["round"] != getattr(self, "round_index", 0):
        self.round_index = game_state["round"]
        self._step_cache.clear()

    try:
        name, _, step_data = choose_action(self, game_state)

        if self.train:
            key = (game_state["round"], game_state["step"])

            if len(self._step_cache) > 4:
                self._step_cache.clear()

            self._step_cache[key] = step_data

            self.environment_steps += 1

        return name

    except Exception as error:
        self.logger.exception(
            f"act() failed ({error}); falling back."
        )

        return _fallback_action(self, game_state)


def _fallback_action(self, game_state: dict) -> str:
    try:
        mask = T.action_mask_for(game_state, self.config)

        if mask.any():
            return ACTIONS[int(np.flatnonzero(mask)[0])]

    except Exception:
        pass

    return ACTIONS[WAIT]
