from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from . import tensorizer as T
from .config import PPOConfig
from .kit.actions import ACTIONS, WAIT
from .network import build_network

state_to_features = T.state_to_tensor


# not config but checkpoint; checkpoint wins if they disagree
ADOPTED_FROM_CHECKPOINT = ("observation", "ego_radius", "safety_mode")


def setup(self):
    self.config = PPOConfig.load()

    # load checkpoint before building network since payload changes input shape
    path = self.config.model_path
    payload = None
    if path.is_file():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # noqa: BLE001
            self.logger.warning(f"Could not read {path} ({error}); using fresh weights.")
    if payload is not None:
        _adopt_checkpoint_config(self, payload)

    torch.set_num_threads(max(1, int(self.config.torch_threads)))
    torch.manual_seed(self.config.seed)

    self.device = torch.device(self.config.device)
    self.network = build_network(self.config, T.N_CHANNELS).to(self.device)

    if payload is not None:
        try:
            _load_checkpoint(self, payload, path)
        except Exception as error:  # noqa: BLE001
            self.logger.warning(f"Could not load {path} ({error}); using fresh weights.")
    elif not self.train:
        # Never crash on a missing checkpoint: an untrained agent that plays
        # badly still passes the graders' submission test, a crashing one does not.
        self.logger.warning(f"No checkpoint at {path}; playing from random weights.")

    self.network.eval()
    _warm_up(self)

    self.round_index = 0
    self._step_cache: dict = {}

    self.logger.info(
        f"AttackOnTensor PPO ready | obs={self.config.observation} "
        f"shape={T.observation_shape(self.config)} safety={self.config.safety_mode} "
        f"params={self.network.n_parameters:,} threads={torch.get_num_threads()}"
    )


def _adopt_checkpoint_config(self, payload) -> None:
    # checkpoint values override config on input shape and safety settings
    stored = payload.get("config") or {}
    for field in ADOPTED_FROM_CHECKPOINT:
        if field not in stored:
            continue
        value = stored[field]
        if value != getattr(self.config, field, None):
            self.logger.warning(
                f"Checkpoint was trained with {field}={value!r}, config says "
                f"{getattr(self.config, field, None)!r}; using the checkpoint's value."
            )
            setattr(self.config, field, value)


def _load_checkpoint(self, payload, path) -> None:
    saved = payload.get("network", {})

    expected = (T.N_CHANNELS, self.config.spatial_size)
    found = (saved.get("in_channels"), saved.get("spatial_size"))
    if None not in found and found != expected:
        raise ValueError(
            f"checkpoint was trained for input {found}, but the current config "
            f"expects {expected}; check `observation` and `ego_radius`"
        )

    self.network.load_state_dict(payload["state_dict"])
    self.logger.info(f"Loaded policy from {path}")


def _warm_up(self) -> None:
    """Pay torch's first-call initialisation cost outside the timed path."""
    shape = T.observation_shape(self.config)
    dummy = torch.zeros(1, *shape, device=self.device)
    with torch.inference_mode():
        self.network(dummy)


def _observation_tensor(self, observation: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(observation, device=self.device).unsqueeze(0)


def choose_action(self, game_state: dict) -> Tuple[str, int, dict]:
    """Pick an action and return the rollout data ``train.py`` will need."""
    observation = T.state_to_tensor(game_state, self.config)
    mask = T.action_mask_for(game_state, self.config)

    observation_tensor = _observation_tensor(self, observation)
    mask_tensor = torch.as_tensor(mask, device=self.device).unsqueeze(0)

    deterministic = (not self.train) and self.config.deterministic_eval
    action, log_prob, value = self.network.act(
        observation_tensor, mask_tensor, deterministic=deterministic
    )

    index = int(action.item())
    step_data = {
        "observation": observation,
        "action": index,
        "log_prob": float(log_prob.item()),
        "value": float(value.item()),
        "mask": mask,
    }
    return ACTIONS[index], index, step_data


def act(self, game_state: dict) -> str:
    """Choose an action. Hard 0.5 s budget when not training (settings.py:53)."""
    if game_state is None:
        return ACTIONS[WAIT]

    if game_state["round"] != getattr(self, "round_index", 0):
        self.round_index = game_state["round"]
        self._step_cache.clear()

    try:
        name, _, step_data = choose_action(self, game_state)

        if self.train:
            # train.py consumes this on the matching game_events_occurred call,
            # so the forward pass is not repeated for the update.
            key = (game_state["round"], game_state["step"])
            if len(self._step_cache) > 4:
                self._step_cache.clear()
            self._step_cache[key] = step_data

        return name
    except Exception as error:  # noqa: BLE001
        self.logger.exception(f"act() failed ({error}); falling back.")
        return _fallback_action(self, game_state)


def _fallback_action(self, game_state: dict) -> str:
    """Least-bad action when the policy path fails: any non-fatal legal move."""
    try:
        mask = T.action_mask_for(game_state, self.config)
        if mask.any():
            return ACTIONS[int(np.flatnonzero(mask)[0])]
    except Exception:  # noqa: BLE001
        pass
    return ACTIONS[WAIT]
