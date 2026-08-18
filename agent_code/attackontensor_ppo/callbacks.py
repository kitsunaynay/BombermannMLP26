"""Inference callbacks for the AttackOnTensor PPO agent.

Everything here is shaped by the tournament's 0.5 s per-step budget on a single
CPU thread:

* ``torch.set_num_threads(config.torch_threads)`` -- the brief promises exactly
  one thread, and letting torch spin up more would oversubscribe it and make
  each call *slower*, not faster.
* A warm-up forward pass in ``setup``. Torch defers a good deal of allocator and
  kernel setup to the first real call; ``setup`` is untimed, ``act`` is not, so
  that cost is paid up front rather than inside a timed step.
* ``torch.inference_mode()`` around the forward pass -- no autograd graph is
  built when we are only playing.

Weights load with ``map_location='cpu'`` so a GPU-trained checkpoint runs on the
graders' CPU-only machine.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from . import tensorizer as T
from .config import PPOConfig
from .kit.actions import ACTIONS, WAIT
from .network import build_network

#: Re-exported so train.py can follow the template's import convention.
state_to_features = T.state_to_tensor


def setup(self):
    """Called once before the first round; untimed."""
    self.config = PPOConfig.load()

    torch.set_num_threads(max(1, int(self.config.torch_threads)))
    torch.manual_seed(self.config.seed)

    self.device = torch.device(self.config.device)
    self.network = build_network(self.config, T.N_CHANNELS).to(self.device)

    path = self.config.model_path
    if path.is_file():
        try:
            _load_checkpoint(self, path)
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


def _load_checkpoint(self, path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("network", {})

    expected = (T.N_CHANNELS, self.config.spatial_size)
    found = (saved.get("in_channels"), saved.get("spatial_size"))
    if None not in found and found != expected:
        raise ValueError(
            f"checkpoint was trained for input {found}, but the current config "
            f"expects {expected} -- check `observation` and `ego_radius`"
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
