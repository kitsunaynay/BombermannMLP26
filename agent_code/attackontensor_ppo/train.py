from __future__ import annotations

import atexit
import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

import events as e
import settings as s

from . import rewards as R
from . import tensorizer as T
from .callbacks import choose_action
from .config import AGENT_DIR
from .kit.actions import ACTION_INDEX, WAIT
from .ppo import PPOLearner, RolloutBuffer

ENTROPY_COLLAPSE_THRESHOLD = 0.2

METRIC_FIELDS = (
    "round",
    "steps",
    "score",
    "coins",
    "crates",
    "kills",
    "suicides",
    "invalid",
    "survived",
    "total_reward",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "grad_norm",
    "learning_rate",
    "updates",
)


def setup_training(self):
    # setup PPO
    self.network.train()

    self.buffer = RolloutBuffer(
        capacity=self.config.rollout_steps,
        observation_shape=T.observation_shape(self.config),
    )
    self.learner = PPOLearner(self.network, self.config, device=self.config.device)

    # step tracking
    self._last_processed_step: Optional[int] = None
    self._last_events_len = 0

    self._last_update = {}
    self.round_index = 0
    _reset_round_stats(self)

    # metrics
    self._metrics_path = _resolve_metrics_path(self.config.metrics_file)
    if self._metrics_path is not None:
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._metrics_path.exists():
            with self._metrics_path.open("w", newline="") as stream:
                csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writeheader()

    # save on exit
    atexit.register(_save_on_exit, self)

    self.logger.info(
        f"PPO training start | rollout={self.config.rollout_steps} "
        f"epochs={self.config.update_epochs} minibatch={self.config.minibatch_size} "
        f"lr={self.config.learning_rate} gamma={self.config.gamma}"
    )


def _save_on_exit(self) -> None:
    # final save
    try:
        self.learner.save(self.config.model_path)
        self.logger.info(f"Final save -> {self.config.model_path}")
    except Exception as error:  # noqa: BLE001 - never raise at interpreter shutdown
        self.logger.error(f"Final save failed: {error}")


def _resolve_metrics_path(raw: str) -> Optional[Path]:
    # normalize path
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else AGENT_DIR / path


def _reset_round_stats(self) -> None:
    # reset stats
    self._stats = {k: 0 for k in ("steps", "coins", "crates", "kills", "suicides", "invalid")}
    self._stats["survived"] = 0
    self._total_reward = 0.0


def _tally(self, events) -> None:
    # count events
    self._stats["steps"] += 1
    for event in events:
        if event == e.COIN_COLLECTED:
            self._stats["coins"] += 1
        elif event == e.CRATE_DESTROYED:
            self._stats["crates"] += 1
        elif event == e.KILLED_OPPONENT:
            self._stats["kills"] += 1
        elif event == e.KILLED_SELF:
            self._stats["suicides"] += 1
        elif event == e.INVALID_ACTION:
            self._stats["invalid"] += 1
        elif event == e.SURVIVED_ROUND:
            self._stats["survived"] = 1


def _step_data_for(self, game_state: dict, self_action: Optional[str]) -> dict:
    # cache step data
    key = (game_state["round"], game_state["step"])
    cached = self._step_cache.get(key)
    if cached is not None:
        return cached

    _, _, step_data = choose_action(self, game_state)
    if self_action is not None and self_action in ACTION_INDEX:
        step_data["action"] = ACTION_INDEX[self_action]
    return step_data


def _store(self, game_state, self_action, reward, done) -> None:
    # add transition
    step_data = _step_data_for(self, game_state, self_action)
    self.buffer.add(
        observation=step_data["observation"],
        action=step_data["action"],
        log_prob=step_data["log_prob"],
        reward=reward,
        value=step_data["value"],
        done=done,
        mask=step_data["mask"],
    )


def _bootstrap_value(self, game_state: Optional[dict]) -> float:
    # critic bootstrap
    if game_state is None:
        return 0.0
    observation = T.state_to_tensor(game_state, self.config)
    mask = T.action_mask_for(game_state, self.config)
    with torch.inference_mode():
        _, value = self.network(
            torch.as_tensor(observation, device=self.device).unsqueeze(0),
            torch.as_tensor(mask, device=self.device).unsqueeze(0),
        )
    return float(value.item())


def _maybe_update(self, next_game_state: Optional[dict]) -> None:
    # wait for full rollout
    if not self.buffer.is_full:
        return

    # GAE + returns
    last_value = _bootstrap_value(self, next_game_state)
    advantages, returns = self.buffer.compute_advantages(
        last_value, self.config.gamma, self.config.gae_lambda
    )

     # PPO update
    self.network.train()
    metrics = self.learner.update(self.buffer, advantages, returns)
    self.network.eval()

    self.buffer.reset()
    self._last_update = metrics.as_dict()

    _check_for_failure(self, metrics)
    self.logger.info(
        f"PPO update | policy={metrics.policy_loss:+.4f} value={metrics.value_loss:.4f} "
        f"entropy={metrics.entropy:.3f} kl={metrics.approx_kl:.4f} "
        f"clip={metrics.clip_fraction:.3f} ev={metrics.explained_variance:+.3f}"
    )


def _check_for_failure(self, metrics) -> None:
    # sanity checks
    if metrics.entropy < ENTROPY_COLLAPSE_THRESHOLD:
        self.logger.error(
            f"Policy entropy {metrics.entropy:.3f} < {ENTROPY_COLLAPSE_THRESHOLD}: "
            "the policy has gone deterministic; raise entropy_coefficient."
        )
    if metrics.approx_kl > 2 * self.config.target_kl:
        self.logger.error(
            f"approx_kl {metrics.approx_kl:.4f} far above target "
            f"{self.config.target_kl}: lower the learning rate."
        )
    if metrics.explained_variance < 0.0:
        self.logger.warning(
            f"explained_variance {metrics.explained_variance:+.3f} < 0: the critic "
            "predicts worse than the mean return."
        )


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    # normal step
    if old_game_state is None:
        return

    _tally(self, events)

    reward, _ = R.compute_reward(
        old_game_state, self_action, new_game_state, events, self.config
    )
    self._total_reward += reward

    _store(self, old_game_state, self_action, reward, done=False)
    _maybe_update(self, new_game_state)

    self._last_processed_step = old_game_state["step"]
    self._last_events_len = len(events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # finish round
    if last_game_state is None:
        _finish_round(self, self.round_index + 1)
        return

    round_number = last_game_state["round"]
    already_seen = self._last_processed_step == last_game_state["step"]

    if already_seen:
        # add only new events
        new_events = list(events[self._last_events_len :])
        if new_events and len(self.buffer) > 0:
            bonus = R.reward_from_events(new_events, self.config)
            self.buffer.rewards[len(self.buffer) - 1] += bonus
            self._total_reward += bonus
        for event in new_events:
            if event == e.SURVIVED_ROUND:
                self._stats["survived"] = 1
        if len(self.buffer) > 0:
            self.buffer.dones[len(self.buffer) - 1] = True
    else:
        # final transition
        _tally(self, events)
        reward, _ = R.compute_reward(
            last_game_state, last_action, None, events, self.config
        )
        self._total_reward += reward
        if not self.buffer.is_full:
            _store(self, last_game_state, last_action, reward, done=True)

    _maybe_update(self, None)
    _finish_round(self, round_number)


def _finish_round(self, round_number: int) -> None:
    # cleanup + metrics
    self.round_index = round_number
    self._last_processed_step = None
    self._last_events_len = 0
    self._step_cache.clear()

    _write_metrics(self)

    # checkpoint
    if self.config.checkpoint_every > 0 and round_number % self.config.checkpoint_every == 0:
        self.learner.save(self.config.model_path)
        self.logger.info(f"Checkpoint at round {round_number} -> {self.config.model_path}")

    _reset_round_stats(self)


def _write_metrics(self) -> None:
    # CSV
    if self._metrics_path is None:
        return

    score = self._stats["coins"] * s.REWARD_COIN + self._stats["kills"] * s.REWARD_KILL
    update = self._last_update

    row = {
        "round": self.round_index,
        "steps": self._stats["steps"],
        "score": score,
        "coins": self._stats["coins"],
        "crates": self._stats["crates"],
        "kills": self._stats["kills"],
        "suicides": self._stats["suicides"],
        "invalid": self._stats["invalid"],
        "survived": self._stats["survived"],
        "total_reward": round(self._total_reward, 4),
        "policy_loss": round(update.get("policy_loss", 0.0), 6),
        "value_loss": round(update.get("value_loss", 0.0), 6),
        "entropy": round(update.get("entropy", 0.0), 6),
        "approx_kl": round(update.get("approx_kl", 0.0), 6),
        "clip_fraction": round(update.get("clip_fraction", 0.0), 6),
        "explained_variance": round(update.get("explained_variance", 0.0), 6),
        "grad_norm": round(update.get("grad_norm", 0.0), 6),
        "learning_rate": self.learner.learning_rate,
        "updates": int(update.get("n_updates", 0)),
    }

    with self._metrics_path.open("a", newline="") as stream:
        csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writerow(row)

    self._last_update = {}
