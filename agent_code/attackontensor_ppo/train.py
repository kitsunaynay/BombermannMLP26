"""Training callbacks: single-process PPO inside the framework.

This is the spec-compliant training path -- the three callbacks the brief
requires, driven by ``python main.py play --train 1``. It is correct but slow,
because the framework's game loop is sequential Python. Serious runs use
``tools/train_ppo.py``, which drives many copies of ``blib/fast_env.py`` in
parallel and writes the same checkpoint format; the brief explicitly permits
that ("multiprocessing or anything else that comes to your mind to improve
training is perfectly fine") as long as the final agent is single-process.

**The delivery quirk this file handles**, same as the Q-learning agent. On the
last step of a round, ``do_step`` calls ``send_game_events`` and then
``end_round`` (environment.py:174, 177). A *surviving* agent therefore receives
the final transition twice -- once via ``game_events_occurred`` and again via
``end_of_round``, with the same state and action, and with ``SURVIVED_ROUND``
appended to the same list object in between. An agent that *died* receives it
only once, through ``end_of_round``, because ``send_game_events`` skips the dead
(environment.py:469). Appending in both places would double-count every
surviving episode's final transition, so we track the last step processed.

Nothing here may import from the repository root beyond the framework's own
modules: the graders copy this directory alone into their tree.
"""

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

#: Entropy below this (nats) means the policy has gone deterministic too early.
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
    # Gradient steps taken during this round. An update only runs when the
    # rollout buffer fills, which is rarer than once per round, so plots must
    # filter on this rather than treating every row as an update sample.
    "updates",
)


def setup_training(self):
    """Called once after ``setup``; ``self`` is shared with callbacks.py."""
    self.network.train()

    self.buffer = RolloutBuffer(
        capacity=self.config.rollout_steps,
        observation_shape=T.observation_shape(self.config),
    )
    self.learner = PPOLearner(self.network, self.config, device=self.config.device)

    # Dedup bookkeeping for the double delivery described in the module docstring.
    self._last_processed_step: Optional[int] = None
    self._last_events_len = 0

    self._last_update = {}
    self.round_index = 0
    _reset_round_stats(self)

    self._metrics_path = _resolve_metrics_path(self.config.metrics_file)
    if self._metrics_path is not None:
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._metrics_path.exists():
            with self._metrics_path.open("w", newline="") as stream:
                csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writeheader()

    # The framework has no "training finished" hook, so without this a run whose
    # round count is not a multiple of checkpoint_every discards its last work.
    atexit.register(_save_on_exit, self)

    self.logger.info(
        f"PPO training start | rollout={self.config.rollout_steps} "
        f"epochs={self.config.update_epochs} minibatch={self.config.minibatch_size} "
        f"lr={self.config.learning_rate} gamma={self.config.gamma}"
    )


def _save_on_exit(self) -> None:
    try:
        self.learner.save(self.config.model_path)
        self.logger.info(f"Final save -> {self.config.model_path}")
    except Exception as error:  # noqa: BLE001 - never raise at interpreter shutdown
        self.logger.error(f"Final save failed: {error}")


def _resolve_metrics_path(raw: str) -> Optional[Path]:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else AGENT_DIR / path


def _reset_round_stats(self) -> None:
    self._stats = {k: 0 for k in ("steps", "coins", "crates", "kills", "suicides", "invalid")}
    self._stats["survived"] = 0
    self._total_reward = 0.0


def _tally(self, events) -> None:
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
    """Rollout data for a state, reusing what ``act`` already computed.

    ``act`` stashes the observation, log-probability and value for each step, so
    the common path costs no extra forward pass. The fallback recomputes, which
    only happens if ``act`` was skipped -- possible in principle when the
    framework drops a slow agent's turn (environment.py:457), though training
    runs with an infinite timeout.
    """
    key = (game_state["round"], game_state["step"])
    cached = self._step_cache.get(key)
    if cached is not None:
        return cached

    _, _, step_data = choose_action(self, game_state)
    if self_action is not None and self_action in ACTION_INDEX:
        # Honour the action the world actually executed, not a fresh sample.
        step_data["action"] = ACTION_INDEX[self_action]
    return step_data


def _store(self, game_state, self_action, reward, done) -> None:
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
    """``V(s_T)`` for a truncated rollout; zero when the episode really ended."""
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
    """Run a PPO update once the rollout buffer is full."""
    if not self.buffer.is_full:
        return

    last_value = _bootstrap_value(self, next_game_state)
    advantages, returns = self.buffer.compute_advantages(
        last_value, self.config.gamma, self.config.gae_lambda
    )

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
    """The plan's PPO failure detectors, logged loudly rather than silently."""
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
    """Called once per step, except for the step an agent dies on."""
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
    """Called once per round per training agent, dead or alive."""
    if last_game_state is None:
        _finish_round(self, self.round_index + 1)
        return

    round_number = last_game_state["round"]
    already_seen = self._last_processed_step == last_game_state["step"]

    if already_seen:
        # Survivor path: the transition is already in the buffer. Credit only
        # the events appended since (SURVIVED_ROUND and friends) and close the
        # episode so GAE stops bootstrapping across the boundary.
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
        # Death path: game_events_occurred was skipped for this step, so this is
        # the only chance to learn from the step that killed us.
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
    self.round_index = round_number
    self._last_processed_step = None
    self._last_events_len = 0
    self._step_cache.clear()

    _write_metrics(self)

    if self.config.checkpoint_every > 0 and round_number % self.config.checkpoint_every == 0:
        self.learner.save(self.config.model_path)
        self.logger.info(f"Checkpoint at round {round_number} -> {self.config.model_path}")

    _reset_round_stats(self)


def _write_metrics(self) -> None:
    if self._metrics_path is None:
        return

    # Coins and kills are the only scoring events, so this reproduces the
    # framework's score exactly.
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

    # Consume the update stats so the next round does not re-report them as if
    # a fresh update had happened.
    self._last_update = {}
