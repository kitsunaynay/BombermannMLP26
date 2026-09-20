from __future__ import annotations

import atexit
import csv
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np

import events as e
import settings as s

from . import rewards as R
from .callbacks import cached_view, compute_mask
from .config import AGENT_DIR
from .kit import symmetry
from .kit.actions import ACTION_INDEX, ACTIONS, WAIT
from .qtable import QTable

# final step delivered twice: game_events_occurred then end_of_round
# track last processed step to avoid double-counting reward on replay
Q_DIVERGENCE_THRESHOLD = 1e3

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
    "epsilon",
    "alpha",
    "n_states",
    "max_abs_q",
    "mean_abs_td",
    "coverage",
)


def setup_training(self):
    # n-step transition buffer
    self.transitions = deque()
    self.round_index = 0

    # stamp table with exact training setup so no silent mismatches at eval
    self.q.metadata.update(
        variant=self.config.variant,
        use_symmetry=self.config.use_symmetry,
        safety_mode=self.config.safety_mode,
        opponent_bomb_lookahead=self.config.opponent_bomb_lookahead,
        gamma=self.config.gamma,
        n_step=self.config.n_step,
        double_q=self.config.double_q,
    )

    self._last_processed_step: Optional[int] = None
    self._last_events_len = 0

    _reset_round_stats(self)

    self._metrics_path = _resolve_metrics_path(self.config.metrics_file)
    if self._metrics_path is not None:
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._metrics_path.exists():
            with self._metrics_path.open("w", newline="") as stream:
                csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writeheader()

    # framework has no training finished hook; register final save
    atexit.register(_save_on_exit, self)

    self.logger.info(
        f"Training start | gamma={self.config.gamma} alpha={self.config.alpha} "
        f"n_step={self.config.n_step} eps={self.config.epsilon_start}->"
        f"{self.config.epsilon_end}"
    )


def _save_on_exit(self) -> None:
    # save q-table on shutdown
    try:
        self.q.save(self.config.model_path)
        self.logger.info(
            f"Final save: {self.q.n_states} states -> {self.config.model_path}"
        )
    except Exception as error:  # noqa: BLE001
        self.logger.error(f"Final save failed: {error}")


def _resolve_metrics_path(raw: str) -> Optional[Path]:
    # normalize path
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else AGENT_DIR / path


def _reset_round_stats(self):
    # reset episode stats and metrics accumulators
    self._stats = {key: 0 for key in ("steps", "coins", "crates", "kills", "suicides", "invalid")}
    self._stats["survived"] = 0
    self._total_reward = 0.0
    self._td_errors: List[float] = []
    self._alphas: List[float] = []


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


def _action_index(self_action: Optional[str]) -> int:
    # framework can replace slow/crashed actions with WAIT or ERROR; fall back to WAIT
    if self_action is None:
        return WAIT
    return ACTION_INDEX.get(self_action, WAIT)


def _alpha_for(self, state, action: int) -> float:
    # alpha_decay shrinks rate as visits grow; required for convergence
    base = self.config.alpha
    if self.config.alpha_decay <= 0.0:
        return base
    visits = int(self.q.visits(state)[action])
    return max(self.config.alpha_min, base / (1.0 + self.config.alpha_decay * visits))


def _apply_oldest(self, next_key, next_mask, terminal: bool) -> None:
    # n-step td update from oldest buffered transition
    gamma = self.config.gamma

    partial_return = 0.0
    for offset, (_, _, reward) in enumerate(self.transitions):
        partial_return += (gamma**offset) * reward

    state, action, _ = self.transitions[0]
    discount = 0.0 if terminal else gamma ** len(self.transitions)

    alpha = _alpha_for(self, state, action)
    td_error = self.q.learn(
        state, action, partial_return, next_key, discount, alpha, next_mask
    )
    self._td_errors.append(abs(td_error))
    self._alphas.append(alpha)
    self.transitions.popleft()


def _record(self, old_game_state, self_action, new_game_state, events, terminal: bool) -> None:
    # buffer transition for n-step learning
    old_view = cached_view(self, old_game_state)
    if old_view is None:
        return

    action = _action_index(self_action)
    # store in canonical frame
    canonical_action = symmetry.transform_action(action, old_view.transform)

    reward, all_events = R.compute_reward(
        old_game_state, self_action, new_game_state, events, self.config
    )
    self._total_reward += reward
    self.transitions.append((old_view.key, canonical_action, reward))

    next_key = None
    next_mask = None
    if not terminal and new_game_state is not None:
        new_view = cached_view(self, new_game_state)
        if new_view is not None:
            next_key = new_view.key
            next_mask = symmetry.transform_action_array(
                compute_mask(self, new_game_state), new_view.transform
            )

    if terminal:
        while self.transitions:
            _apply_oldest(self, None, None, terminal=True)
    elif len(self.transitions) >= self.config.n_step:
        _apply_oldest(self, next_key, next_mask, terminal=False)

    self.logger.debug(
        f"step {old_game_state['step']} action={ACTIONS[action]} "
        f"reward={reward:+.3f} events={','.join(all_events)}"
    )


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    # normal step: tally events and buffer transition
    if old_game_state is None:
        return

    _tally(self, events)
    _record(self, old_game_state, self_action, new_game_state, events, terminal=False)

    self._last_processed_step = old_game_state["step"]
    self._last_events_len = len(events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # finish round: handle survival bonus or death transition
    if last_game_state is None:
        _finish_round(self, events, self.round_index + 1)
        return

    round_number = last_game_state["round"]
    already_seen = self._last_processed_step == last_game_state["step"]

    if already_seen:
        # agent survived: credit only new events (SURVIVED_ROUND), flush buffer
        new_events = list(events[self._last_events_len :])
        if new_events and self.transitions:
            bonus = R.reward_from_events(new_events, self.config)
            state, action, reward = self.transitions[-1]
            self.transitions[-1] = (state, action, reward + bonus)
            self._total_reward += bonus
        for event in events[self._last_events_len :]:
            if event == e.SURVIVED_ROUND:
                self._stats["survived"] = 1

        while self.transitions:
            _apply_oldest(self, None, None, terminal=True)
    else:
        # agent died: game_events_occurred was skipped; learn from killing step
        _tally(self, events)
        _record(self, last_game_state, last_action, None, events, terminal=True)

    _finish_round(self, events, round_number)


def _finish_round(self, events, round_number: int) -> None:
    # use game's round counter to avoid skewing epsilon schedule
    self.round_index = round_number
    self.transitions.clear()
    self._last_processed_step = None
    self._last_events_len = 0
    self._view_cache.clear()

    _check_for_divergence(self)
    _write_metrics(self, last_events=events)

    if self.config.checkpoint_every > 0 and self.round_index % self.config.checkpoint_every == 0:
        self.q.save(self.config.model_path)
        self.logger.info(
            f"Checkpoint at round {self.round_index}: {self.q.n_states} states "
            f"-> {self.config.model_path}"
        )
        _write_snapshot(self)

    _reset_round_stats(self)


def _write_snapshot(self) -> None:
    # keep dated snapshots for later selection; last write isn't reliably best
    if not self.config.snapshot_dir:
        return
    directory = Path(self.config.snapshot_dir)
    if not directory.is_absolute():
        directory = AGENT_DIR / directory
    try:
        self.q.save(directory / f"q_table_r{self.round_index:06d}.pkl")
    except Exception as error:  # noqa: BLE001
        self.logger.warning(f"Could not write snapshot: {error}")


def _check_for_divergence(self) -> None:
    # warn on non-finite values or excessive q magnitudes
    if not self.q.is_finite():
        self.logger.error("Q-table contains non-finite values; learning has diverged.")
        return
    magnitude = self.q.max_abs_q
    if magnitude > Q_DIVERGENCE_THRESHOLD:
        self.logger.error(
            f"max|Q| = {magnitude:.1f} exceeds {Q_DIVERGENCE_THRESHOLD}: "
            "lower alpha or check the reward scale."
        )


def _write_metrics(self, last_events) -> None:
    # write round summary to csv
    if self._metrics_path is None:
        return

    score = self._stats["coins"] * s.REWARD_COIN + self._stats["kills"] * s.REWARD_KILL

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
        "epsilon": round(self.config.epsilon(self.round_index), 5),
        # actual alpha applied, not config.alpha; drops with alpha_decay
        "alpha": round(float(np.mean(self._alphas)) if self._alphas else self.config.alpha, 6),
        "n_states": self.q.n_states,
        "max_abs_q": round(self.q.max_abs_q, 4),
        "mean_abs_td": round(float(np.mean(self._td_errors)) if self._td_errors else 0.0, 5),
        "coverage": round(self.q.coverage(), 5),
    }

    with self._metrics_path.open("a", newline="") as stream:
        csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writerow(row)
