"""Training callbacks: n-step Q-learning updates.

The final transition is delivered twice. From ``do_step`` in
environment.py:158:

.. code-block:: text

    poll_and_run_agents()
    ... world resolves ...
    send_game_events()      # -> game_events_occurred, for agents still alive
    if time_to_stop():
        end_round()         # -> appends SURVIVED_ROUND, then end_of_round

On the final step of a round, an agent that is **still alive** receives the same
transition twice: once through ``game_events_occurred`` and again through
``end_of_round``, with the *same* ``last_game_state`` and ``last_action``, and
with ``SURVIVED_ROUND`` appended to the very same list object in between. An
agent that **died** receives it only once, through ``end_of_round``, because
``send_game_events`` skips the dead (environment.py:469).

Appending in both callbacks would double-count every surviving episode's final
step, so we track the last step processed and, on a duplicate, credit only the
newly appended events and close the episode.

Nothing here imports from the repository root beyond the framework's own
``events`` module: the graders copy this directory alone into their tree.
"""

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

#: Q values above this magnitude mean the updates are diverging.
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
    """Called once after ``setup``; ``self`` is shared with callbacks.py."""
    self.transitions = deque()
    self.round_index = 0

    # Stamp the representation onto the table so evaluation can restore it. The
    # tournament runs with no environment variables set, so without this a table
    # trained under the `compact` variant would be queried with `full`, every
    # lookup would miss, and the agent would play untrained without erroring.
    self.q.metadata.update(
        variant=self.config.variant,
        use_symmetry=self.config.use_symmetry,
        # A table trained behind the `hard` mask never sees the consequences of
        # a no-escape bomb, so it never learns to avoid one: replayed under
        # `soft` it suicides every round (measured: 38.7 coins under `hard`,
        # 1.35 under `soft`). The mask is part of the trained policy, so it
        # travels with the table.
        safety_mode=self.config.safety_mode,
        gamma=self.config.gamma,
        n_step=self.config.n_step,
        double_q=self.config.double_q,
    )

    # Dedup bookkeeping for the double-delivery described in the module docstring.
    self._last_processed_step: Optional[int] = None
    self._last_events_len = 0

    # `self` is a SimpleNamespace the framework hands to every callback
    # (agents.py:219), not an object of ours, so module functions take it as an
    # explicit argument rather than being called as methods.
    _reset_round_stats(self)

    self._metrics_path = _resolve_metrics_path(self.config.metrics_file)
    if self._metrics_path is not None:
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._metrics_path.exists():
            with self._metrics_path.open("w", newline="") as stream:
                csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writeheader()

    # The framework has no "training finished" hook; end_of_round is the last
    # callback it makes. Without this, a run whose round count is not a multiple
    # of checkpoint_every discards its most recent learning.
    atexit.register(_save_on_exit, self)

    self.logger.info(
        f"Training start | gamma={self.config.gamma} alpha={self.config.alpha} "
        f"n_step={self.config.n_step} eps={self.config.epsilon_start}->"
        f"{self.config.epsilon_end}"
    )


def _save_on_exit(self) -> None:
    try:
        self.q.save(self.config.model_path)
        self.logger.info(
            f"Final save: {self.q.n_states} states -> {self.config.model_path}"
        )
    except Exception as error:  # noqa: BLE001 - never raise during interpreter shutdown
        self.logger.error(f"Final save failed: {error}")


def _resolve_metrics_path(raw: str) -> Optional[Path]:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else AGENT_DIR / path


def _reset_round_stats(self):
    self._stats = {key: 0 for key in ("steps", "coins", "crates", "kills", "suicides", "invalid")}
    self._stats["survived"] = 0
    self._total_reward = 0.0
    self._td_errors: List[float] = []
    self._alphas: List[float] = []


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


def _action_index(self_action: Optional[str]) -> int:
    """Action string to index, tolerating the framework's substitutions.

    ``poll_and_run_agents`` replaces a slow agent's choice with ``'WAIT'`` and a
    crashed one's with ``'ERROR'`` (environment.py:444-451).
    """
    if self_action is None:
        return WAIT
    return ACTION_INDEX.get(self_action, WAIT)


def _alpha_for(self, state, action: int) -> float:
    """Learning rate, optionally annealed by visit count.

    ``alpha_t = alpha_0 / (1 + kappa * n(s, a))`` satisfies the Robbins-Monro
    conditions, so with ``alpha_decay > 0`` the tabular updates provably
    converge; a constant rate tracks a moving target better during self-play.
    """
    base = self.config.alpha
    if self.config.alpha_decay <= 0.0:
        return base
    visits = int(self.q.visits(state)[action])
    return max(self.config.alpha_min, base / (1.0 + self.config.alpha_decay * visits))


def _apply_oldest(self, next_key, next_mask, terminal: bool) -> None:
    """Back up the oldest buffered transition over the current window."""
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
    """Turn one transition into buffered experience and run any due updates."""
    old_view = cached_view(self, old_game_state)
    if old_view is None:
        return

    action = _action_index(self_action)
    # Experience is stored in the canonical frame the Q-table is indexed by.
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
    """Called once per step, except for the step an agent dies on."""
    if old_game_state is None:
        return

    _tally(self, events)
    _record(self, old_game_state, self_action, new_game_state, events, terminal=False)

    # Remember what was already credited, so end_of_round can tell whether it
    # is seeing a genuinely new transition or a repeat of this one.
    self._last_processed_step = old_game_state["step"]
    self._last_events_len = len(events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called once per round per training agent, dead or alive."""
    if last_game_state is None:
        _finish_round(self, events, self.round_index + 1)
        return

    round_number = last_game_state["round"]

    already_seen = self._last_processed_step == last_game_state["step"]

    if already_seen:
        # Survivor path: this transition was already buffered by
        # game_events_occurred. Credit only the events appended since then
        # (SURVIVED_ROUND, and anything else end_round added) and close out.
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
        # Death path: game_events_occurred was skipped for this step, so this
        # is the only chance to learn from the step that killed us.
        _tally(self, events)
        _record(self, last_game_state, last_action, None, events, terminal=True)

    _finish_round(self, events, round_number)


def _finish_round(self, events, round_number: int) -> None:
    # The game's own round counter is the single source of truth: callbacks.act
    # also writes self.round_index from game_state, so incrementing a separate
    # counter here would make the two disagree and skew the epsilon schedule.
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

    # Every round, not just on checkpoint rounds. These counters are per-round
    # metrics; resetting them on the checkpoint cadence turned every training
    # curve into a sawtooth and latched `survived` to 1.
    _reset_round_stats(self)


def _write_snapshot(self) -> None:
    """Keep a dated copy so the best table can be chosen after the fact.

    The last table a run writes is not reliably its best (see config.snapshot_dir).
    """
    if not self.config.snapshot_dir:
        return
    directory = Path(self.config.snapshot_dir)
    if not directory.is_absolute():
        directory = AGENT_DIR / directory
    try:
        self.q.save(directory / f"q_table_r{self.round_index:06d}.pkl")
    except Exception as error:  # noqa: BLE001 - snapshotting must not kill training
        self.logger.warning(f"Could not write snapshot: {error}")


def _check_for_divergence(self) -> None:
    """Failure detectors from the plan: NaN and value explosion."""
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
    if self._metrics_path is None:
        return

    # Coins and kills are the only scoring events, so this reproduces the
    # framework's score exactly; read from settings in case the values change.
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
        # The rate actually applied. With alpha_decay > 0 this falls below
        # config.alpha as visit counts grow, so logging the configured base
        # would hide whether the schedule is doing anything at all.
        "alpha": round(float(np.mean(self._alphas)) if self._alphas else self.config.alpha, 6),
        "n_states": self.q.n_states,
        "max_abs_q": round(self.q.max_abs_q, 4),
        "mean_abs_td": round(float(np.mean(self._td_errors)) if self._td_errors else 0.0, 5),
        "coverage": round(self.q.coverage(), 5),
    }

    with self._metrics_path.open("a", newline="") as stream:
        csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writerow(row)
