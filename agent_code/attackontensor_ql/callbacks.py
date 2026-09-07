"""Inference callbacks for the AttackOnTensor Q-learning agent.

The framework requires exactly two functions here with exactly these
signatures; ``AgentRunner`` checks the arity at load time and refuses to start
otherwise (agents.py:209-217).

Acting happens in the *canonical* symmetry frame: features are reduced to a
representative of their D4 orbit, the Q-table is queried there, and the chosen
action is mapped back onto the real board with the inverse transform. Getting
that round-trip wrong makes the agent play a mirrored policy, so
:func:`tests.test_ql_agent` pins it down.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import features as F
from .config import QLConfig
from .kit import pathfind as P
from .kit import safety, symmetry
from .kit.actions import ACTIONS, N_ACTIONS, WAIT
from .qtable import QTable

#: Re-exported so ``train.py`` can follow the template's import convention.
state_to_features = F.state_to_features


def setup(self):
    """Called once before the first round; no time limit applies here."""
    self.config = QLConfig.load()
    self.rng = np.random.default_rng()

    path = self.config.model_path
    if self.train:
        # Training always starts from whatever is on disk if it is loadable, so
        # curriculum stages can chain; tools/train_ql.py deletes the file to
        # force a cold start.
        self.q = _load_or_create(self, path)
    elif path.is_file():
        self.q = QTable.load(path)
        _adopt_checkpoint_representation(self)
        self.logger.info(f"Loaded Q-table from {path} ({self.q.n_states} states)")
    else:
        # Never crash for a missing checkpoint: an untrained agent that plays
        # badly still completes the graders' submission test, a crashing one
        # does not.
        self.logger.warning(f"No Q-table at {path}; playing from an empty table.")
        self.q = QTable(
            optimistic_init=self.config.optimistic_init,
            double=self.config.double_q,
        )

    self.round_index = 0
    self._view_cache: dict = {}

    self.logger.info(
        f"AttackOnTensor QL ready | variant={self.config.variant} "
        f"safety={self.config.safety_mode} symmetry={self.config.use_symmetry} "
        f"double_q={self.config.double_q} states={self.q.n_states}"
    )


def _adopt_checkpoint_representation(self) -> None:
    """Make the live config match the representation the table was built with.

    The feature variant and the symmetry flag both change what a key *is*. A
    table trained under ``compact`` produces 9-element keys; querying it with
    ``full`` produces 12-element keys, which match nothing, so every lookup
    returns a fresh row and the agent plays untrained without erroring. The
    tournament sets no environment variables, so the config would otherwise fall
    back to defaults and hit exactly that. The checkpoint is the authority.
    """
    metadata = getattr(self.q, "metadata", {}) or {}

    for field in ("variant", "use_symmetry"):
        stored = metadata.get(field)
        if stored is None:
            continue
        if stored != getattr(self.config, field):
            self.logger.warning(
                f"Checkpoint was trained with {field}={stored!r}, config says "
                f"{getattr(self.config, field)!r}; using the checkpoint's value."
            )
            setattr(self.config, field, stored)

    expected = len(self.config.feature_names)
    lengths = self.q.key_lengths()
    if not lengths:
        return

    usable = lengths.get(expected, 0)
    total = sum(lengths.values())

    if usable == 0:
        self.logger.error(
            f"No Q-table key has {expected} features (variant "
            f"'{self.config.variant}'); stored key shapes are {dict(lengths)}. "
            "Every lookup will miss; retrain or fix the variant."
        )
    elif usable < total:
        # Resuming under a different variant leaves the old entries behind.
        # They are unreachable rather than harmful, so this is a warning.
        self.logger.warning(
            f"{total - usable} of {total} Q-table entries have a key shape this "
            f"variant cannot look up (shapes {dict(lengths)}); they are dead "
            "weight from a run resumed under a different variant."
        )


def _load_or_create(self, path) -> QTable:
    if not path.is_file():
        self.logger.info("Starting from a fresh Q-table.")
        return QTable(
            optimistic_init=self.config.optimistic_init,
            double=self.config.double_q,
        )
    try:
        table = QTable.load(path)
        self.logger.info(f"Resuming training from {path} ({table.n_states} states)")
        # A curriculum stage may change the variant on purpose (compact for
        # Tasks 1-2, full for 3-4), so warn rather than adopt: when resuming, the
        # caller's choice wins but the key mismatch is still worth flagging.
        stored = (table.metadata or {}).get("variant")
        if stored is not None and stored != self.config.variant:
            self.logger.warning(
                f"Resuming a table trained with variant={stored!r} under "
                f"{self.config.variant!r}; previously learned states will not be reused."
            )
        return table
    except Exception as error:  # noqa: BLE001 - a stale checkpoint must not abort training
        self.logger.warning(f"Could not load {path} ({error}); starting fresh.")
        return QTable(
            optimistic_init=self.config.optimistic_init,
            double=self.config.double_q,
        )


def cached_view(self, game_state: dict) -> Optional[F.FeatureView]:
    """Feature view for a state, memoised within the step.

    ``act`` and then ``game_events_occurred`` are both handed the very same
    state dictionary each step. Extracting features involves a BFS and a
    survival search, so recomputing them would roughly double the agent's cost
    for nothing.
    """
    if game_state is None:
        return None

    key = (game_state["round"], game_state["step"])
    cached = self._view_cache.get(key)
    if cached is None:
        cached = F.extract(game_state, self.config)
        # Two entries is enough: the current step and the previous one.
        if len(self._view_cache) > 4:
            self._view_cache.clear()
        self._view_cache[key] = cached
    return cached


def compute_mask(self, game_state: dict) -> np.ndarray:
    """Safety mask over actions, in real-board coordinates."""
    field, position, bomb_available, _, _, _, danger, passable = P.game_state_context(
        game_state
    )
    return safety.action_mask(
        field,
        position,
        danger,
        passable,
        bomb_available,
        mode=self.config.safety_mode,
    )


def choose_action(self, game_state: dict, epsilon: float) -> Tuple[str, int, F.FeatureView]:
    """Pick an action and report it alongside the canonical view it came from."""
    view = cached_view(self, game_state)
    mask = compute_mask(self, game_state)

    # The Q-table lives in the canonical frame, so the mask must be rotated
    # into it before it can be applied.
    canonical_mask = symmetry.transform_action_array(mask, view.transform)

    canonical_action = self.q.epsilon_greedy(view.key, epsilon, canonical_mask)
    action = symmetry.inverse_transform_action(canonical_action, view.transform)

    return ACTIONS[action], action, view


def act(self, game_state: dict) -> str:
    """Choose an action. Hard 0.5 s budget when not training (settings.py:53)."""
    if game_state is None:
        return ACTIONS[WAIT]

    if game_state["round"] != getattr(self, "round_index", 0):
        self.round_index = game_state["round"]
        self._view_cache.clear()

    epsilon = self.config.epsilon(self.round_index) if self.train else 0.0

    try:
        name, _, _ = choose_action(self, game_state, epsilon)
        return name
    except Exception as error:  # noqa: BLE001
        # A raised exception costs the whole game. Degrade to a safe-ish move
        # and leave a stack trace in the agent log instead.
        self.logger.exception(f"act() failed ({error}); falling back.")
        return _fallback_action(self, game_state)


def _fallback_action(self, game_state: dict) -> str:
    """Least-bad action when the policy path fails: any non-fatal legal move."""
    try:
        field, position, bomb_available, _, _, _, danger, passable = P.game_state_context(
            game_state
        )
        mask = safety.certain_death_mask(field, position, danger, passable, bomb_available)
        if mask.any():
            return ACTIONS[int(np.flatnonzero(mask)[0])]
    except Exception:  # noqa: BLE001
        pass
    return ACTIONS[WAIT]
