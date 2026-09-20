from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import features as F
from .config import QLConfig
from .kit import pathfind as P
from .kit import safety, symmetry
from .kit.actions import ACTIONS, N_ACTIONS, WAIT
from .qtable import QTable

state_to_features = F.state_to_features


def setup(self):
    self.config = QLConfig.load()
    seed = self.config.rng_seed
    self.rng = np.random.default_rng(seed)

    path = self.config.model_path
    if self.train:
        # resume from disk or start fresh (tools/train_ql.py deletes for cold start)
        self.q = _load_or_create(self, path)
    elif path.is_file():
        self.q = QTable.load(path, seed=seed)
        _adopt_checkpoint_representation(self)
        self.logger.info(f"Loaded Q-table from {path} ({self.q.n_states} states)")
    else:
        # missing checkpoint must not crash; play with empty table
        self.logger.error(f"No Q-table at {path}; playing from an empty table.")
        self.q = QTable(
            optimistic_init=self.config.optimistic_init,
            double=self.config.double_q,
            seed=seed,
        )

    self.round_index = 0
    self._view_cache: dict = {}

    self.logger.info(
        f"AttackOnTensor QL ready | variant={self.config.variant} "
        f"safety={self.config.safety_mode} symmetry={self.config.use_symmetry} "
        f"double_q={self.config.double_q} states={self.q.n_states}"
    )


def _adopt_checkpoint_representation(self) -> None:
    # checkpoint's variant/symmetry/safety shape the key; wrong shape misses every lookup
    # safety_mode override needed: hard table never learned no-escape bomb states
    metadata = getattr(self.q, "metadata", {}) or {}

    for field in ("variant", "use_symmetry", "safety_mode", "opponent_bomb_lookahead"):
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
        self.logger.warning(
            f"{total - usable} of {total} Q-table entries have a key shape this "
            f"variant cannot look up (shapes {dict(lengths)}); they are dead "
            "weight from a run resumed under a different variant."
        )


def _load_or_create(self, path) -> QTable:
    # resume training or start fresh
    seed = self.config.rng_seed
    if not path.is_file():
        self.logger.info("Starting from a fresh Q-table.")
        return QTable(
            optimistic_init=self.config.optimistic_init,
            double=self.config.double_q,
            seed=seed,
        )
    try:
        table = QTable.load(path, seed=seed)
        self.logger.info(f"Resuming training from {path} ({table.n_states} states)")
        # curriculum stages may switch variant/symmetry; warn but accept caller's choice
        metadata = table.metadata or {}

        for field, live in (("variant", self.config.variant),
                            ("use_symmetry", self.config.use_symmetry)):
            stored = metadata.get(field)
            if stored is not None and stored != live:
                self.logger.warning(
                    f"Resuming a table trained with {field}={stored!r} under "
                    f"{live!r}; previously learned states will not be reused."
                )

        # double_q is fixed once set; can't merge/split tables mid-run
        if table.double != self.config.double_q:
            self.logger.warning(
                f"Checkpoint was trained with double_q={table.double}; keeping that "
                f"instead of the configured {self.config.double_q}. Train from "
                "scratch to change it."
            )
        return table
    except Exception as error:  # noqa: BLE001
        self.logger.warning(f"Could not load {path} ({error}); starting fresh.")
        return QTable(
            optimistic_init=self.config.optimistic_init,
            double=self.config.double_q,
            seed=seed,
        )


def cached_view(self, game_state: dict) -> Optional[F.FeatureView]:
    # cache feature extraction since act() and game_events_occurred() see same state twice
    if game_state is None:
        return None

    key = (game_state["round"], game_state["step"])
    cached = self._view_cache.get(key)
    if cached is None:
        cached = F.extract(game_state, self.config)
        if len(self._view_cache) > 4:
            self._view_cache.clear()
        self._view_cache[key] = cached
    return cached


def compute_mask(self, game_state: dict) -> np.ndarray:
    field, position, bomb_available, _, _, _, danger, passable = P.game_state_context(
        game_state
    )
    threats = ()
    if self.config.opponent_bomb_lookahead and self.config.safety_mode == "hard":
        threats = P.armed_opponents(game_state)
    return safety.action_mask(
        field,
        position,
        danger,
        passable,
        bomb_available,
        mode=self.config.safety_mode,
        threats=threats,
    )


def choose_action(self, game_state: dict, epsilon: float) -> Tuple[str, int, F.FeatureView]:
    # lookup in canonical frame + transform action back to board frame
    view = cached_view(self, game_state)
    mask = compute_mask(self, game_state)

    canonical_mask = symmetry.transform_action_array(mask, view.transform)
    canonical_action = self.q.epsilon_greedy(view.key, epsilon, canonical_mask)
    action = symmetry.inverse_transform_action(canonical_action, view.transform)

    return ACTIONS[action], action, view


def act(self, game_state: dict) -> str:
    # select action with bounded cost and crash recovery
    if game_state is None:
        return ACTIONS[WAIT]

    if game_state["round"] != getattr(self, "round_index", 0):
        self.round_index = game_state["round"]
        self._view_cache.clear()

    epsilon = (self.config.epsilon(self.round_index) if self.train
               else self.config.eval_epsilon)

    try:
        name, _, _ = choose_action(self, game_state, epsilon)
        return name
    except Exception as error:  # noqa: BLE001
        self.logger.exception(f"act() failed ({error}); falling back.")
        return _fallback_action(self, game_state)


def _fallback_action(self, game_state: dict) -> str:
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
