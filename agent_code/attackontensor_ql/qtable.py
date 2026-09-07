"""Sparse Q-table with optional Double Q-learning.

Storage is a plain ``dict`` keyed by the feature tuple, holding a length-6
float array of action values. Sparse, not dense: the full feature variant spans
~1.8M states and only 10^4..10^5 are ever visited.

Not a ``defaultdict``, because one with a lambda factory cannot be pickled and
this table has to survive a checkpoint round-trip.

Double Q-learning (van Hasselt, 2010) is available: plain Q-learning's ``max``
biases values upward, and an overestimated ``BOMB`` value means suicide here.
Two tables are kept; each update picks one at random, takes the greedy action
from it and evaluates that action with the other.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .kit.actions import N_ACTIONS

State = Tuple[int, ...]

#: Bumped whenever the on-disk layout changes, so stale checkpoints fail loudly.
FORMAT_VERSION = 1


class QTable:
    """Action-value store for discrete states."""

    def __init__(
        self,
        n_actions: int = N_ACTIONS,
        optimistic_init: float = 0.0,
        double: bool = True,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        self.n_actions = n_actions
        self.optimistic_init = float(optimistic_init)
        self.double = bool(double)
        # Which representation the keys were built from: feature variant, and
        # whether symmetry canonicalisation was on. Both change what a key means,
        # so a table loaded under a different one misses every lookup and plays
        # untrained. Stored with the table so they cannot drift apart.
        self.metadata: Dict[str, object] = dict(metadata or {})
        self._tables: Tuple[Dict[State, np.ndarray], ...] = tuple(
            {} for _ in range(2 if double else 1)
        )
        self._visits: Dict[State, np.ndarray] = {}
        self._rng = np.random.default_rng(seed)

    # -- access -------------------------------------------------------------
    def _row(self, table: Dict[State, np.ndarray], state: State) -> np.ndarray:
        row = table.get(state)
        if row is None:
            # Optimistic init explores untried actions without relying purely
            # on epsilon-greedy noise.
            row = np.full(self.n_actions, self.optimistic_init, dtype=np.float64)
            table[state] = row
        return row

    def q(self, state: State) -> np.ndarray:
        """Action values used for acting: the mean over tables when double."""
        if not self.double:
            return self._row(self._tables[0], state)
        return 0.5 * (self._row(self._tables[0], state) + self._row(self._tables[1], state))

    def visits(self, state: State) -> np.ndarray:
        counts = self._visits.get(state)
        if counts is None:
            counts = np.zeros(self.n_actions, dtype=np.int64)
            self._visits[state] = counts
        return counts

    # -- action selection ---------------------------------------------------
    def greedy(self, state: State, mask: Optional[np.ndarray] = None) -> int:
        """Highest-valued permitted action, ties broken uniformly at random.

        Random tie-breaking matters more than it looks: a freshly initialised
        table is all zeros, and a deterministic ``argmax`` would make the agent
        walk into the same wall every step of its first episodes.
        """
        values = self.q(state)
        if mask is not None and mask.any():
            values = np.where(mask, values, -np.inf)
        best = np.flatnonzero(values == values.max())
        return int(self._rng.choice(best))

    def epsilon_greedy(
        self,
        state: State,
        epsilon: float,
        mask: Optional[np.ndarray] = None,
    ) -> int:
        if epsilon > 0.0 and self._rng.random() < epsilon:
            if mask is not None and mask.any():
                return int(self._rng.choice(np.flatnonzero(mask)))
            return int(self._rng.integers(self.n_actions))
        return self.greedy(state, mask)

    # -- learning -----------------------------------------------------------
    def learn(
        self,
        state: State,
        action: int,
        partial_return: float,
        next_state: Optional[State],
        discount: float,
        alpha: float,
        next_mask: Optional[np.ndarray] = None,
    ) -> float:
        """Apply one (possibly n-step) Q-learning update. Returns the TD error.

        ``partial_return`` is the accumulated discounted reward over the backup
        window and ``discount`` is ``gamma ** n``; pass ``0.0`` to make the
        update terminal, which is what ``end_of_round`` does.
        """
        self.visits(state)[action] += 1

        if self.double:
            index = int(self._rng.integers(2))
        else:
            index = 0
        table = self._tables[index]
        row = self._row(table, state)

        bootstrap = 0.0
        if next_state is not None and discount != 0.0:
            if self.double:
                # Select with this table, evaluate with the other: the whole
                # point of Double Q-learning.
                selector = self._row(table, next_state)
                evaluator = self._row(self._tables[1 - index], next_state)
            else:
                selector = evaluator = self._row(table, next_state)

            if next_mask is not None and next_mask.any():
                selector = np.where(next_mask, selector, -np.inf)
            bootstrap = float(evaluator[int(np.argmax(selector))])

        target = partial_return + discount * bootstrap
        td_error = target - row[action]
        row[action] += alpha * td_error
        return float(td_error)

    # -- diagnostics --------------------------------------------------------
    @property
    def n_states(self) -> int:
        """Distinct states seen. Linear growth in steps means the features are
        too fine-grained to generalise; one of the failure detectors."""
        seen = set()
        for table in self._tables:
            seen.update(table.keys())
        return len(seen)

    @property
    def max_abs_q(self) -> float:
        """Largest magnitude value; a divergence alarm."""
        best = 0.0
        for table in self._tables:
            for row in table.values():
                row_max = float(np.abs(row).max()) if row.size else 0.0
                best = max(best, row_max)
        return best

    def is_finite(self) -> bool:
        return all(np.isfinite(row).all() for table in self._tables for row in table.values())

    def coverage(self) -> float:
        """Fraction of (state, action) pairs tried at least once."""
        if not self._visits:
            return 0.0
        total = sum(int((counts > 0).sum()) for counts in self._visits.values())
        return total / (len(self._visits) * self.n_actions)

    # -- persistence --------------------------------------------------------
    def save(self, path: Path | str) -> None:
        """Atomically write the table.

        Training gets interrupted constantly (Ctrl-C, curriculum switches, node
        preemption). Writing to a temporary file in the same directory and then
        ``os.replace``-ing keeps a half-written pickle from destroying hours of
        work.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": FORMAT_VERSION,
            "n_actions": self.n_actions,
            "optimistic_init": self.optimistic_init,
            "double": self.double,
            "metadata": self.metadata,
            "tables": [dict(table) for table in self._tables],
            "visits": dict(self._visits),
        }

        handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path | str, seed: Optional[int] = None) -> "QTable":
        with open(path, "rb") as stream:
            payload = pickle.load(stream)

        version = payload.get("version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Q-table at {path} has format version {version!r}, expected {FORMAT_VERSION}. "
                "Retrain or migrate the checkpoint."
            )

        table = cls(
            n_actions=payload["n_actions"],
            optimistic_init=payload["optimistic_init"],
            double=payload["double"],
            seed=seed,
            metadata=payload.get("metadata"),
        )
        table._tables = tuple(payload["tables"])
        table._visits = payload["visits"]
        return table

    def key_length(self) -> Optional[int]:
        """Most common feature-tuple length, or ``None`` for an empty table."""
        lengths = self.key_lengths()
        if not lengths:
            return None
        return max(lengths, key=lambda length: lengths[length])

    def key_lengths(self) -> Dict[int, int]:
        """How many stored keys have each length.

        A table can hold more than one key shape: resuming under a different
        feature variant leaves the old entries alongside the new ones. They are
        unreachable rather than harmful, so count all shapes instead of sampling
        one key, which would report a mismatch when most lookups are fine.
        """
        counts: Dict[int, int] = {}
        seen = set()
        for table in self._tables:
            for key in table:
                if key in seen:
                    continue
                seen.add(key)
                counts[len(key)] = counts.get(len(key), 0) + 1
        return counts

    def prune_foreign_keys(self, expected_length: int) -> int:
        """Drop entries whose keys can never be looked up. Returns the count.

        Only worth doing on an artifact about to be shipped: unreachable rows
        cost lookup nothing but inflate the file the tournament has to load.
        """
        removed = 0
        for table in self._tables:
            for key in [k for k in table if len(k) != expected_length]:
                del table[key]
                removed += 1
        for key in [k for k in self._visits if len(k) != expected_length]:
            del self._visits[key]
        return removed

    def __len__(self) -> int:
        return self.n_states

    def __repr__(self) -> str:
        return (
            f"QTable(states={self.n_states}, double={self.double}, "
            f"max|Q|={self.max_abs_q:.3f})"
        )
