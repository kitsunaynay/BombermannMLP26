from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .kit.actions import N_ACTIONS

State = Tuple[int, ...]

FORMAT_VERSION = 1  # bump when the on-disk layout changes, so old checkpoints fail loudly


class QTable:
    # sparse dict keyed by feature tuple; not dense array or defaultdict (not picklable)
    # optional double q-learning (each table picks action, other scores it)
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
        # store variant/symmetry so key format never drifts from table
        self.metadata: Dict[str, object] = dict(metadata or {})
        self._tables: Tuple[Dict[State, np.ndarray], ...] = tuple(
            {} for _ in range(2 if double else 1)
        )
        self._visits: Dict[State, np.ndarray] = {}
        self._rng = np.random.default_rng(seed)

    def _row(self, table: Dict[State, np.ndarray], state: State) -> np.ndarray:
        # lazy allocation with optimistic init
        row = table.get(state)
        if row is None:
            row = np.full(self.n_actions, self.optimistic_init, dtype=np.float64)
            table[state] = row
        return row

    def q(self, state: State) -> np.ndarray:
        # average both tables in double q-learning
        if not self.double:
            return self._row(self._tables[0], state)
        return 0.5 * (self._row(self._tables[0], state) + self._row(self._tables[1], state))

    def visits(self, state: State) -> np.ndarray:
        # lazy allocation for visit counts
        counts = self._visits.get(state)
        if counts is None:
            counts = np.zeros(self.n_actions, dtype=np.int64)
            self._visits[state] = counts
        return counts

    def greedy(self, state: State, mask: Optional[np.ndarray] = None) -> int:
        # ties broken randomly to avoid deterministic loops in fresh tables
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
        # n-step td update; discount=0 for terminal (end_of_round)
        self.visits(state)[action] += 1

        # pick random table for update
        if self.double:
            index = int(self._rng.integers(2))
        else:
            index = 0
        table = self._tables[index]
        row = self._row(table, state)

        # bootstrap from next state
        bootstrap = 0.0
        if next_state is not None and discount != 0.0:
            if self.double:
                # pick action with this table; score with other table
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

    @property
    def n_states(self) -> int:
        seen = set()
        for table in self._tables:
            seen.update(table.keys())
        return len(seen)

    @property
    def max_abs_q(self) -> float:
        best = 0.0
        for table in self._tables:
            for row in table.values():
                row_max = float(np.abs(row).max()) if row.size else 0.0
                best = max(best, row_max)
        return best

    def is_finite(self) -> bool:
        return all(np.isfinite(row).all() for table in self._tables for row in table.values())

    def coverage(self) -> float:
        if not self._visits:
            return 0.0
        total = sum(int((counts > 0).sum()) for counts in self._visits.values())
        return total / (len(self._visits) * self.n_actions)

    def save(self, path: Path | str) -> None:
        # atomic save via temp file and rename
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
        # load with format version check
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
        lengths = self.key_lengths()
        if not lengths:
            return None
        return max(lengths, key=lambda length: lengths[length])

    def key_lengths(self) -> Dict[int, int]:
        # table may hold multiple key shapes if resumed under different variant
        # count all shapes since old-shape entries are harmless but unreachable
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
        # drop wrong-length keys; only needed at shipping to reduce file size
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
