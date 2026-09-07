"""Hyperparameters for the feature-based Q-learning agent.

The framework gives an agent no command line of its own; ``main.py`` decides
everything. Configuration arrives by two routes, both of which leave the
tournament-critical files untouched:

* ``config.json`` next to this file, if it exists;
* environment variables prefixed ``AOT_QL_`` (e.g. ``AOT_QL_EPSILON_START=0.5``),
  which is how ``tools/train_ql.py`` drives curriculum stages.

Environment variables win over the JSON file, which wins over these defaults.
The defaults are the *tournament* settings: no exploration, greedy play.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AGENT_DIR = Path(__file__).resolve().parent
ENV_PREFIX = "AOT_QL_"

#: Feature subsets. Ablating the feature set is a config change, not a code edit.
VARIANTS: Dict[str, Tuple[str, ...]] = {
    # Tasks 1-2: navigation and self-preservation. No opponents, no crate hunting.
    "compact": (
        "coin_dir",
        "escape_dir",
        "danger_here",
        "neighbour_up",
        "neighbour_right",
        "neighbour_down",
        "neighbour_left",
        "bomb_available",
        "escape_if_bomb",
    ),
    # Tasks 3-4: adds crate targeting and opponent awareness.
    "full": (
        "coin_dir",
        "crate_dir",
        "escape_dir",
        "danger_here",
        "neighbour_up",
        "neighbour_right",
        "neighbour_down",
        "neighbour_left",
        "bomb_available",
        "crates_adjacent",
        "opponent_near",
        "escape_if_bomb",
    ),
}


def default_event_rewards() -> Dict[str, float]:
    """Event rewards.

    Scaled so the game's own scoring (1 per coin, 5 per kill) stays dominant and
    the auxiliary terms only break ties. Auxiliary rewards are absent in official
    games, so they must not overwhelm the real objective.
    """
    return {
        # --- real game objectives -------------------------------------------
        "COIN_COLLECTED": 3.0,
        "KILLED_OPPONENT": 15.0,
        "CRATE_DESTROYED": 0.6,
        "COIN_FOUND": 0.4,
        "SURVIVED_ROUND": 3.0,
        # --- penalties -------------------------------------------------------
        # Careful: blowing yourself up fires BOTH of these. environment.py:252
        # adds KILLED_SELF and then line 264 adds GOT_KILLED to the same agent,
        # so the effective suicide penalty is their sum (-32), not -20.
        "KILLED_SELF": -20.0,
        "GOT_KILLED": -12.0,
        "INVALID_ACTION": -1.0,
        "WAITED": -0.15,
        # --- custom events (see rewards.py) ----------------------------------
        "MOVED_TOWARD_COIN": 0.35,
        "MOVED_AWAY_FROM_COIN": -0.4,
        "ESCAPED_DANGER": 1.0,
        "ENTERED_DANGER": -1.0,
        "WAITED_IN_DANGER": -1.5,
        "USEFUL_BOMB": 0.8,
        "USELESS_BOMB": -0.6,
        "SUICIDAL_BOMB": -3.0,
        "SURVIVED_STEP": 0.02,
    }


@dataclass
class QLConfig:
    # --- representation ------------------------------------------------------
    variant: str = "full"
    use_symmetry: bool = True

    # --- exploration ---------------------------------------------------------
    epsilon_start: float = 0.0  # tournament default: pure greedy
    epsilon_end: float = 0.0
    epsilon_decay_rounds: float = 2000.0

    # --- learning ------------------------------------------------------------
    gamma: float = 0.95
    alpha: float = 0.1
    alpha_decay: float = 0.0  # 0 disables the visit-count schedule
    alpha_min: float = 0.01
    n_step: int = 3
    double_q: bool = True
    optimistic_init: float = 0.0

    # --- safety filter -------------------------------------------------------
    safety_mode: str = "soft"  # none | soft | hard

    # --- reward shaping ------------------------------------------------------
    use_potential_shaping: bool = True
    use_custom_events: bool = True
    potential_coin: float = 0.12
    potential_crate: float = 0.04
    potential_danger: float = 0.5
    event_rewards: Dict[str, float] = field(default_factory=default_event_rewards)

    # --- evaluation ----------------------------------------------------------
    # Exploration rate used when NOT training. 0.0 is pure greedy, the
    # tournament default and the historical behaviour.
    #
    # Why this exists: a deterministic policy in a near-deterministic
    # environment can enter a movement limit cycle it cannot leave -- the agent
    # paces between two tiles until the step limit, scoring ~0 while surviving
    # 100% of rounds. Measured in 52 of 71 collapsed Task-2 snapshots, and the
    # PPO agent hit the identical failure in Phase 0 (argmax 13.36 vs sampling
    # 31.56 coins). A small amount of evaluation noise is the cheapest known
    # escape. Left at 0.0 until measured.
    eval_epsilon: float = 0.0

    # --- reproducibility -----------------------------------------------------
    # Seed for the agent's own RNG: epsilon-greedy draws, greedy tie-breaks and
    # the Double-Q table coin flip. Negative means "unseeded", which is the
    # tournament default -- there the framework decides the seeding and a fixed
    # one would make every game identical. tools/train_ql.py sets it per run so
    # that a seeded replicate is actually replayable.
    seed: int = -1

    # --- persistence and telemetry ------------------------------------------
    model_file: str = "q_table.pkl"
    checkpoint_every: int = 200
    metrics_file: str = ""  # empty disables the per-round CSV

    # Directory for periodic snapshots kept alongside the live model. With a
    # constant learning rate the Q-values track a moving target instead of
    # converging, so the table a run ends on is not reliably its best. Snapshots
    # let tools/train_ql.py select on held-out seeds rather than trusting the
    # last write. Empty disables snapshotting.
    snapshot_dir: str = "checkpoints"

    # -----------------------------------------------------------------------
    @property
    def feature_names(self) -> Tuple[str, ...]:
        try:
            return VARIANTS[self.variant]
        except KeyError:
            raise ValueError(
                f"Unknown feature variant {self.variant!r}; choose from {sorted(VARIANTS)}"
            ) from None

    @property
    def rng_seed(self) -> Optional[int]:
        """``seed`` as numpy wants it: ``None`` when unseeded."""
        return None if self.seed < 0 else int(self.seed)

    @property
    def model_path(self) -> Path:
        """Absolute path to the Q-table.

        Resolved from ``__file__`` rather than the process cwd. The framework
        chdirs into the agent directory before each callback (agents.py:304) but
        the training tools call in from the repository root. Hand-written
        absolute paths break inside the grading container.
        """
        return AGENT_DIR / self.model_file

    def epsilon(self, round_index: int) -> float:
        """Exponentially decayed exploration rate for a given round."""
        if self.epsilon_decay_rounds <= 0:
            return self.epsilon_end
        import math

        decayed = self.epsilon_start * math.exp(-round_index / self.epsilon_decay_rounds)
        return max(self.epsilon_end, decayed)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # -----------------------------------------------------------------------
    @classmethod
    def load(cls) -> "QLConfig":
        """Build a config from defaults, then ``config.json``, then the environment."""
        values: Dict[str, Any] = {}

        config_file = AGENT_DIR / "config.json"
        if config_file.is_file():
            values.update(json.loads(config_file.read_text()))

        values.update(cls._from_environment())

        # Reward overrides may arrive as a partial dict; merge onto the defaults.
        overrides = values.pop("event_rewards", None)
        instance = cls(**values)
        if overrides:
            merged = default_event_rewards()
            merged.update(overrides)
            instance.event_rewards = merged
        return instance

    @staticmethod
    def _from_environment() -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        typed = {f.name: f.type for f in fields(QLConfig)}

        for name, annotation in typed.items():
            raw = os.environ.get(ENV_PREFIX + name.upper())
            if raw is None:
                continue
            parsed[name] = _coerce(raw, annotation)
        return parsed


def _coerce(raw: str, annotation: Any) -> Any:
    """Turn an environment string into the type the dataclass field declares."""
    text = str(annotation)
    # Containers first: "Dict[str, float]" also contains "float".
    if "Dict" in text or "dict" in text:
        return json.loads(raw)
    if "bool" in text:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if "int" in text:
        return int(raw)
    if "float" in text:
        return float(raw)
    return raw
