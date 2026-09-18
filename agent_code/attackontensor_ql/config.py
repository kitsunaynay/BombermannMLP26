from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AGENT_DIR = Path(__file__).resolve().parent
ENV_PREFIX = "AOT_QL_"

# Feature subsets, swappable per curriculum stage without touching code.
VARIANTS: Dict[str, Tuple[str, ...]] = {
    # Navigation and self-preservation only: no opponents, no crate hunting.
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
    # Adds crate targeting and opponent awareness.
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
    # "full" plus dead-end / trap awareness, since rule_based_agent bombs from
    # dead ends without lookahead and treats our body as a wall.
    "trap": (
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
        "in_dead_end",
        "opponent_in_dead_end",
        "can_seal_opponent",
    ),
}


def default_event_rewards() -> Dict[str, float]:
    return {
        "COIN_COLLECTED": 3.0,
        "KILLED_OPPONENT": 15.0,
        "CRATE_DESTROYED": 0.6,
        "COIN_FOUND": 0.4,
        "SURVIVED_ROUND": 3.0,
        # Suicide fires both KILLED_SELF and GOT_KILLED for the same agent
        # (environment.py:252/264), so the real penalty is their sum, -32.
        "KILLED_SELF": -20.0,
        "GOT_KILLED": -12.0,
        "INVALID_ACTION": -1.0,
        "WAITED": -0.15,
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
    variant: str = "full"
    use_symmetry: bool = True

    epsilon_start: float = 0.0  # tournament default: pure greedy
    epsilon_end: float = 0.0
    epsilon_decay_rounds: float = 2000.0

    gamma: float = 0.95
    alpha: float = 0.1
    alpha_decay: float = 0.0  # 0 disables the visit-count schedule
    alpha_min: float = 0.01
    n_step: int = 3
    double_q: bool = True
    optimistic_init: float = 0.0

    safety_mode: str = "soft"  # none | soft | hard

    # When the "hard" escape search plans, treat every armed opponent as
    # bombing from its current tile. Off by default (kept as an ablation
    # arm); with opponents that actually bomb rather than just block,
    # suicide rate goes up noticeably without this on.
    opponent_bomb_lookahead: bool = False

    use_potential_shaping: bool = True
    use_custom_events: bool = True
    potential_coin: float = 0.12
    potential_crate: float = 0.04
    potential_danger: float = 0.5
    event_rewards: Dict[str, float] = field(default_factory=default_event_rewards)

    # Exploration rate used outside training. 0.0 (pure greedy) is the
    # tournament default. A fully greedy policy in a near-deterministic
    # environment can get stuck pacing between two tiles for the rest of the
    # round; a small nonzero value is the cheapest way out of that, but it
    # trades off against playing worse on average, so it stays at 0.0 unless
    # measured to help.
    eval_epsilon: float = 0.0

    # RNG seed for epsilon-greedy draws, tie-breaks and the Double-Q coin
    # flip. Negative means unseeded, the tournament default (the framework
    # controls seeding there, and a fixed seed would make every game
    # identical). tools/train_ql.py sets a real seed per run so a replicate
    # is actually replayable.
    seed: int = -1

    model_file: str = "q_table.pkl"
    checkpoint_every: int = 200
    metrics_file: str = ""  # empty disables the per-round CSV

    # Directory for periodic snapshots kept alongside the live table. With a
    # constant learning rate the Q-values track a moving target rather than
    # converging, so the table a run happens to end on isn't reliably its
    # best; snapshots let tools/train_ql.py pick the best one on held-out
    # seeds instead of trusting the last write. Empty disables snapshotting.
    snapshot_dir: str = "checkpoints"

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
        return None if self.seed < 0 else int(self.seed)

    @property
    def model_path(self) -> Path:
        # Resolved from __file__, not cwd: the framework chdirs into the
        # agent directory before each callback, but training tools run from
        # the repo root. An absolute path here would break in the grading
        # container.
        return AGENT_DIR / self.model_file

    def epsilon(self, round_index: int) -> float:
        if self.epsilon_decay_rounds <= 0:
            return self.epsilon_end
        import math

        decayed = self.epsilon_start * math.exp(-round_index / self.epsilon_decay_rounds)
        return max(self.epsilon_end, decayed)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls) -> "QLConfig":
        values: Dict[str, Any] = {}

        config_file = AGENT_DIR / "config.json"
        if config_file.is_file():
            values.update(json.loads(config_file.read_text()))

        values.update(cls._from_environment())

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
    text = str(annotation)
    # Container types checked first since "Dict[str, float]" also contains
    # the word "float".
    if "Dict" in text or "dict" in text:
        return json.loads(raw)
    if "bool" in text:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if "int" in text:
        return int(raw)
    if "float" in text:
        return float(raw)
    return raw
