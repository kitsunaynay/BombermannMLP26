from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parent
ENV_PREFIX = "AOT_DQN_"



def default_event_rewards() -> dict[str, float]:
    return {
        "COIN_COLLECTED": 3.0,
        "KILLED_OPPONENT": 15.0,
        "CRATE_DESTROYED": 0.6,
        "COIN_FOUND": 0.4,
        "SURVIVED_ROUND": 3.0,
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
        "OPPONENT_ELIMINATED": 0.0,
        "TRAPPED_OPPONENT": 0.0,
    }


@dataclass
class DQNConfig:
    # Observation / network
    observation: str = "global"
    ego_radius: int = 4
    spatial_size: int = 17
    channels: tuple[int, ...] = (32, 64, 64)
    hidden_dim: int = 256
    survival_channels: bool = False

    # Safety
    safety_mode: str = "hard"
    bomb_gate: str = "escape"

    # Reward shaping
    use_potential_shaping: bool = True
    use_custom_events: bool = True
    potential_coin: float = 0.12
    potential_crate: float = 0.04
    potential_danger: float = 0.5
    event_rewards: dict[str, float] = field(default_factory=default_event_rewards)

    # DQN
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 128
    replay_capacity: int = 30_000
    replay_warmup: int = 5_000
    train_frequency: int = 4
    target_update_interval: int = 5_000
    max_grad_norm: float = 10.0

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000

    # Runtime
    device: str = "cpu"
    torch_threads: int = 1
    seed: int = 0
    deterministic_eval: bool = True

    # Checkpoint
    model_file: str = "dqn.pt"

    @property
    def model_path(self) -> Path:
        return AGENT_DIR / self.model_file

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls) -> "DQNConfig":
        values: dict[str, Any] = {}

        config_file = AGENT_DIR / "config.json"
        if config_file.is_file():
            values.update(json.loads(config_file.read_text()))

        values.update(cls._from_environment())

        overrides = values.pop("event_rewards", None)

        if "channels" in values:
            values["channels"] = tuple(values["channels"])

        instance = cls(**values)

        if overrides:
            merged = default_event_rewards()
            merged.update(overrides)
            instance.event_rewards = merged

        return instance

    @staticmethod
    def _from_environment() -> dict[str, Any]:
        parsed: dict[str, Any] = {}

        for spec in fields(DQNConfig):
            raw = os.environ.get(ENV_PREFIX + spec.name.upper())

            if raw is None:
                continue

            parsed[spec.name] = _coerce(raw, spec.type)

        return parsed

def _coerce(raw: str, annotation: Any) -> Any:
    text = str(annotation)

    if "Dict" in text or "dict" in text:
        return json.loads(raw)

    if "Tuple" in text or "tuple" in text:
        return tuple(json.loads(raw))

    if "bool" in text:
        return raw.strip().lower() in ("1", "true", "yes", "on")

    if "int" in text:
        return int(raw)

    if "float" in text:
        return float(raw)

    return raw