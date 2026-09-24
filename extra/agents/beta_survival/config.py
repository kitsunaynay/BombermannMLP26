from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Tuple

AGENT_DIR = Path(__file__).resolve().parent
ENV_PREFIX = "AOT_PPO_"


def default_event_rewards() -> Dict[str, float]:
    # suicide is double-counted by environment (killed_self + got_killed = -32)
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
        # Kill-directed shaping, zero by default so it never changes the
        # shipped policy. Turn on with
        # --event-reward OPPONENT_ELIMINATED=3 --event-reward TRAPPED_OPPONENT=1
        "OPPONENT_ELIMINATED": 0.0,
        "TRAPPED_OPPONENT": 0.0,
    }


@dataclass
class PPOConfig:
    observation: str = "global"  # global | ego
    ego_radius: int = 4  # ego window is (2r+1) squared
    # Extra planes from kit.pathfind.survival_profile (escape duration/breadth/
    # blast margin/contested), see tensorizer.py. Off by default: the shipped
    # policy was trained on 13 planes. Stored in the checkpoint and adopted at
    # load time (see callbacks.py), so a 17-plane checkpoint just works.
    survival_channels: bool = True

    channels: Tuple[int, ...] = (32, 64, 64)
    hidden_dim: int = 256

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    entropy_coefficient_final: float = 0.001
    max_grad_norm: float = 0.5
    normalise_advantages: bool = True
    clip_value_loss: bool = True

    learning_rate: float = 3e-4
    learning_rate_final: float = 1e-5
    anneal_schedules: bool = True
    update_epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 2048
    # stop update early if policy diverges too much
    target_kl: float = 0.03

    safety_mode: str = "soft"  # none | soft | hard
    # bomb gate harshness: escape=any exit, robust=redundant exit
    bomb_gate: str = "escape"  # escape | robust

    # greedy at eval is stronger than sampling once trained
    deterministic_eval: bool = True

    use_potential_shaping: bool = True
    use_custom_events: bool = True
    potential_coin: float = 0.12
    potential_crate: float = 0.04
    potential_danger: float = 0.5
    event_rewards: Dict[str, float] = field(default_factory=default_event_rewards)

    symmetry_augmentation: bool = True

    model_file: str = "policy.pt"
    device: str = "cpu"
    torch_threads: int = 1  # one tournament thread
    seed: int = 0

    checkpoint_every: int = 50
    metrics_file: str = ""

    @property
    def model_path(self) -> Path:
        return AGENT_DIR / self.model_file

    @property
    def spatial_size(self) -> int:
        import settings as s

        if self.observation == "ego":
            return 2 * self.ego_radius + 1
        return s.COLS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls) -> "PPOConfig":
        values: Dict[str, Any] = {}

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
    def _from_environment() -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        for spec in fields(PPOConfig):
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
