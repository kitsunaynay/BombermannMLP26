"""Hyperparameters for the PPO actor-critic agent.

Same configuration route as the Q-learning agent: defaults here, overridden by
``config.json`` beside this file, overridden in turn by ``AOT_PPO_*``
environment variables. Defaults are the *tournament* settings -- greedy action
selection, single CPU thread, no training machinery.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Tuple

AGENT_DIR = Path(__file__).resolve().parent
ENV_PREFIX = "AOT_PPO_"


def default_event_rewards() -> Dict[str, float]:
    """Event rewards, scaled so the game's own objective stays dominant."""
    return {
        "COIN_COLLECTED": 3.0,
        "KILLED_OPPONENT": 15.0,
        "CRATE_DESTROYED": 0.6,
        "COIN_FOUND": 0.4,
        "SURVIVED_ROUND": 3.0,
        # Careful: blowing yourself up fires BOTH of these. environment.py:252
        # adds KILLED_SELF and then line 264 adds GOT_KILLED to the same agent,
        # so the effective suicide penalty is their sum (-32), not -20.
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
class PPOConfig:
    # --- observation ---------------------------------------------------------
    observation: str = "global"  # global | ego
    ego_radius: int = 4  # ego window is (2r+1) squared

    # --- network -------------------------------------------------------------
    channels: Tuple[int, ...] = (32, 64, 64)
    hidden_dim: int = 256

    # --- PPO objective -------------------------------------------------------
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    entropy_coefficient_final: float = 0.001
    max_grad_norm: float = 0.5
    normalise_advantages: bool = True
    clip_value_loss: bool = True

    # --- optimisation --------------------------------------------------------
    learning_rate: float = 3e-4
    learning_rate_final: float = 1e-5
    anneal_schedules: bool = True
    update_epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 2048
    target_kl: float = 0.03  # early-stop an update that moves the policy too far

    # --- exploration and safety ---------------------------------------------
    safety_mode: str = "soft"  # none | soft | hard

    # Sample from the policy at evaluation instead of taking the argmax.
    #
    # Measured on Task 1 (coin-heaven, 25 seeds, 250k-step checkpoint):
    #
    #     sampling  31.56 coins  95% CI [29.6, 33.5]
    #     argmax    13.36 coins  95% CI [10.2, 16.7]
    #
    # Non-overlapping intervals, a 2.4x difference. The cause is that the policy
    # is still high-entropy (~0.95 nats of a possible 1.79): argmax discards most
    # of what it learned, and a deterministic policy in a near-deterministic
    # environment has no way out of a movement cycle -- the agent paces between
    # two tiles until the step limit. Sampling breaks those loops.
    #
    # Flip to True and re-measure once the policy is sharp; for an unconverged
    # one, sampling is strictly better.
    deterministic_eval: bool = False

    # --- reward shaping ------------------------------------------------------
    use_potential_shaping: bool = True
    use_custom_events: bool = True
    potential_coin: float = 0.12
    potential_crate: float = 0.04
    potential_danger: float = 0.5
    event_rewards: Dict[str, float] = field(default_factory=default_event_rewards)

    # --- augmentation --------------------------------------------------------
    symmetry_augmentation: bool = True

    # --- runtime -------------------------------------------------------------
    model_file: str = "policy.pt"
    device: str = "cpu"
    torch_threads: int = 1  # the brief guarantees exactly one tournament thread
    seed: int = 0

    # --- telemetry -----------------------------------------------------------
    checkpoint_every: int = 50
    metrics_file: str = ""

    # -----------------------------------------------------------------------
    @property
    def model_path(self) -> Path:
        """Resolved from ``__file__``, not the process cwd.

        The framework chdirs into the agent directory before each callback
        (agents.py:304) but the training tools call in from the repository root,
        and the brief warns that hand-written absolute paths break inside the
        grading container.
        """
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
    # Containers first: "Dict[str, float]" and "Tuple[int, ...]" also contain
    # the scalar type names.
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
