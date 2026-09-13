"""PPO worker processes must see the CLI overrides.

`blib.factories.make_ppo_transform` and `make_ppo_reward_fn` run *inside* each
SubprocVecEnv worker and rebuild their own `PPOConfig.load()`. A CLI flag that
only mutates the parent's config object therefore never reaches the rollout.
That is not hypothetical: until 2026-09-09 `--safety-mode hard` did nothing to
PPO training, and a `hard` arm and a `soft` arm launched together produced
bit-identical learning curves.
"""

import argparse
import os

import pytest

from agent_code.attackontensor_ppo.config import PPOConfig
from blib.factories import make_ppo_transform
from tools.train_ppo import WORKER_VISIBLE, apply_overrides


def _args(**overrides):
    base = dict(observation=None, safety_mode=None, survival_channels=None, bomb_gate=None,
                learning_rate=None, entropy_coefficient=None, device="cpu", seed=0)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for field in WORKER_VISIBLE:
        monkeypatch.delenv(f"AOT_PPO_{field.upper()}", raising=False)


def test_safety_mode_is_exported_to_the_environment():
    config = apply_overrides(_args(safety_mode="hard"))
    assert config.safety_mode == "hard"
    assert os.environ["AOT_PPO_SAFETY_MODE"] == "hard"


def test_observation_is_exported_to_the_environment():
    apply_overrides(_args(observation="ego"))
    assert os.environ["AOT_PPO_OBSERVATION"] == "ego"


@pytest.mark.parametrize("mode", ["none", "soft", "hard"])
def test_a_worker_rebuilding_config_sees_the_override(mode):
    """The actual failure: what a worker's own PPOConfig.load() ends up with."""
    apply_overrides(_args(safety_mode=mode))
    assert PPOConfig.load().safety_mode == mode


def test_the_transform_a_worker_builds_reflects_the_override(monkeypatch):
    """One step further: the closure the worker actually calls."""
    apply_overrides(_args(safety_mode="hard"))
    make_ppo_transform()  # must not raise, and loads config at build time
    assert PPOConfig.load().safety_mode == "hard"


def test_learner_only_settings_are_not_exported():
    """Exporting these would let a stale env var outlive the run that set it."""
    apply_overrides(_args(learning_rate=1e-5, entropy_coefficient=0.5))
    assert "AOT_PPO_LEARNING_RATE" not in os.environ
    assert "AOT_PPO_ENTROPY_COEFFICIENT" not in os.environ


def test_defaults_are_exported_too_so_a_stale_var_cannot_leak_in(monkeypatch):
    """A safety_mode left over from an earlier run must not silently apply."""
    monkeypatch.setenv("AOT_PPO_SAFETY_MODE", "hard")
    config = apply_overrides(_args())          # no --safety-mode given
    assert os.environ["AOT_PPO_SAFETY_MODE"] == config.safety_mode


def test_survival_channels_reach_the_workers():
    """The tensorizer runs inside each worker, so the plane count must travel."""
    apply_overrides(_args(survival_channels=True))
    assert os.environ["AOT_PPO_SURVIVAL_CHANNELS"] == "True"
    assert PPOConfig.load().survival_channels is True
    apply_overrides(_args(survival_channels=False))
    assert PPOConfig.load().survival_channels is False


def test_bomb_gate_reaches_the_workers():
    apply_overrides(_args(bomb_gate="robust"))
    assert os.environ["AOT_PPO_BOMB_GATE"] == "robust"
    assert PPOConfig.load().bomb_gate == "robust"
    apply_overrides(_args(bomb_gate="escape"))
    assert PPOConfig.load().bomb_gate == "escape"
