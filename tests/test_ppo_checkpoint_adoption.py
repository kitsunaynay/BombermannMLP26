"""A PPO checkpoint must dictate the config it was trained under."""

import logging
from types import SimpleNamespace

import pytest

from agent_code.attackontensor_ppo.callbacks import (
    ADOPTED_FROM_CHECKPOINT, _adopt_checkpoint_config,
)
from agent_code.attackontensor_ppo.config import PPOConfig


def _agent(**config_overrides):
    config = PPOConfig()
    for key, value in config_overrides.items():
        setattr(config, key, value)
    return SimpleNamespace(config=config, logger=logging.getLogger("test_ppo_adoption"))


def test_safety_mode_is_taken_from_the_checkpoint():
    agent = _agent(safety_mode="soft")
    _adopt_checkpoint_config(agent, {"config": {"safety_mode": "hard"}})
    assert agent.config.safety_mode == "hard"


def test_observation_is_taken_from_the_checkpoint():
    """It decides the input tensor shape, so a mismatch is fatal, not silent."""
    agent = _agent(observation="global")
    _adopt_checkpoint_config(agent, {"config": {"observation": "ego", "ego_radius": 3}})
    assert (agent.config.observation, agent.config.ego_radius) == ("ego", 3)


@pytest.mark.parametrize("field", ["device", "model_file", "seed", "learning_rate"])
def test_run_specific_fields_are_not_adopted(field):
    """The stored config records the training run; that must not leak into play."""
    agent = _agent()
    before = getattr(agent.config, field)
    _adopt_checkpoint_config(agent, {"config": {field: "POISON" if isinstance(before, str) else 12345}})
    assert getattr(agent.config, field) == before
    assert field not in ADOPTED_FROM_CHECKPOINT


def test_a_checkpoint_without_a_config_block_is_harmless():
    """Older checkpoints predate the stored config and must still load."""
    agent = _agent(safety_mode="soft")
    _adopt_checkpoint_config(agent, {})
    _adopt_checkpoint_config(agent, {"config": None})
    assert agent.config.safety_mode == "soft"


def test_agreement_is_left_alone():
    agent = _agent(safety_mode="hard")
    _adopt_checkpoint_config(agent, {"config": {"safety_mode": "hard"}})
    assert agent.config.safety_mode == "hard"


def test_the_real_checkpoint_carries_a_safety_mode():
    """Guards the producer as well as the consumer: training must store it."""
    import torch
    from agent_code.attackontensor_ppo.config import AGENT_DIR
    path = AGENT_DIR / "policy.pt"
    if not path.is_file():
        pytest.skip("no policy.pt in the agent directory")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "safety_mode" in (payload.get("config") or {}), \
        "checkpoint has no stored safety_mode, so adoption cannot protect it"


def test_survival_channels_are_inferred_from_the_network_block():
    """A 17-plane checkpoint written before the flag existed must still load."""
    from agent_code.attackontensor_ppo import tensorizer as T
    agent = _agent(survival_channels=False)
    _adopt_checkpoint_config(agent, {"config": {}, "network": {"in_channels": T.N_CHANNELS}})
    assert agent.config.survival_channels is True

    agent = _agent(survival_channels=True)
    _adopt_checkpoint_config(agent, {"config": {}, "network": {"in_channels": T.BASE_CHANNELS}})
    assert agent.config.survival_channels is False


def test_the_real_checkpoint_loads_into_the_live_network(caplog):
    """The shipped policy.pt must load under `setup` with no environment set.

    On 2026-09-16 the working tree's tensorizer produced 17 planes while the
    shipped policy.pt was trained on 13; `setup` caught the mismatch, logged a
    warning and played from *random weights*, and no test noticed.
    """
    import torch
    from agent_code.attackontensor_ppo import callbacks
    from agent_code.attackontensor_ppo.config import AGENT_DIR
    path = AGENT_DIR / "policy.pt"
    if not path.is_file():
        pytest.skip("no policy.pt in the agent directory")

    agent = SimpleNamespace(logger=logging.getLogger("test_ppo_real_load"), train=False)
    with caplog.at_level(logging.WARNING, logger="test_ppo_real_load"):
        callbacks.setup(agent)

    assert not any("fresh weights" in record.message for record in caplog.records), \
        "policy.pt did not load; the agent would play from random weights"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    live = agent.network.state_dict()
    for key, stored in payload["state_dict"].items():
        assert torch.equal(live[key], stored), f"{key} differs from the checkpoint"
